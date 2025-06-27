import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributions as dists
from matplotlib.gridspec import GridSpec
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

MNIST_SIZE = 28
HIDDEN_DIM = 400
LATENT_DIM = 50
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.5
NUM_EPOCHS = 2
LIST_OF_KS = [10]
LOG_INTERVAL = 10  # Log every 10 batches
SAVE_INTERVAL = 5  # Save images every 5 epochs


class Binarized_MNIST(datasets.MNIST):    
    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(Binarized_MNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return dists.Bernoulli(img).sample().type(torch.float32)
    
n_samples = 7
binarized_MNIST = Binarized_MNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor())

class VAE(nn.Module):
    
    def __init__(self, k):
        super(VAE, self).__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MNIST_SIZE**2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, MNIST_SIZE**2),
            nn.Sigmoid()
        )
        return
    
    def compute_loss(self, x, k=None):
        if not k:
            k = self.k
        [x_tilde, z, mu_z, log_var_z] = self.forward(x, k)
        # upsample x 
        x_s = x.unsqueeze(1).repeat(1, k, 1, 1, 1)
        # compute negative log-likelihood
        NLL = nn.functional.binary_cross_entropy(x_tilde, x_s, reduction='none').sum(dim=(2, 3, 4)).mean()
        # copmute kl divergence
        KL_Div = 0.5 * (mu_z**2 + log_var_z.exp() - log_var_z - 1).sum(1).mean()
        # compute loss
        loss = NLL + KL_Div
        return loss
    
    def forward(self, x, k=None):
        """feed image (x) through VAE
        
        Args:
            x (torch tensor): input [batch, img_channels, img_dim, img_dim]
        
        Returns:
            x_tilde (torch tensor): [batch, k, img_channels, img_dim, img_dim]
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]
            mu_z (torch tensor): mean latent space [batch, LATENT_DIM]
            log_var_z (torch tensor): log var latent space [batch, LATENT_DIM] 
        """
        if not k:
            k = self.k
        z, mu_z, log_var_z = self.encode(x, k)
        x_tilde = self.decode(z, k)
        return [x_tilde, z, mu_z, log_var_z]
    
    def encode(self, x, k):
        """computes the approximated posterior distribution parameters and 
        samples from this distribution
        
        Args:
            x (torch tensor): input [batch, img_channels, img_dim, img_dim]
            
        Returns:
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]
            mu_E (torch tensor): mean latent space [batch, LATENT_DIM]
            log_var_E (torch tensor): log var latent space [batch, LATENT_DIM]
        """
        # Get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)

        # Sample noise variable for each batch and sample: shape [batch, k, LATENT_DIM]
        epsilon = torch.randn(mu_E.size(0), k, mu_E.size(1), device=mu_E.device)

        # Expand mu and log_var for broadcasting without repeating
        mu_E = mu_E.unsqueeze(1)  # shape [batch, 1, LATENT_DIM]
        log_var_E = log_var_E.unsqueeze(1)  # shape [batch, 1, LATENT_DIM]

        # Reparameterization trick with broadcasting
        z = mu_E + torch.exp(0.5 * log_var_E) * epsilon

        return z, mu_E.squeeze(1), log_var_E.squeeze(1)
    
    def decode(self, z, k):
        """computes the Bernoulli mean of p(x|z)
        note that linear automatically parallelizes computation
        
        Args:
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]
        
        Returns:
            x_tilde (torch tensor): [batch, k, img_channels, img_dim, img_dim]
        """
        # get decoder distribution parameters 
        x_tilde = self.decoder(z)  # [batch*samples, MNIST_SIZE**2]      
        # reshape into [batch, samples, 1, MNIST_SIZE, MNIST_SIZE] (input shape)
        x_tilde = x_tilde.view(-1, k, 1, MNIST_SIZE, MNIST_SIZE)
        return x_tilde
    
    def create_latent_traversal(self, image_batch, n_pert, pert_min_max=2, n_latents=5):
        device = image_batch.device
        # initialize images of latent traversal
        images = torch.zeros(n_latents, n_pert, *image_batch.shape[1::])
        # select the latent_dims with lowest variance (most informative)
        [x_tilde, z, mu_z, log_var_z] = self.forward(image_batch)
        i_lats = log_var_z.mean(axis=0).sort()[1][:n_latents]
        # sweep for latent traversal
        sweep = np.linspace(-pert_min_max, pert_min_max, n_pert)
        # take first image and encode
        [z, mu_E, log_var_E] = self.encode(image_batch[0:1], k=1)
        for latent_dim, i_lat in enumerate(i_lats):
            for pertubation_dim, z_replaced in enumerate(sweep):
                z_new = z.detach().clone()
                z_new[0][0][i_lat] = z_replaced

                img_rec = self.decode(z_new.to(device), k=1).squeeze(0)
                img_rec = img_rec[0].clamp(0, 1).cpu()

                images[latent_dim][pertubation_dim] = img_rec
        return images
    
    def compute_marginal_log_likelihood(self, x, k=None):
        """computes the marginal log-likelihood in which the sampling
        distribution is exchanged to q_{\phi} (z|x),
        this function can also be used for the IWAE loss computation 
        
        Args:
            x (torch tensor): images [batch, img_channels, img_dim, img_dim]
            
        Returns:
            log_marginal_likelihood (torch tensor): scalar
            log_w (torch tensor): unnormalized log importance weights [batch, k]
        """
        if not k:
            k = self.k
        x_tilde, z, mu_z, log_var_z = self.forward(x, k)

        # upsample mu_z, std_z, x_s
        mu_z = mu_z.unsqueeze(1)  # [batch, 1, latent_dim]
        log_var_z = log_var_z.unsqueeze(1)  # [batch, 1, latent_dim]
        std_z = torch.exp(0.5 * log_var_z)

        x = x.unsqueeze(1)
        # compute logarithmic unnormalized importance weights [batch, k]       
        log_p_x_g_z = dists.Bernoulli(x_tilde).log_prob(x).sum(axis=(2, 3, 4))
        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(2)
        log_q_z_g_x = dists.Normal(mu_z, std_z).log_prob(z).sum(2)
        log_w = log_p_x_g_z + log_prior_z - log_q_z_g_x
        # compute marginal log-likelihood        
        log_marginal_likelihood = (torch.logsumexp(log_w, 1) -  np.log(k)).mean()
        return log_marginal_likelihood, log_w
    
class IWAE(VAE):
    
    def __init__(self, k):
        super(IWAE, self).__init__(k)
        return
    
    def compute_loss(self, x, k=None, mode='original'):
        if not k:
            k = self.k
        # compute unnormalized importance weights in log_units
        log_likelihood, log_w = self.compute_marginal_log_likelihood(x, k)
        # loss computation (several ways possible)
        if mode == 'original':
            ####################### ORIGINAL IMPLEMENTAION #######################
            # numerical stability (found in original implementation)
            # Stabilize weights
            log_w_max, _ = torch.max(log_w, dim=1, keepdim=True)
            w_tilde = torch.softmax(log_w - log_w_max, dim=1).detach()
            # Compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(dim=1).mean()
        elif mode == 'normalized weights':
            ######################## LOG-NORMALIZED TRICK ########################
            # copmute normalized importance weights (no gradient)
            log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w_tilde = log_w_tilde.exp().detach()
            # compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(1).mean()
        elif mode == 'fast':
            ########################## SIMPLE AND FAST ###########################
            loss = -log_likelihood
        return loss

def helper_tensorboard_loss_batch(writer, loss, scheduler, global_step, batch_start_time, model):
    writer.add_scalar(f'{model} - Training Loss Batch (NLL + KL Div)', loss.item(), global_step)
    writer.add_scalar(f'{model} - Learning Rate Batch', scheduler.get_last_lr()[0], global_step)
    writer.add_scalar(f'{model} - Training Time Batch (Minutes)', (time.time() - batch_start_time) / 60, global_step)

def helper_tensorboard_loss_total(writer, loss_vae, loss_iwae, epoch, total_start_time, total_vae_time, total_iwae_time):
    writer.add_scalar('VAE - Training Loss Total Epoch (NLL + KL Div)', loss_vae, epoch)
    writer.add_scalar('IWAE - Training Loss Total Epoch (NLL + KL Div)', loss_iwae, epoch)
    writer.add_scalar('VAE - Training Time Epoch (Minutes)', total_vae_time / 60, epoch)
    writer.add_scalar('IWAE - Training Time Epoch (Minutes)', total_iwae_time / 60, epoch)
    writer.add_scalar('Training Time Total (Minutes)', (time.time() - total_start_time) / 60, epoch)

def train(dataset, vae_model, iwae_model, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/VAE_experiment_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0)
    vae_model.to(device)
    iwae_model.to(device)
    
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_iwae = torch.optim.Adam(iwae_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    scheduler_iwae = torch.optim.lr_scheduler.StepLR(optimizer_iwae, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Fixed test batch for consistent visualization
    fixed_test_batch = next(iter(data_loader)).to(device)
    
    # Log model architecture
    writer.add_graph(vae_model, fixed_test_batch[:1])

    # Time tracking variables
    epoch_times = []
    total_time_start = time.time()

    for epoch in range(1, num_epochs + 1):
        total_iwae_loss = 0
        total_vae_loss = 0
        total_iwae_time = 0
        total_vae_time = 0
        epoch_start_time = time.time()
        iwae_step = 0
        vae_step = 0

        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            if batch_idx % LOG_INTERVAL == 0:
                batch_start_time_iwae = time.time()

            # IWAE update
            optimizer_iwae.zero_grad()
            loss_iwae = iwae_model.compute_loss(x)
            loss_iwae.backward()
            optimizer_iwae.step()

            total_iwae_loss += loss_iwae.item()
            total_iwae_time += time.time() - batch_start_time_iwae
            
            if batch_idx % LOG_INTERVAL == 0:
                helper_tensorboard_loss_batch(writer, loss_iwae, scheduler_iwae, iwae_step, batch_start_time_iwae, 'IWAE')
                iwae_step += 1

                batch_start_time_vae = time.time()

            # VAE update
            optimizer_vae.zero_grad()
            loss_vae = vae_model.compute_loss(x)
            loss_vae.backward()
            optimizer_vae.step()

            total_vae_loss += loss_vae.item()
            total_vae_time += time.time() - batch_start_time_vae

            if batch_idx % LOG_INTERVAL == 0:
                helper_tensorboard_loss_batch(writer, loss_vae, scheduler_vae, vae_step, batch_start_time_vae, 'VAE')
                vae_step += 1            

        print("Epoch " + str(epoch) + " from " + str(num_epochs) )

        # step schedulers
        scheduler_iwae.step()
        scheduler_vae.step()
        epoch_times.append(time.time() - epoch_start_time)

        helper_tensorboard_loss_total(writer, total_vae_loss, total_iwae_loss, epoch, total_time_start, total_vae_time, total_iwae_time)

    return vae_model, iwae_model

binarized_MNIST = Binarized_MNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor())

for k in LIST_OF_KS:
    vae_model = VAE(k)
    iwae_model = IWAE(k)
    trained_vae, trained_iwae = train(binarized_MNIST, vae_model, iwae_model, NUM_EPOCHS)
    torch.save(trained_vae, f'./results/trained_vae_{k}.pth')
    torch.save(trained_iwae, f'./results/trained_iwae_{k}.pth')