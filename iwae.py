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
from scipy import stats

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
SAVE_INTERVAL = 1  # Save images every 5 epochs
ACTIVE_LATENT_DIM_THRESHOLD = 1e-2


class Binarized_MNIST(datasets.MNIST):    
    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(Binarized_MNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return dists.Bernoulli(img).sample().type(torch.float32)
    
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
    
    def sample(self, n_samples, device):
        """Generate samples from prior"""
        with torch.no_grad():
            z = torch.randn(n_samples, 1, LATENT_DIM).to(device)
            samples = self.decode(z, 1).squeeze(1)
            return samples
    
    def reconstruct(self, x, k=1):
        """Reconstruct input images"""
        with torch.no_grad():
            x_recon, _, _, _ = self.forward(x, k)
            return x_recon.mean(dim=1)  # Average over k samples
    
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

def helper_tensorboard_loss_batch(writer, loss, total_loss, global_step, batch_start_time, epoch):
    writer.add_scalars(f'BATCH_LOSS/Training Loss (NLL + KL Div)', {f'Epoch_{epoch+1}': loss.item()}, global_step)
    avg_loss = np.sum(total_loss) / LOG_INTERVAL
    writer.add_scalars('BATCH_LOSS/Running Avg Training Loss (NLL + KL Div)', {f'Epoch_{epoch+1}': avg_loss}, global_step)
    writer.add_scalars(f'BATCH_TIME/Training Time (Minutes)', {f'Epoch_{epoch+1}': (time.time() - batch_start_time) / 60}, global_step)
    #writer.add_scalar(f'BATCH/DIMENSIONS/Active Latent Dimensions', len(active_dims), global_step)

def helper_tensorboard_loss_total(writer, loss_vae, epoch, total_vae_time, train_loader, active_dims):
    sum_loss = np.sum(loss_vae)
    writer.add_scalar('EPOCH_LOSS/Training Loss (NLL + KL Div)', sum_loss, epoch)
    writer.add_scalar('EPOCH_LOSS/Avg. Training Loss (NLL + KL Div)', sum_loss / len(train_loader), epoch)
    writer.add_scalar('EPOCH_LOSS/Standard Error Loss (NLL + KL Div)', stats.sem(loss_vae), epoch)
    writer.add_scalar('EPOCH_TIME/Training Time (Minutes)', total_vae_time / 60, epoch)
    writer.add_scalar('EPOCH_DIM/Active Latent Dimensions', len(active_dims), epoch)

def create_comparison_image(original, reconstructed, n_images=8):
    """Create side-by-side comparison of original and reconstructed images"""
    # Take first n_images from batch
    orig = original[:n_images]
    recon = reconstructed[:n_images]
    
    # Interleave original and reconstructed images
    comparison = torch.zeros(2 * n_images, *orig.shape[1:])
    comparison[0::2] = orig
    comparison[1::2] = recon
    
    return comparison

def compute_activity_paper_method(model, data_loader, device, threshold, n_batches=None):
    """
    Compute latent activity using the paper's method:
    A_u = Cov_x E_{u~q(u|x)}[u]
    
    Args:
        model: VAE or IWAE model
        data_loader: DataLoader for the dataset
        device: torch device
        n_batches: Number of batches to use (None = use all)
    
    Returns:
        activity_scores: torch.Tensor of shape [latent_dim] with activity scores
        active_dims: torch.Tensor with indices of active dimensions (A_u > 1e-2)
        posterior_means: torch.Tensor of all posterior means for further analysis
    """
    model.eval()
    all_posterior_means = []
    
    with torch.no_grad():
        batch_count = 0
        for x in data_loader:
            if n_batches is not None and batch_count >= n_batches:
                break
                
            x = x.to(device)
            # Get posterior means (mu_z from encoder)
            _, mu_z, _ = model.encode(x, k=1)  # k=1 since we only need the mean
            all_posterior_means.append(mu_z.cpu())
            batch_count += 1
    
    # Concatenate all posterior means: [total_samples, latent_dim]
    all_posterior_means = torch.cat(all_posterior_means, dim=0)
    
    # Compute covariance across samples (data points) for each latent dimension
    # This is equivalent to computing the variance of the posterior means
    activity_scores = torch.var(all_posterior_means, dim=0, unbiased=True)
    
    # Define active dimensions using paper's threshold
    active_dims = torch.where(activity_scores > threshold)[0]
    
    return activity_scores, active_dims, all_posterior_means

def train_general(data_loader, data_loader_test, model, lr, num_epochs, K, writer, optimizer, fixed_tensorboard_batch, scheduler):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model.to(device)
    writer.add_graph(vae_model, fixed_tensorboard_batch[:1])

    epoch_times = []

    for epoch in range(1, num_epochs + 1):
        total_loss = []
        total_time = 0
        epoch_start_time = time.time()

        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)

            batch_start_time = time.time()

            # IWAE update
            optimizer.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_time += time.time() - batch_start_time
            
            if batch_idx % LOG_INTERVAL == 0:
                helper_tensorboard_loss_batch(writer, loss, total_loss, batch_idx, batch_start_time, epoch)

        print("Epoch " + str(epoch) + " from " + str(num_epochs) )

        # step schedulers
        scheduler.step()
        epoch_times.append(time.time() - epoch_start_time)

        activity_scores, active_dims, all_posterior_means = compute_activity_paper_method(model, data_loader, device, ACTIVE_LATENT_DIM_THRESHOLD, 5)
        helper_tensorboard_loss_total(writer, total_loss, epoch, total_time, data_loader, active_dims)

        # Log images every SAVE_INTERVAL epochs
        if epoch % SAVE_INTERVAL == 0:
            with torch.no_grad():
                # Reconstructions
                reconstructions = model.reconstruct(fixed_tensorboard_batch[:16])
                comparison = create_comparison_image(fixed_tensorboard_batch[:8], reconstructions[:8])
                
                writer.add_images('Reconstructions/Original_vs_Reconstructed', 
                                comparison, epoch, dataformats='NCHW')
                
                # Generated samples
                generated_samples = model.sample(16, device)
                writer.add_images('Generated_Samples', generated_samples, epoch, dataformats='NCHW')

                # Latent traversal (optional - can be expensive)
                if epoch % (SAVE_INTERVAL * 2) == 0:
                    traversal_images = vae_model.create_latent_traversal(
                        fixed_tensorboard_batch[:1], n_pert=10, n_latents=5
                    )
                    # Reshape for tensorboard: [n_latents*n_pert, C, H, W]
                    traversal_flat = traversal_images.view(-1, 1, 28, 28)
                    writer.add_images('Latent_Traversal', traversal_flat, epoch, dataformats='NCHW')

    hparams = {
        'learning rate': lr,
        'epochs': num_epochs,
        'k': K
    }
    
    writer.add_hparams(hparams, {'NLL': np.mean(total_loss)})
    return model

def train_paper(dataset, dataset_test, vae_model, iwae_model, lr, num_epochs, K):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/VAE_experiment_{timestamp}_{lr}_{num_epochs}_{K}"
    log_dir2 = f"runs/IWAE_experiment_{timestamp}_{lr}_{num_epochs}_{K}"
    writer_vae = SummaryWriter(log_dir)
    writer_iwae = SummaryWriter(log_dir2)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    optimizer_iwae = torch.optim.Adam(iwae_model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    scheduler_iwae = torch.optim.lr_scheduler.StepLR(optimizer_iwae, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Fixed test batch for consistent visualization
    fixed_test_batch = next(iter(data_loader_test)).to(device)
    
    vae_model = train_general(data_loader, data_loader_test, vae_model, lr, num_epochs, K, writer_vae, optimizer_vae, fixed_test_batch, scheduler_vae)
    #iwae_model = train_general(data_loader, data_loader_test, iwae_model, lr, num_epochs, K, writer_iwae, optimizer_iwae, fixed_test_batch, scheduler_iwae)
    
    torch.save(trained_vae, f'./results/trained_vae_{timestamp}_{LEARNING_RATE}_{NUM_EPOCHS}_{k}.pth')
    torch.save(trained_iwae, f'./results/trained_iwae_{timestamp}_{LEARNING_RATE}_{NUM_EPOCHS}_{k}.pth')
    return vae_model, iwae_model, timestamp

#n_samples = 7
binarized_MNIST = Binarized_MNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor())
binarized_MNIST_Test = Binarized_MNIST('./data', train=False, download=True,
                                  transform=transforms.ToTensor())
for k in LIST_OF_KS:
    vae_model = VAE(k)
    iwae_model = IWAE(k)
    trained_vae, trained_iwae, timestamp = train_paper(binarized_MNIST, binarized_MNIST_Test, vae_model, iwae_model, LEARNING_RATE, NUM_EPOCHS, k)
    