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
from tqdm import tqdm
import random
import os

MNIST_SIZE = 28
HIDDEN_DIM = 200
LATENT_DIM = 50
BATCH_SIZE = 20
LEARNING_RATE = 1e-3
#WEIGHT_DECAY = 1e-6
#SCHEDULER_STEP_SIZE = 50
#SCHEDULER_GAMMA = 0.5
NUM_EPOCHS = 80#150
LIST_OF_KS = [1,5] #1,5,50
LOG_INTERVAL = 100  # Log every 100 batches (from approx 3.000)
SAVE_INTERVAL = 10  # Save images every 10. epoch
ACTIVE_LATENT_DIM_THRESHOLD = 1e-2
#NORMALISATION =
INITIALISATION = ['XavierUni', 'XavierNormal', 'KaimingUni', 'KaimingNormal', 'TruncNormal']
SEEDS = [135,630,924,10,32]

def set_seed(seed=42):
    """
    Set random seed for reproducible results across PyTorch, NumPy, and Python (CPU only)
    
    Args:
        seed (int): Random seed value
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set environment variable for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

class Binarized_MNIST(datasets.MNIST):    
    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(Binarized_MNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return dists.Bernoulli(img).sample().type(torch.float32)

def bias_init(m):
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m

def weights_init(m, method = 'XavierUni'):
    if method == 'XavierUni':
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            bias_init(m)
    elif method == 'XavierNormal':
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            bias_init(m)
    elif method == 'KaimingUni':
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            bias_init(m)
    elif method == 'KaimingNormal':
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            bias_init(m)
    elif method == 'TruncNormal':
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight)
            bias_init(m)

def avg_loss_SError(total_loss, total_nll, total_kl):
    avg_loss = np.mean(total_loss)
    avg_loss_SE = stats.sem(total_loss)
    avg_nll = np.mean(total_nll)
    avg_nll_SE = stats.sem(total_nll)
    avg_kl = np.mean(total_kl)
    avg_kl_SE = stats.sem(total_kl)
    return avg_loss, avg_loss_SE, avg_nll, avg_nll_SE, avg_kl, avg_kl_SE

class VAE(nn.Module):
    def __init__(self, k, m = 'XavierUni'):
        super(VAE, self).__init__()
        self.k = k
        self.model_type = 'VAE'
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MNIST_SIZE**2, HIDDEN_DIM),  
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),     
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),    
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),     
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, MNIST_SIZE**2),  
            nn.Sigmoid()
        )
        
        # Weight initialization as paper
        self.encoder.apply(lambda module: weights_init(module, method=m))
        self.decoder.apply(lambda module: weights_init(module, method=m))
    
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
        return loss, NLL, KL_Div
    
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
        return log_marginal_likelihood, log_w, log_p_x_g_z, log_q_z_g_x, log_prior_z
    
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

    def __init__(self, k, m = 'XavierUni'):
        super(IWAE, self).__init__(k, m)
        self.model_type = 'IWAE'
    
    def compute_loss(self, x, k=None, mode='original'):
        if not k:
            k = self.k
        # compute unnormalized importance weights in log_units
        log_likelihood, log_w, log_p_x_g_z, log_q_z_g_x, log_prior_z = self.compute_marginal_log_likelihood(x, k)
        # loss computation (several ways possible)
        if mode == 'original':
            ####################### ORIGINAL IMPLEMENTAION #######################
            # numerical stability (found in original implementation)
            log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
            # compute normalized importance weights (no gradient)
            w = log_w_minus_max.exp()
            w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
            # compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(1).mean()
        elif mode == 'normalized weights':
            ######################## LOG-NORMALIZED TRICK ########################
            # copmute normalized importance weights (no gradient)
            log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w_tilde = log_w_tilde.exp().detach()
            # compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(1).mean()

        # Calculate reconstruction loss: weighted average of negative log-likelihood
        recon_loss = -(w_tilde * log_p_x_g_z).sum(1).mean()

        # Calculate KL divergence: weighted average of (log q(z|x) - log p(z))
        kl_divergence = (w_tilde * (log_q_z_g_x - log_prior_z)).sum(1).mean()
        return loss, recon_loss, kl_divergence

def evaluate_test_loss(model, test_loader, device, k=None):
    model.eval()
    total_loss = []
    total_nll = []
    total_kl = []
    num_batches = 0
    num_samples = 0
    total_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device (GPU/CPU)
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)  # if batch contains (data, labels)
            else:
                x = batch.to(device)     # if batch is just data
            
            # Compute loss using your existing method
            loss, NLL, KL_Div = model.compute_loss(x, k=k)
            
            # Accumulate losses
            total_loss.append(loss.item())
            total_nll.append(NLL.item())
            total_kl.append(KL_Div.item())
            num_batches += 1
            num_samples += x.size(0)
    
    total_time = total_time - time.time()
    
    return total_loss, total_nll, total_kl, total_time, num_samples

def log_activity_scores(writer, activity_scores, epoch, filename_hyperparameters, test = False):
    """
    Log activity scores as a bar plot to TensorBoard.
    
    Args:
        writer: SummaryWriter instance.
        activity_scores: Tensor of shape [latent_dim].
        epoch: Current training epoch.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.titlesize': 12,
        'axes.titleweight': 'normal'
    })

    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(activity_scores)), activity_scores.cpu().numpy(), color='black')

    plt.xlabel('Latent Dimension')
    plt.ylabel('Activity Score')
    plt.title(f'Activity Scores at Epoch {epoch}')
    plt.grid(axis='y')  # Add horizontal grid lines only
    plt.tight_layout()

    if test:
        temp_name = 'test'
    else:
        temp_name = 'training'
    
    plt.savefig(f'./results/PDF/{filename_hyperparameters}_dim_acitivity_scores_{temp_name}_epoch{epoch}.pdf', format='pdf')
    writer.add_figure('EPOCH_DIM/activity_scores_{temp_name}', plt.gcf(), global_step=epoch)
    torch.save(activity_scores,  f'./results/torch/{filename_hyperparameters}_latent_matrix_{temp_name}_epoch{epoch}.pt')

def helper_tensorboard_loss_batch(writer, loss, NLL, KL_div, total_loss, total_NLL, total_kl_div, batch_idx, batch_start_time, epoch):
    avg_loss, avg_loss_SE, avg_nll, avg_nll_SE, avg_kl, avg_kl_SE = avg_loss_SError(total_loss, total_NLL, total_kl_div)

    writer.add_scalars(f'BATCH_LOSS/Training Loss (NLL + KL Div)', {f'Epoch_{epoch}': loss.item()}, batch_idx)
    writer.add_scalars(f'BATCH_LOSS/Training Loss (NLL)', {f'Epoch_{epoch}': NLL}, batch_idx)
    writer.add_scalars(f'BATCH_LOSS/Training Loss (KL_div)', {f'Epoch_{epoch}': KL_div}, batch_idx)
    
    writer.add_scalars('BATCH_AVG_LOSS/Avg. Training Loss (NLL + KL Div)', {f'Epoch_{epoch}': avg_loss}, batch_idx)
    writer.add_scalars('BATCH_AVG_LOSS/Avg. Training Loss (NLL)', {f'Epoch_{epoch}': avg_nll}, batch_idx)
    writer.add_scalars('BATCH_AVG_LOSS/Avg. Training Loss (KL Div)', {f'Epoch_{epoch}': avg_kl}, batch_idx)

    writer.add_scalars('BATCH_AVG_LOSS/Standard Error Training Loss (NLL + KL Div)', {f'Epoch_{epoch}': avg_loss_SE}, batch_idx)
    writer.add_scalars('BATCH_AVG_LOSS/Standard Error Training Loss (NLL)', {f'Epoch_{epoch}': avg_nll_SE}, batch_idx)
    writer.add_scalars('BATCH_AVG_LOSS/Standard Error Training Loss (KL Div)', {f'Epoch_{epoch}': avg_kl_SE}, batch_idx)

    writer.add_scalars(f'BATCH_TIME/Training Time (Minutes)', {f'Epoch_{epoch}': (time.time() - batch_start_time) / 60}, batch_idx)
    #writer.add_scalar(f'BATCH/DIMENSIONS/Active Latent Dimensions', len(active_dims), batch_idx)

def helper_tensorboard_training_loss(writer, total_loss, total_NLL, total_kl_div, epoch, total_time, active_dims, activity_scores, model, timestamp, filename_hyperparameters):
    avg_loss, avg_loss_SE, avg_nll, avg_nll_SE, avg_kl, avg_kl_SE = avg_loss_SError(total_loss, total_NLL, total_kl_div)
    writer.add_scalar('EPOCH_LOSS/Training Loss (NLL + KL Div)', np.sum(total_loss), epoch)
    writer.add_scalar('EPOCH_LOSS/Training Loss (NLL)', np.sum(total_NLL), epoch)
    writer.add_scalar('EPOCH_LOSS/Training Loss (KL Div)', np.sum(total_kl_div), epoch)

    writer.add_scalar('EPOCH_LOSS/Avg. Training Loss (NLL + KL Div)', avg_loss, epoch)
    writer.add_scalar('EPOCH_LOSS/Avg. Training Loss (NLL)', avg_nll, epoch)
    writer.add_scalar('EPOCH_LOSS/Avg. Training Loss (KL Div)', avg_kl, epoch)

    writer.add_scalar('EPOCH_LOSS/Standard Error Training Loss (NLL + KL Div)', stats.sem(total_loss), epoch)
    writer.add_scalar('EPOCH_LOSS/Standard Error Training Loss (NLL)', stats.sem(total_NLL), epoch)
    writer.add_scalar('EPOCH_LOSS/Standard Error Training Loss (KL Div)', stats.sem(total_kl_div), epoch)
    
    writer.add_scalar('EPOCH_TIME/Training Time (Minutes)', total_time / 60, epoch)

    writer.add_scalar('EPOCH_DIM/Active Latent Dimensions - Training', len(active_dims), epoch)
    image = activity_scores.view(5, 10)
    image = image.unsqueeze(0)
    writer.add_image('EPOCH_DIM/Active Latent Dimensions Matrix - Training', image, global_step=epoch)
    log_activity_scores(writer, activity_scores, epoch, filename_hyperparameters, False)

def helper_tensorboard_test_loss(writer, total_loss, total_NLL, total_kl_div, epoch, total_time, active_dims, activity_scores, model, timestamp, filename_hyperparameters):
    avg_loss, avg_loss_SE, avg_nll, avg_nll_SE, avg_kl, avg_kl_SE = avg_loss_SError(total_loss, total_NLL, total_kl_div)
    writer.add_scalar('EPOCH_LOSS/Test Loss (NLL + KL Div)', np.sum(total_loss), epoch)
    writer.add_scalar('EPOCH_LOSS/Test Loss (NLL)', np.sum(total_NLL), epoch)
    writer.add_scalar('EPOCH_LOSS/Test Loss (KL Div)', np.sum(total_kl_div), epoch)

    writer.add_scalar('EPOCH_LOSS/Avg. Test Loss (NLL + KL Div)', avg_loss, epoch)
    writer.add_scalar('EPOCH_LOSS/Avg. Test Loss (NLL)', avg_nll, epoch)
    writer.add_scalar('EPOCH_LOSS/Avg. Test Loss (KL Div)', avg_kl, epoch)

    writer.add_scalar('EPOCH_LOSS/Standard Error Test Loss (NLL + KL Div)', stats.sem(total_loss), epoch)
    writer.add_scalar('EPOCH_LOSS/Standard Error Test Loss (NLL)', stats.sem(total_NLL), epoch)
    writer.add_scalar('EPOCH_LOSS/Standard Error Test Loss (KL Div)', stats.sem(total_kl_div), epoch)
    
    writer.add_scalar('EPOCH_TIME/Test Time (Minutes)', total_time / 60, epoch)

    writer.add_scalar('EPOCH_DIM/Active Latent Dimensions - Test', len(active_dims), epoch)
    image = activity_scores.view(5, 10)
    image = image.unsqueeze(0)
    writer.add_image('EPOCH_DIM/Active Latent Dimensions Matrix - Test', image, global_step=epoch)
    log_activity_scores(writer, activity_scores, epoch, filename_hyperparameters, True)

def helper_tensorboard_test_snr(writer, overall_snr_db, snr_per_dim_db, mean_snr_db, overall_snr_db_test, snr_per_dim_db_test, mean_snr_db_test, epoch):
    writer.add_scalar('SNR/Test Overall SNR', overall_snr_db_test, epoch)
    writer.add_scalar('SNR/Test per DIM SNR', snr_per_dim_db_test.item(), epoch)
    writer.add_scalar('SNR/Test Mean SNR', mean_snr_db_test, epoch)

    writer.add_scalar('SNR/Training Overall SNR', overall_snr_db, epoch)
    writer.add_scalar('SNR/Training per DIM SNR', snr_per_dim_db.item(), epoch)
    writer.add_scalar('SNR/Training Mean SNR', mean_snr_db, epoch)

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

def latent_snr(model, data_loader, device):
    model.eval()
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            [x_tilde, z, mu_z, log_var_z] = model.forward(x)
            
            # Average over k samples
            mu_avg = mu_z.mean(dim=1)  # [batch, latent_dim]
            logvar_avg = log_var_z.mean(dim=1)
            
            all_mu.append(mu_avg)
            all_logvar.append(logvar_avg)
    
    all_mu = torch.cat(all_mu, dim=0)  # [total_samples, latent_dim]
    all_logvar = torch.cat(all_logvar, dim=0)
    
    # Signal: variance of the mean across all samples
    signal_var = torch.var(all_mu, dim=0)  # [latent_dim]
    
    # Noise: average predicted variance
    noise_var = torch.exp(all_logvar).mean(dim=0)  # [latent_dim]
    
    # SNR per latent dimension
    snr_per_dim = signal_var / noise_var
    snr_db_per_dim = 10 * torch.log10(snr_per_dim)
    
    # Overall SNR
    overall_snr = signal_var.mean() / noise_var.mean()
    overall_snr_db = 10 * torch.log10(overall_snr)
    
    return overall_snr_db.item(), snr_db_per_dim.cpu().numpy(), snr_db_per_dim.mean().item()

def train_general(data_loader, data_loader_test, model, lr, num_epochs, K, writer, optimizer, fixed_tensorboard_batch, scheduler, SNR_bool, timestamp, filename_hyperparameters):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.train()
    writer.add_graph(model, fixed_tensorboard_batch[:1])

    epoch_times = []
    global_step = 0  # For continuous tracking across epochs

    for epoch in range(1, num_epochs + 1):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

        total_loss = []
        total_NLL = []
        total_kl_div = []
        total_time = 0
        epoch_start_time = time.time()

        for batch_idx, x in loop:
            x = x.to(device)
            batch_start_time = time.time()

            # IWAE update
            optimizer.zero_grad()
            loss, NLL, KL_div = model.compute_loss(x)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_NLL.append(NLL.item())
            total_kl_div.append(KL_div.item())
            total_time += time.time() - batch_start_time

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                helper_tensorboard_loss_batch(writer, loss, NLL, KL_div, total_loss, total_NLL, total_kl_div, batch_idx, batch_start_time, epoch)

            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())

            global_step += 1  # For continuous logging if desired

        scheduler.step()
        epoch_times.append(time.time() - epoch_start_time)

        # evaluations
        activity_scores, active_dims, all_posterior_means = compute_activity_paper_method(model, data_loader, device, ACTIVE_LATENT_DIM_THRESHOLD, None)

        helper_tensorboard_training_loss(writer, total_loss, total_NLL, total_kl_div, epoch, total_time, active_dims, activity_scores, model, timestamp, filename_hyperparameters)
      

        # Log images every SAVE_INTERVAL epochs
        if epoch % SAVE_INTERVAL == 0:
            with torch.no_grad():
                # Reconstructions
                reconstructions = model.reconstruct(fixed_tensorboard_batch[:16])
                comparison = create_comparison_image(fixed_tensorboard_batch[:8], reconstructions[:8])

                writer.add_images('Reconstructions/Original_vs_Reconstructed', comparison, epoch, dataformats='NCHW')

                # Generated samples
                generated_samples = model.sample(16, device)
                writer.add_images('Generated_Samples', generated_samples, epoch, dataformats='NCHW')

                # Latent traversal (optional)
                if epoch % (SAVE_INTERVAL * 2) == 0:
                    traversal_images = model.create_latent_traversal(fixed_tensorboard_batch[:1], n_pert=10, n_latents=5)
                    traversal_flat = traversal_images.view(-1, 1, 28, 28)
                    writer.add_images('Latent_Traversal', traversal_flat, epoch, dataformats='NCHW')

            if SNR_bool:
                overall_snr_db, snr_per_dim_db, mean_snr_db = latent_snr(model, data_loader, device)
                overall_snr_db_test, snr_per_dim_db_test, mean_snr_db_test = latent_snr(model, data_loader_test, device)
                helper_tensorboard_test_snr(writer, overall_snr_db, snr_per_dim_db, mean_snr_db, overall_snr_db_test, snr_per_dim_db_test, mean_snr_db_test, epoch)
        
        activity_scores_test, active_dims_test, all_posterior_means_test = compute_activity_paper_method(model, data_loader_test, device, ACTIVE_LATENT_DIM_THRESHOLD, None)
        total_loss_eval, total_nll_eval, total_kl_eval, total_time_eval, num_samples_eval = evaluate_test_loss(model, data_loader_test, device, k=model.k)
        helper_tensorboard_test_loss(writer, total_loss_eval, total_nll_eval, total_kl_eval, epoch, total_time_eval, active_dims_test, activity_scores_test, model, timestamp, filename_hyperparameters)
        torch.save(activity_scores_test,  f'./results/{filename_hyperparameters}_latent_matrix_test_epoch_{epoch}.pt')

    return model

def lr_lambda(epoch):
    # Determine which phase the current epoch belongs to
    phases = [3**i for i in range(8)]
    cumulative_epochs = 0
    for i, phase_len in enumerate(phases):
        if epoch < cumulative_epochs + phase_len:
            # Learning rate multiplier for phase i
            return 10 ** (-i / 7)
        cumulative_epochs += phase_len
    # After all phases, keep last LR
    return 10 ** (-7 / 7)

def train_paper(dataset, dataset_test, vae_model, iwae_model, lr, num_epochs, K):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/VAE_experiment_{timestamp}_{lr}_{num_epochs}_{K}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}"
    log_dir2 = f"runs/IWAE_experiment_{timestamp}_{lr}_{num_epochs}_{K}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}"
    writer_vae = SummaryWriter(log_dir)
    writer_iwae = SummaryWriter(log_dir2)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), 
                                     lr=lr,
                                     betas=(0.9, 0.999),        
                                     eps=1e-4)
    optimizer_iwae = torch.optim.Adam(iwae_model.parameters(), 
                                      lr=lr,
                                      betas=(0.9, 0.999),        # beta1=0.9, beta2=0.999
                                      eps=1e-4)
    scheduler_vae = torch.optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda=lr_lambda)
    scheduler_iwae = torch.optim.lr_scheduler.LambdaLR(optimizer_iwae, lr_lambda=lr_lambda)

    # Fixed test batch for consistent visualization
    fixed_test_batch = next(iter(data_loader_test)).to(device)
    
    vae_model = train_general(data_loader, data_loader_test, vae_model, lr, num_epochs, K, writer_vae, optimizer_vae, fixed_test_batch, scheduler_vae, False, timestamp)
    iwae_model = train_general(data_loader, data_loader_test, iwae_model, lr, num_epochs, K, writer_iwae, optimizer_iwae, fixed_test_batch, scheduler_iwae, False, timestamp)
    
    torch.save(vae_model, f'./results/trained_vae_{timestamp}_{LEARNING_RATE}_{NUM_EPOCHS}_{k}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}.pth')
    torch.save(iwae_model, f'./results/trained_iwae_{timestamp}_{LEARNING_RATE}_{NUM_EPOCHS}_{k}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}.pth')

    writer_vae.flush() 
    time.sleep(0.1)
    writer_vae.close()
    writer_iwae.flush() 
    time.sleep(0.1)
    writer_iwae.close()
    
def train_experiment_active_latent_dimensions(dataset, dataset_test, vae_model, iwae_model, lr, num_epochs, K, init_method, seed):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_hyperparameters_vae = f'{timestamp}_{vae_model.model_type}_{lr}_{num_epochs}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}_{LATENT_DIM}_{K}_{init_method}_{seed}'
    filename_hyperparameters_iwae = f'{timestamp}_{iwae_model.model_type}_{lr}_{num_epochs}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}_{LATENT_DIM}_{K}_{init_method}_{seed}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_dir = f"runs/VAE_experiment_{filename_hyperparameters_vae}"
    log_dir2 = f"runs/IWAE_experiment_{filename_hyperparameters_iwae}"
    writer_vae = SummaryWriter(log_dir)
    writer_iwae = SummaryWriter(log_dir2)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), 
                                     lr=lr,
                                     betas=(0.9, 0.999),        
                                     eps=1e-4)
    optimizer_iwae = torch.optim.Adam(iwae_model.parameters(), 
                                      lr=lr,
                                      betas=(0.9, 0.999),        
                                      eps=1e-4)
    
    scheduler_vae = torch.optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda=lr_lambda)
    scheduler_iwae = torch.optim.lr_scheduler.LambdaLR(optimizer_iwae, lr_lambda=lr_lambda)

    # Fixed test batch for consistent visualization
    fixed_test_batch = next(iter(data_loader_test)).to(device)
    
    vae_model = train_general(data_loader, data_loader_test, vae_model, lr, num_epochs, K, writer_vae, optimizer_vae, fixed_test_batch, scheduler_vae, True, timestamp, filename_hyperparameters_vae)
    iwae_model = train_general(data_loader, data_loader_test, iwae_model, lr, num_epochs, K, writer_iwae, optimizer_iwae, fixed_test_batch, scheduler_iwae, True, timestamp, filename_hyperparameters_iwae)
    
    torch.save(vae_model, f'./results/trained_vae_{filename_hyperparameters_vae}.pth')
    torch.save(iwae_model, f'./results/trained_iwae_{filename_hyperparameters_iwae}.pth')

    writer_vae.flush() 
    writer_vae.close()

    writer_iwae.flush() 
    writer_iwae.close()

binarized_MNIST = Binarized_MNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor())
binarized_MNIST_Test = Binarized_MNIST('./data', train=False, download=True,
                                  transform=transforms.ToTensor())

#global_step = 0
for seed in SEEDS:
    set_seed(seed)
    for k in LIST_OF_KS:
        for m in INITIALISATION:
            vae_model = VAE(k, m)
            iwae_model = IWAE(k, m)
            train_experiment_active_latent_dimensions(binarized_MNIST, binarized_MNIST_Test, vae_model, iwae_model, LEARNING_RATE, NUM_EPOCHS, k, m, seed)


'''
for k in LIST_OF_KS:
    vae_model = VAE(k)
    iwae_model = IWAE(k)

    # Log hyperparameters and final loss
    
    hparams = {
        'learning rate': float(LEARNING_RATE), 
        'epochs': int(NUM_EPOCHS), 
        'k': int(k)
        }
    
    for m in INITIALISATION:
        train_paper(binarized_MNIST, binarized_MNIST_Test, vae_model, iwae_model, LEARNING_RATE, NUM_EPOCHS, k)

    avg_loss, avg_loss_SE, avg_nll, avg_nll_SE, avg_kl, avg_kl_SE = avg_loss_SError(total_loss, total_NLL, total_kl_div)
    metrics = {
        'hparam/final_loss': float(avg_loss),
        'hparam/final_NLL': float(avg_nll),  
        'hparam/final_KL': float(avg_kl)
        }  
    
    hparam_log_dir = f'runs/hparams/{timestamp}_{vae_model.model_type}__lr_{LEARNING_RATE}_K_{k}_{BATCH_SIZE}_{ACTIVE_LATENT_DIM_THRESHOLD}'
    hparam_writer = SummaryWriter(log_dir=hparam_log_dir)
    hparam_writer.add_hparams(hparams, metrics, global_step = global_step)
    hparam_writer.flush() 
    time.sleep(0.1)
    hparam_writer.close()

    global_step += 1 
    '''