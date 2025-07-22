import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import numpy as np
import torch
import os

# === Font Configuration ===
font_path = "/usr/share/fonts/truetype/latin-modern/LMRoman10-Regular.ttf"
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.size'] = 12
    print(f"Using font: {prop.get_name()}")
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    print("Latin Modern Roman not found; using serif")

# === Load Data ===
data = {
    'Seed 630': torch.load('results/Secound Run/torch/20250708_042910_IWAE_0.001_80_20_0.01_50_5_XavierNormal_630_latent_matrix_test_epoch80.pt'),
    'Seed 135': torch.load('results/Secound Run/torch/20250706_200401_IWAE_0.001_80_20_0.01_50_5_XavierNormal_135_latent_matrix_test_epoch80.pt'),
    'Seed 32': torch.load('results/Secound Run/torch/20250718_111329_IWAE_0.001_80_20_0.01_50_5_XavierNormal_32_latent_matrix_test_epoch_80.pt'),
    'Seed 10': torch.load('results/Secound Run/torch/20250717_100029_IWAE_0.001_80_20_0.01_50_5_XavierNormal_10_latent_matrix_test_epoch_80.pt'),
    'Seed 924': torch.load('results/Secound Run/torch/20250716_090025_IWAE_0.001_80_20_0.01_50_5_XavierNormal_924_latent_matrix_test_epoch_80.pt')
}

# Load mean data and integrate
mean_data = np.load('results/Secound Run/torch/mean_matrices_IWAE_k5_XavierNormal.npy')
assert mean_data.shape[0] == 50, "Expected 50 latent dims"
data['Aggregated per Dimension over Seeds\n(Mean)'] = torch.tensor(mean_data)

# === Threshold and Colormap ===
threshold = 1e-2
vmax = 1.1
colors = [(1, 1, 1), (0, 0.6, 0)]  # White to dark green
green_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_green", colors)
norm = mcolors.Normalize(vmin=threshold, vmax=vmax)

# === Plot Setup ===
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.1)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# === Unified Plotting Loop ===
for idx, (label, tensor_data) in enumerate(data.items()):
    tensor_np = tensor_data.detach().cpu().numpy() if isinstance(tensor_data, torch.Tensor) else np.array(tensor_data)
    assert tensor_np.shape[0] == 50, f"{label}: Expected 50 dimensions, got {tensor_np.shape[0]}"

    grid_data = tensor_np.reshape(5, 10)
    masked_data = np.where(grid_data > threshold, grid_data, 0)

    ax = axes[idx]
    im = ax.imshow(masked_data, cmap=green_cmap, norm=norm)

    ax.set_title(label, fontsize=12)
    ax.set_xticks(range(10))
    ax.set_yticks(range(5))
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Latent dim column')
    if idx % 3 == 0:
        ax.set_ylabel('Latent dim row')

# === Colorbar ===
cbar_ax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Activation Value', rotation=270, labelpad=15)

# Threshold marker
cbar.ax.axhline(y=norm(threshold), color='red', linestyle='--', linewidth=2)
cbar.ax.text(1.2, norm(threshold), f'Threshold = {threshold:0.01}', color='black', fontsize=10,
             va='center', ha='left', transform=cbar.ax.transAxes)

# === Save and Show ===
plt.savefig('results/active_dim.pdf', format='pdf', bbox_inches='tight')
plt.show()

# === Summary Stats ===
print("Summary Statistics:")
print("-" * 50)
for label, tensor_data in data.items():
    tensor_np = tensor_data.detach().cpu().numpy() if isinstance(tensor_data, torch.Tensor) else np.array(tensor_data)
    active_neurons = np.sum(tensor_np > threshold)
    max_val = np.max(tensor_np)
    mean_active = np.mean(tensor_np[tensor_np > threshold]) if active_neurons > 0 else 0

    print(f"{label}:")
    print(f"  Active neurons (> {threshold}): {active_neurons}/50")
    print(f"  Maximum value: {max_val:.4f}")
    print(f"  Mean of active neurons: {mean_active:.4f}\n")
