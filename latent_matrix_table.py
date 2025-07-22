import torch
import numpy as np
import pandas as pd
import os

# Load the data as you specified
data = {
    'Seed 630': torch.load('results/Secound Run/torch/20250708_042910_IWAE_0.001_80_20_0.01_50_5_XavierNormal_630_latent_matrix_test_epoch80.pt'),
    'Seed 135': torch.load('results/Secound Run/torch/20250706_200401_IWAE_0.001_80_20_0.01_50_5_XavierNormal_135_latent_matrix_test_epoch80.pt'),
    'Seed 32': torch.load('results/Secound Run/torch/20250718_111329_IWAE_0.001_80_20_0.01_50_5_XavierNormal_32_latent_matrix_test_epoch_80.pt'),
    'Seed 10': torch.load('results/Secound Run/torch/20250717_100029_IWAE_0.001_80_20_0.01_50_5_XavierNormal_10_latent_matrix_test_epoch_80.pt'),
    'Seed 924': torch.load('results/Secound Run/torch/20250716_090025_IWAE_0.001_80_20_0.01_50_5_XavierNormal_924_latent_matrix_test_epoch_80.pt')
}

# Load mean data
mean_data = np.load('results/Secound Run/torch/mean_matrices_IWAE_k5_XavierNormal.npy')

# Create output directory if it doesn't exist
output_dir = 'csv_output'
os.makedirs(output_dir, exist_ok=True)

# Prepare data for combined CSV
all_data = []

# Add individual seed data
for seed_name, tensor_data in data.items():
    # Convert tensor to numpy if it's a tensor
    if isinstance(tensor_data, torch.Tensor):
        numpy_data = tensor_data.detach().cpu().numpy()
    else:
        numpy_data = tensor_data
    
    # Flatten the data if it's multidimensional and take the first row or appropriate slice
    if numpy_data.ndim > 1:
        # Assuming we want the first row or need to flatten appropriately
        if numpy_data.shape[1] == 50:  # If it's already 50 latent dimensions
            flattened_data = numpy_data[0]  # Take first row
        else:
            flattened_data = numpy_data.flatten()[:50]  # Take first 50 elements
    else:
        flattened_data = numpy_data[:50]  # Take first 50 elements
    
    # Ensure we have exactly 50 dimensions, pad with zeros if needed
    if len(flattened_data) < 50:
        flattened_data = np.pad(flattened_data, (0, 50 - len(flattened_data)))
    else:
        flattened_data = flattened_data[:50]
    
    # Create row with seed name and latent dimensions
    row = [seed_name] + flattened_data.tolist()
    all_data.append(row)

# Add mean data
if mean_data.ndim > 1:
    if mean_data.shape[1] == 50:
        mean_flattened = mean_data[0]
    else:
        mean_flattened = mean_data.flatten()[:50]
else:
    mean_flattened = mean_data[:50]

# Ensure mean data has exactly 50 dimensions
if len(mean_flattened) < 50:
    mean_flattened = np.pad(mean_flattened, (0, 50 - len(mean_flattened)))
else:
    mean_flattened = mean_flattened[:50]

mean_row = ['Mean'] + mean_flattened.tolist()
all_data.append(mean_row)

# Create column names
columns = ['Seed'] + [f'Latent_Dim_{i+1}' for i in range(50)]

# Create DataFrame
df = pd.DataFrame(all_data, columns=columns)

# Save to CSV
filename = f"{output_dir}/combined_latent_data.csv"
df.to_csv(filename, index=False)

print(f"Combined data saved to {filename}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())