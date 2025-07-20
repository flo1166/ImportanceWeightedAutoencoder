"""
Latent Matrix Data Aggregator and Statistical Analysis
Aggregates latent matrix data across seeds, calculates means and standard errors based on model and initialization.
Processes PyTorch .pt files containing latent matrices.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LatentMatrixAggregator:
    def __init__(self, pt_base_dir):
        self.pt_base_dir = Path(pt_base_dir)
        
    def extract_seed_from_filename(self, filename):
        """Extract numerical seed from filename."""
        # Remove .pt extension
        name = filename.replace('.pt', '')
        
        # Split by underscores and look for the specific seeds
        parts = name.split('_')
        
        # Look for the specific seeds: 135, 630, 924, 10, 32
        valid_seeds = ['135', '630', '924', '10', '32']
        
        for part in parts:
            if part in valid_seeds:
                return part
        
        return 'unknown'
    
    def parse_filename(self, filename):
        """
        Parse filename to extract model, k, and initialization method.
        Example: '20250706_010948_IWAE_0.001_80_20_0.01_50_1_XavierUni_135_latent_matrix_test_epoch1.pt'
        Structure: date_time_MODEL_lr_hidden1_hidden2_lr2_epochs_k_INIT_seed_latent_matrix_test_epoch1.pt
        """
        # Remove .pt extension
        name = filename.replace('.pt', '')
        
        # Split by underscores
        parts = name.split('_')
        
        # Initialize variables
        model = None
        k = None
        initialization = None
        
        # Find model (VAE or IWAE) - should be at index 2
        if len(parts) > 2 and parts[2] in ['VAE', 'IWAE']:
            model = parts[2]
        
        # Find initialization method (common patterns)
        init_patterns = ['XavierUni', 'XavierNormal', 'KaimingUni', 'KaimingNormal']
        init_index = None
        for i, part in enumerate(parts):
            if any(init_pattern in part for init_pattern in init_patterns):
                initialization = part
                init_index = i
                break
        
        # Find k value - should be right before the initialization method
        if init_index is not None and init_index > 0:
            # k should be at init_index - 1
            k_candidate = parts[init_index - 1]
            if k_candidate.isdigit():
                k = int(k_candidate)
        
        # If we couldn't find k in the expected position, try the old logic
        if k is None:
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 1:
                    # Make sure it's not the seed (which comes after initialization)
                    if init_index is None or i < init_index:
                        k = int(part)
                        break
        
        return {
            'model': model,
            'k': k,
            'initialization': initialization,
            'filename': filename
        }
    
    def calculate_latent_statistics(self, latent_matrix):
        """Return the latent matrix itself for element-wise averaging."""
        if latent_matrix is None:
            return None, None
        
        # Convert to numpy if it's a torch tensor
        if torch.is_tensor(latent_matrix):
            matrix = latent_matrix.detach().cpu().numpy()
        else:
            matrix = latent_matrix
        
        # Return the matrix itself, not scalar statistics
        shape_str = 'x'.join(map(str, matrix.shape))
        
        return matrix, shape_str
    
    def load_latent_matrix(self, pt_path):
        """Load latent matrix from PyTorch file."""
        try:
            latent_matrix = torch.load(pt_path, map_location='cpu')
            return latent_matrix
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            return None
    
    def collect_all_data(self):
        """Collect data from all .pt files."""
        all_data = []
        shape_data = []  # Separate collection for shape information
        
        # Get all .pt files
        all_pt_files = list(self.pt_base_dir.glob('*.pt'))
        
        # Filter to only epoch 80 files that contain "test" in filename
        pt_files = [f for f in all_pt_files if ('epoch80.pt' in f.name or 'epoch_80' in f.name) and 'test' in f.name]
        
        print(f"Found {len(all_pt_files)} total .pt files in {self.pt_base_dir}")
        print(f"Filtering to {len(pt_files)} epoch 80 test files")
        
        if not pt_files:
            print("No epoch 80 test .pt files found! Check the directory path.")
            return all_data, shape_data
        
        for pt_file in pt_files:
            print(f"Processing: {pt_file.name}")
            
            # Parse filename
            parsed = self.parse_filename(pt_file.name)
            print(f"  Parsed - Model: {parsed['model']}, k: {parsed['k']}, Init: {parsed['initialization']}")
            
            if not all([parsed['model'], parsed['initialization']]):
                print(f"  Skipping due to incomplete parsing: {pt_file.name}")
                continue
            
            # Load latent matrix
            latent_matrix = self.load_latent_matrix(pt_file)
            if latent_matrix is None:
                print(f"  Failed to load matrix from: {pt_file.name}")
                continue
            
            print(f"  Loaded matrix with shape: {latent_matrix.shape if hasattr(latent_matrix, 'shape') else 'unknown'}")
            
            # Calculate statistics
            matrix, shape_str = self.calculate_latent_statistics(latent_matrix)
            print(f"  Matrix shape: {shape_str}")
            print(f"  Extracted seed: {self.extract_seed_from_filename(pt_file.name)}")  # Debug seed extraction
            
            if matrix is None:
                print(f"  Failed to process matrix from: {pt_file.name}")
                continue
            
            # Store shape information separately
            shape_info = {
                'Model': parsed['model'],
                'k': parsed['k'],
                'Initialization_Method': parsed['initialization'],
                'Shape': shape_str,
                'Filename': pt_file.name
            }
            shape_data.append(shape_info)
            
            # Store the matrix itself for element-wise averaging
            data_point = {
                'Model': parsed['model'],
                'k': parsed['k'],
                'Initialization_Method': parsed['initialization'],
                'Measure': 'Latent_Matrix',
                'Matrix': matrix,  # Store the actual matrix
                'Filename': pt_file.name
            }
            all_data.append(data_point)
        
        print(f"Total matrix data points collected: {len(all_data)}")
        print(f"Total shape records collected: {len(shape_data)}")
        return all_data, shape_data
    
    def calculate_statistics(self, data_points):
        """Calculate element-wise mean matrices for grouped data."""
        df = pd.DataFrame(data_points)
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by Model, k, Initialization_Method, and Measure
        grouped = df.groupby(['Model', 'k', 'Initialization_Method', 'Measure'])
        
        results = []
        
        for name, group in grouped:
            model, k, init_method, measure = name
            matrices = group['Matrix'].values  # Get the actual matrices
            
            if len(matrices) == 0:
                continue
            
            # Stack matrices and compute element-wise mean
            try:
                stacked_matrices = np.stack(matrices)
                mean_matrix = np.mean(stacked_matrices, axis=0)
                
                # Calculate standard error element-wise
                if len(matrices) > 1:
                    se_matrix = stats.sem(stacked_matrices, axis=0)
                else:
                    se_matrix = np.zeros_like(mean_matrix)
                
                result = {
                    'Model': model,
                    'k': k,
                    'Initialization_Method': init_method,
                    'Measure': measure,
                    'Mean_Matrix': mean_matrix,
                    'SE_Matrix': se_matrix,
                    'n_seeds': len(matrices),
                    'Seeds_Used': ', '.join([self.extract_seed_from_filename(f) for f in group['Filename']]),
                    'Matrix_Shape': 'x'.join(map(str, mean_matrix.shape))
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing matrices for {name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_shape_summary(self, shape_data, output_file="shape_summary.csv"):
        """Create a summary of matrix shapes across experiments."""
        if not shape_data:
            return None
        
        shape_df = pd.DataFrame(shape_data)
        
        # Check for consistent shapes within groups
        shape_summary = shape_df.groupby(['Model', 'k', 'Initialization_Method']).agg({
            'Shape': lambda x: list(x.unique()),
            'Filename': 'count'
        }).reset_index()
        
        shape_summary.columns = ['Model', 'k', 'Initialization_Method', 'Unique_Shapes', 'File_Count']
        
        # Check if shapes are consistent
        shape_summary['Shape_Consistent'] = shape_summary['Unique_Shapes'].apply(lambda x: len(x) == 1)
        shape_summary['Primary_Shape'] = shape_summary['Unique_Shapes'].apply(lambda x: x[0] if len(x) == 1 else f"Multiple: {', '.join(x)}")
        
        # Save shape summary
        if output_file:
            output_path = self.pt_base_dir / output_file
            shape_summary.to_csv(output_path, index=False)
            print(f"Shape summary saved to: {output_path}")
        
        return shape_summary
    
    def save_mean_matrices(self, results_df, output_prefix="mean_matrices"):
        """Save the mean matrices to separate files."""
        if results_df.empty:
            return
        
        for _, row in results_df.iterrows():
            # Create filename
            filename = f"{output_prefix}_{row['Model']}_k{row['k']}_{row['Initialization_Method']}.npy"
            filepath = self.pt_base_dir / filename
            
            # Save mean matrix
            np.save(filepath, row['Mean_Matrix'])
            
            # Also save SE matrix
            se_filename = f"{output_prefix}_{row['Model']}_k{row['k']}_{row['Initialization_Method']}_SE.npy"
            se_filepath = self.pt_base_dir / se_filename
            np.save(se_filepath, row['SE_Matrix'])
            
            print(f"Saved mean matrix: {filename}")
            print(f"Saved SE matrix: {se_filename}")
    
    def create_summary_table(self, output_file=None):
        """Create the main summary table with element-wise mean matrices."""
        print("Collecting data from all .pt files...")
        
        # Collect all data points
        data_points, shape_data = self.collect_all_data()
        
        if not data_points:
            print("No matrix data points found!")
            return None, None
        
        print(f"Found {len(data_points)} matrix data points")
        
        # Calculate element-wise statistics
        print("Calculating element-wise mean matrices...")
        results_df = self.calculate_statistics(data_points)
        
        if results_df.empty:
            print("No statistics could be calculated!")
            return None, None
        
        # Sort the results for better readability
        results_df = results_df.sort_values(['Model', 'k', 'Initialization_Method', 'Measure'])
        
        # Create shape summary
        shape_summary = self.create_shape_summary(shape_data)
        
        # Save mean matrices as .npy files
        print("Saving mean matrices...")
        self.save_mean_matrices(results_df)
        
        # Create a simplified summary table without the actual matrices (for CSV export)
        summary_for_csv = results_df.drop(columns=['Mean_Matrix', 'SE_Matrix']).copy()
        
        # Save summary to file if specified
        if output_file:
            output_path = self.pt_base_dir / output_file
            summary_for_csv.to_csv(output_path, index=False)
            print(f"Summary results saved to: {output_path}")
        
        return results_df, shape_summary
    
    def create_pivot_tables(self, results_df, output_prefix="latent_pivot"):
        """Create pivot tables for different views of the data."""
        if results_df.empty:
            return
        
        # Pivot by Model and Initialization for each measure
        measures = results_df['Measure'].unique()
        
        for measure in measures:
            measure_data = results_df[results_df['Measure'] == measure]
            
            if measure_data.empty:
                continue
            
            # Create pivot table: Model x Initialization_Method
            try:
                pivot_mean = measure_data.pivot_table(
                    index=['Model', 'k'], 
                    columns='Initialization_Method', 
                    values='Mean',
                    aggfunc='first'
                )
                
                pivot_se = measure_data.pivot_table(
                    index=['Model', 'k'], 
                    columns='Initialization_Method', 
                    values='SE',
                    aggfunc='first'
                )
                
                # Save pivot tables
                safe_measure = measure.replace('/', '_').replace('+', '_')
                
                mean_file = self.pt_base_dir / f"{output_prefix}_{safe_measure}_mean.csv"
                se_file = self.pt_base_dir / f"{output_prefix}_{safe_measure}_se.csv"
                
                pivot_mean.to_csv(mean_file)
                pivot_se.to_csv(se_file)
                
                print(f"Pivot tables saved: {mean_file.name}, {se_file.name}")
                
            except Exception as e:
                print(f"Could not create pivot table for {measure}: {e}")
    
    def print_summary_statistics(self, results_df, shape_summary=None):
        """Print summary statistics about the data."""
        if results_df.empty:
            return
        
        print("\n" + "="*60)
        print("LATENT MATRIX SUMMARY STATISTICS")
        print("="*60)
        
        print(f"Total combinations: {len(results_df)}")
        print(f"Models: {sorted(results_df['Model'].unique())}")
        print(f"K values: {sorted(results_df['k'].unique())}")
        print(f"Initialization methods: {sorted(results_df['Initialization_Method'].unique())}")
        print(f"Measures: {sorted(results_df['Measure'].unique())}")
        
        print(f"\nSeeds per combination (mean): {results_df['n_seeds'].mean():.1f}")
        print(f"Combinations with all seeds: {len(results_df[results_df['n_seeds'] == results_df['n_seeds'].max()])}")
        
        # Show combinations with missing seeds
        max_seeds = results_df['n_seeds'].max()
        missing_seeds = results_df[results_df['n_seeds'] < max_seeds]
        if not missing_seeds.empty:
            print(f"\nCombinations with missing seeds ({len(missing_seeds)}):")
            for _, row in missing_seeds.iterrows():
                print(f"  {row['Model']} k={row['k']} {row['Initialization_Method']} {row['Measure']}: {row['n_seeds']} seeds")
        
        # Print shape summary if available
        if shape_summary is not None and not shape_summary.empty:
            print(f"\nSHAPE CONSISTENCY CHECK:")
            inconsistent_shapes = shape_summary[~shape_summary['Shape_Consistent']]
            if inconsistent_shapes.empty:
                print("✓ All matrix shapes are consistent within each group")
            else:
                print(f"⚠ Found {len(inconsistent_shapes)} groups with inconsistent shapes:")
                for _, row in inconsistent_shapes.iterrows():
                    print(f"  {row['Model']} k={row['k']} {row['Initialization_Method']}: {row['Primary_Shape']}")
    
    def filter_by_measures(self, results_df, measures_to_keep=None):
        """Filter results to keep only specific measures."""
        if measures_to_keep is None:
            # Default to only mean since that's what we're calculating
            measures_to_keep = ['Latent_mean']
        
        return results_df[results_df['Measure'].isin(measures_to_keep)]

def main():
    parser = argparse.ArgumentParser(description='Aggregate latent matrix data and calculate statistics')
    parser.add_argument('pt_dir', nargs='?', default='results/Secound Run/torch',
                       help='Directory containing .pt files (default: results/Secound Run/torch)')
    parser.add_argument('--output', '-o', default='latent_aggregated_results.csv',
                       help='Output file name (default: latent_aggregated_results.csv)')
    parser.add_argument('--pivot', action='store_true',
                       help='Create pivot tables for each measure')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information for filename parsing')
    parser.add_argument('--filter-measures', nargs='*',
                       help='Specific measures to include (default: most useful subset)')
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = LatentMatrixAggregator(args.pt_dir)
    
    if args.debug:
        print("Debug mode: Testing filename parsing...")
        pt_path = Path(args.pt_dir)
        if pt_path.exists():
            for pt_file in pt_path.glob('*.pt'):
                parsed = aggregator.parse_filename(pt_file.name)
                print(f"  {pt_file.name}")
                print(f"    Model: {parsed['model']}, k: {parsed['k']}, Init: {parsed['initialization']}")
        return
    
    print(f"Processing .pt files from: {args.pt_dir}")
    
    # Create summary table
    results_df, shape_summary = aggregator.create_summary_table(args.output)
    
    if results_df is not None:
        # Filter measures if specified
        if args.filter_measures is not None:
            results_df = aggregator.filter_by_measures(results_df, args.filter_measures)
            print(f"Filtered to {len(results_df)} rows with specified measures")
        
        # Print statistics
        aggregator.print_summary_statistics(results_df, shape_summary)
        
        # Create pivot tables if requested
        if args.pivot:
            print("\nCreating pivot tables...")
            aggregator.create_pivot_tables(results_df)
        
        # Show sample of results (without the actual matrices)
        print(f"\nSample results (first 10 rows):")
        display_df = results_df.drop(columns=['Mean_Matrix', 'SE_Matrix']) if 'Mean_Matrix' in results_df.columns else results_df
        print(display_df.head(10).to_string(index=False))
        
        print(f"\n✓ Processing completed!")
        print(f"Main results saved to: {Path(args.pt_dir) / args.output}")
    
    else:
        print("No results generated. Check your .pt directory and file structure.")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Basic usage
python latent_matrix_aggregator.py

# Custom .pt directory
python latent_matrix_aggregator.py /path/to/pt/files

# Custom output file and create pivot tables
python latent_matrix_aggregator.py --output my_latent_results.csv --pivot

# Filter to specific measures only
python latent_matrix_aggregator.py --filter-measures Latent_mean Latent_std Latent_min Latent_max

# Debug filename parsing
python latent_matrix_aggregator.py --debug

# Full example with all options
python latent_matrix_aggregator.py results/Second_Run/torch --output final_latent_results.csv --pivot
"""