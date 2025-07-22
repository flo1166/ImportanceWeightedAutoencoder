"""
Enhanced IWAE Data Aggregator with Seeds/Mean/SE Table Output
Aggregates data across seeds, calculates means and standard errors based on model and initialization.
Creates a table where each measure shows seeds, mean, and SE as columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class IWAEDataAggregator:
    def __init__(self, csv_base_dir):
        self.csv_base_dir = Path(csv_base_dir)
        
        # Define the mapping from CSV directories to measure names
        self.measure_mappings = {
            'test_activity': 'Test_Activity',
            'training_activity': 'Training_Activity',
            'test_loss': {
                'KL': 'Test_KL',
                'NLL': 'Test_NLL', 
                'NLL+KL': 'Test_NLL+KL'
            },
            'training_loss': {
                'KL': 'Training_KL',
                'NLL': 'Training_NLL',
                'NLL+KL': 'Training_NLL+KL'
            }
        }
    
    def extract_seed_from_filename(self, filename):
        """Extract numerical seed from filename."""
        # Remove .csv extension
        name = filename.replace('.csv', '')
        
        # Split by underscores and look for numerical seeds
        parts = name.split('_')
        
        # Look for numerical values that could be seeds (typically 2-4 digits)
        # Common seeds are like 10, 32, 135, 630, 924
        for part in reversed(parts):  # Start from the end
            if part.isdigit() and 2 <= len(part) <= 4:
                return part
        
        return 'unknown'
    
    def parse_filename(self, filename):
        """
        Parse filename to extract model, k, and initialization method.
        Example: 'IWAE_experiment_20250715_175020_IWAE_0.001_80_20_0.01_50_1_XavierUni_924_KL.csv'
        """
        # Remove .csv extension
        name = filename.replace('.csv', '')
        
        # Split by underscores
        parts = name.split('_')
        
        # Initialize variables
        model = None
        k = None
        initialization = None
        
        # Find model (VAE or IWAE) - usually appears twice
        model_indices = [i for i, part in enumerate(parts) if part in ['VAE', 'IWAE']]
        if model_indices:
            model = parts[model_indices[0]]  # Take the first occurrence
        
        # Find initialization method (common patterns)
        init_patterns = ['XavierUni', 'XavierNormal', 'KaimingUni', 'KaimingNormal']
        for i, part in enumerate(parts):
            if any(init_pattern in part for init_pattern in init_patterns):
                initialization = part
                break
        
        # Find k value (usually a single digit near the end, before initialization)
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 1:
                # Check if this might be k (usually before initialization)
                if i < len(parts) - 2:  # Not the seed (which is usually last or second to last)
                    k = int(part)
        
        return {
            'model': model,
            'k': k,
            'initialization': initialization,
            'filename': filename
        }
    
    def get_last_value_from_csv(self, csv_path):
        """Get the last value from a CSV file."""
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return None
            
            # Sort by step to ensure we get the actual last step
            df_sorted = df.sort_values('step')
            last_value = df_sorted.iloc[-1]['value']
            return last_value
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None
    
    def collect_data_from_directory(self, directory_name, measure_suffix=None):
        """Collect data from a specific directory."""
        directory_path = self.csv_base_dir / directory_name
        
        if not directory_path.exists():
            print(f"Directory {directory_path} does not exist")
            return []
        
        data_points = []
        
        for csv_file in directory_path.glob('*.csv'):
            # Skip combined files
            if csv_file.name.startswith('combined_'):
                continue
            
            # For loss directories, only process files that end with the expected suffix
            if measure_suffix is not None:
                # Extract the actual suffix from filename
                name_without_ext = csv_file.name.replace('.csv', '')
                parts = name_without_ext.split('_')
                if len(parts) == 0:
                    continue
                
                actual_suffix = parts[-1]  # Last part should be the measure
                
                # Only process if the actual suffix matches the expected suffix
                if (actual_suffix != measure_suffix) or ('SE' in parts):
                    continue
            
            # Parse filename
            parsed = self.parse_filename(csv_file.name)
            
            if not all([parsed['model'], parsed['initialization']]):
                print(f"Could not parse filename: {csv_file.name}")
                continue
            
            # Get the last value
            last_value = self.get_last_value_from_csv(csv_file)
            if last_value is None:
                continue
            
            # Extract seed
            seed = self.extract_seed_from_filename(csv_file.name)
            
            # Determine measure name
            if measure_suffix:
                measure = self.measure_mappings[directory_name][measure_suffix]
            else:
                measure = self.measure_mappings[directory_name]
            
            data_point = {
                'Model': parsed['model'],
                'k': parsed['k'],
                'Initialization_Method': parsed['initialization'],
                'Measure': measure,
                'Seed': seed,
                'Value': last_value,
                'Filename': csv_file.name
            }
            
            data_points.append(data_point)
        
        return data_points
    
    def collect_all_data(self):
        """Collect data from all directories."""
        all_data = []
        
        # Process activity directories (no suffix)
        for activity_dir in ['test_activity', 'training_activity']:
            data = self.collect_data_from_directory(activity_dir)
            all_data.extend(data)
        
        # Process loss directories (with suffixes)
        for loss_dir in ['test_loss', 'training_loss']:
            loss_path = self.csv_base_dir / loss_dir
            if not loss_path.exists():
                continue
            
            # Define the expected suffixes for loss measures
            expected_suffixes = ['KL', 'NLL', 'NLL+KL']
            
            # Collect data for each expected suffix
            for suffix in expected_suffixes:
                if suffix in self.measure_mappings[loss_dir]:
                    data = self.collect_data_from_directory(loss_dir, suffix)
                    all_data.extend(data)
        
        return all_data
    
    def create_seeds_mean_se_table(self, data_points, output_file=None):
        """Create a table where each measure shows seeds, mean, and SE as columns."""
        df = pd.DataFrame(data_points)
        
        if df.empty:
            print("No data points found!")
            return pd.DataFrame()
        
        print(f"Found {len(data_points)} data points")
        
        # Group by Model, k, Initialization_Method, and Measure
        grouped = df.groupby(['Model', 'k', 'Initialization_Method', 'Measure'])
        
        results = []
        
        for name, group in grouped:
            model, k, init_method, measure = name
            values = group['Value'].values
            seeds = group['Seed'].values
            
            if len(values) == 0:
                continue
            
            mean_value = np.mean(values)
            std_error = stats.sem(values) if len(values) > 1 else 0.0
            
            # Create a base row with identifiers
            base_row = {
                'Model': model,
                'k': k,
                'Initialization_Method': init_method
            }
            
            # Add measure-specific columns
            seeds_col = f"{measure}_Seeds"
            mean_col = f"{measure}_Mean"
            se_col = f"{measure}_SE"
            
            base_row[seeds_col] = ', '.join(map(str, sorted(seeds)))
            base_row[mean_col] = mean_value
            base_row[se_col] = std_error
            
            results.append((base_row, measure))
        
        if not results:
            print("No results to process!")
            return pd.DataFrame()
        
        # Combine all results into a single table
        # First, collect all unique combinations of Model, k, Initialization_Method
        combinations = {}
        all_measures = set()
        
        for row_data, measure in results:
            key = (row_data['Model'], row_data['k'], row_data['Initialization_Method'])
            if key not in combinations:
                combinations[key] = {
                    'Model': row_data['Model'],
                    'k': row_data['k'],
                    'Initialization_Method': row_data['Initialization_Method']
                }
            
            # Add measure-specific columns
            seeds_col = f"{measure}_Seeds"
            mean_col = f"{measure}_Mean"
            se_col = f"{measure}_SE"
            
            combinations[key][seeds_col] = row_data[seeds_col]
            combinations[key][mean_col] = row_data[mean_col]
            combinations[key][se_col] = row_data[se_col]
            all_measures.add(measure)
        
        # Convert to DataFrame
        final_df = pd.DataFrame(list(combinations.values()))
        
        # Ensure all measure columns exist (fill with NaN if missing)
        for measure in all_measures:
            seeds_col = f"{measure}_Seeds"
            mean_col = f"{measure}_Mean"
            se_col = f"{measure}_SE"
            
            if seeds_col not in final_df.columns:
                final_df[seeds_col] = np.nan
            if mean_col not in final_df.columns:
                final_df[mean_col] = np.nan
            if se_col not in final_df.columns:
                final_df[se_col] = np.nan
        
        # Sort columns: identifiers first, then measures in alphabetical order
        identifier_cols = ['Model', 'k', 'Initialization_Method']
        measure_cols = []
        
        for measure in sorted(all_measures):
            measure_cols.extend([
                f"{measure}_Seeds",
                f"{measure}_Mean", 
                f"{measure}_SE"
            ])
        
        final_df = final_df[identifier_cols + measure_cols]
        
        # Sort rows
        final_df = final_df.sort_values(['Model', 'k', 'Initialization_Method'])
        
        # Save to file if specified
        if output_file:
            output_path = self.csv_base_dir / output_file
            final_df.to_csv(output_path, index=False)
            print(f"Seeds/Mean/SE table saved to: {output_path}")
        
        return final_df
    
    def calculate_statistics(self, data_points):
        """Calculate mean and standard error for grouped data (original method)."""
        df = pd.DataFrame(data_points)
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by Model, k, Initialization_Method, and Measure
        grouped = df.groupby(['Model', 'k', 'Initialization_Method', 'Measure'])
        
        results = []
        
        for name, group in grouped:
            model, k, init_method, measure = name
            values = group['Value'].values
            
            if len(values) == 0:
                continue
            
            mean_value = np.mean(values)
            std_error = stats.sem(values) if len(values) > 1 else 0.0
            
            result = {
                'Model': model,
                'k': k,
                'Initialization_Method': init_method,
                'Measure': measure,
                'Mean': mean_value,
                'SE': std_error,
                'n_seeds': len(values),
                'Seeds_Used': ', '.join([self.extract_seed_from_filename(f) for f in group['Filename']]),
                'Values_Aggregated': ', '.join([f'{v:.6f}' for v in values])
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def create_summary_table(self, output_file=None):
        """Create the main summary table with statistics (original method)."""
        print("Collecting data from all CSV files...")
        
        # Collect all data points
        data_points = self.collect_all_data()
        
        if not data_points:
            print("No data points found!")
            return None
        
        print(f"Found {len(data_points)} data points")
        
        # Calculate statistics
        print("Calculating statistics...")
        results_df = self.calculate_statistics(data_points)
        
        if results_df.empty:
            print("No statistics could be calculated!")
            return None
        
        # Sort the results for better readability
        results_df = results_df.sort_values(['Model', 'k', 'Initialization_Method', 'Measure'])
        
        # Save to file if specified
        if output_file:
            output_path = self.csv_base_dir / output_file
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return results_df
    
    def print_summary_statistics(self, results_df):
        """Print summary statistics about the data."""
        if results_df.empty:
            return
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        if 'n_seeds' in results_df.columns:
            # Original format
            print(f"Total combinations: {len(results_df)}")
            print(f"Models: {sorted(results_df['Model'].unique())}")
            print(f"K values: {sorted(results_df['k'].unique())}")
            print(f"Initialization methods: {sorted(results_df['Initialization_Method'].unique())}")
            print(f"Measures: {sorted(results_df['Measure'].unique())}")
            
            print(f"\nSeeds per combination (mean): {results_df['n_seeds'].mean():.1f}")
            print(f"Combinations with all seeds: {len(results_df[results_df['n_seeds'] == results_df['n_seeds'].max()])}")
        else:
            # Seeds/Mean/SE format
            print(f"Total combinations: {len(results_df)}")
            print(f"Models: {sorted(results_df['Model'].unique())}")
            print(f"K values: {sorted(results_df['k'].unique())}")
            print(f"Initialization methods: {sorted(results_df['Initialization_Method'].unique())}")
            
            # Count measures from column names
            measure_cols = [col for col in results_df.columns if col.endswith('_Mean')]
            measures = [col.replace('_Mean', '') for col in measure_cols]
            print(f"Measures: {sorted(measures)}")

def main():
    parser = argparse.ArgumentParser(description='Aggregate IWAE data and calculate statistics')
    parser.add_argument('csv_dir', nargs='?', default='runs/Second_Run/csv',
                       help='Directory containing CSV files (default: runs/Second_Run/csv)')
    parser.add_argument('--output', '-o', default='aggregated_results.csv',
                       help='Output file name (default: aggregated_results.csv)')
    parser.add_argument('--seeds-table', action='store_true',
                       help='Create Seeds/Mean/SE table format instead of standard format')
    parser.add_argument('--seeds-output', default='seeds_mean_se_table.csv',
                       help='Output file name for Seeds/Mean/SE table (default: seeds_mean_se_table.csv)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information for filename parsing')
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = IWAEDataAggregator(args.csv_dir)
    
    if args.debug:
        print("Debug mode: Testing filename parsing...")
        csv_path = Path(args.csv_dir)
        for subdir in ['test_activity', 'training_activity', 'test_loss', 'training_loss']:
            dir_path = csv_path / subdir
            if dir_path.exists():
                print(f"\n{subdir}:")
                for csv_file in dir_path.glob('*.csv'):
                    if not csv_file.name.startswith('combined_'):
                        parsed = aggregator.parse_filename(csv_file.name)
                        seed = aggregator.extract_seed_from_filename(csv_file.name)
                        print(f"  {csv_file.name}")
                        print(f"    Model: {parsed['model']}, k: {parsed['k']}, Init: {parsed['initialization']}, Seed: {seed}")
        return
    
    print(f"Processing CSV files from: {args.csv_dir}")
    
    if args.seeds_table:
        # Create Seeds/Mean/SE table
        print("Creating Seeds/Mean/SE table...")
        data_points = aggregator.collect_all_data()
        results_df = aggregator.create_seeds_mean_se_table(data_points, args.seeds_output)
        
        if not results_df.empty:
            # Print statistics
            aggregator.print_summary_statistics(results_df)
            
            # Show sample of results
            print(f"\nSample results:")
            print(results_df.to_string(index=False))
            
            print(f"\n✓ Processing completed!")
            print(f"Seeds/Mean/SE table saved to: {Path(args.csv_dir) / args.seeds_output}")
        else:
            print("No results generated. Check your CSV directory and file structure.")
    
    else:
        # Create standard summary table
        results_df = aggregator.create_summary_table(args.output)
        
        if results_df is not None:
            # Print statistics
            aggregator.print_summary_statistics(results_df)
            
            # Show sample of results
            print(f"\nSample results (first 10 rows):")
            print(results_df.head(10).to_string(index=False))
            
            print(f"\n✓ Processing completed!")
            print(f"Main results saved to: {Path(args.csv_dir) / args.output}")
        
        else:
            print("No results generated. Check your CSV directory and file structure.")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Create Seeds/Mean/SE table
python iwae_data_aggregator.py --seeds-table

# Create Seeds/Mean/SE table with custom output file
python iwae_data_aggregator.py --seeds-table --seeds-output my_seeds_table.csv

# Debug filename and seed parsing
python iwae_data_aggregator.py --debug

# Standard usage (original format)
python iwae_data_aggregator.py

# Custom CSV directory with Seeds/Mean/SE table
python iwae_data_aggregator.py /path/to/csv/files --seeds-table
"""