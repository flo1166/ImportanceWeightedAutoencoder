"""
IWAE Data Aggregator and Statistical Analysis
Aggregates data across seeds, calculates means and standard errors based on model and initialization.
Takes the final (last) value from each CSV file for analysis.
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
    
    def calculate_statistics(self, data_points):
        """Calculate mean and standard error for grouped data."""
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
            
            # Calculate standard error using scipy.stats.sem
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
        """Create the main summary table with statistics."""
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
    
    def create_pivot_tables(self, results_df, output_prefix="pivot"):
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
                
                mean_file = self.csv_base_dir / f"{output_prefix}_{safe_measure}_mean.csv"
                se_file = self.csv_base_dir / f"{output_prefix}_{safe_measure}_se.csv"
                
                pivot_mean.to_csv(mean_file)
                pivot_se.to_csv(se_file)
                
                print(f"Pivot tables saved: {mean_file.name}, {se_file.name}")
                
            except Exception as e:
                print(f"Could not create pivot table for {measure}: {e}")
    
    def print_summary_statistics(self, results_df):
        """Print summary statistics about the data."""
        if results_df.empty:
            return
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
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

def main():
    parser = argparse.ArgumentParser(description='Aggregate IWAE data and calculate statistics')
    parser.add_argument('csv_dir', nargs='?', default='runs/Second_Run/csv',
                       help='Directory containing CSV files (default: runs/Second_Run/csv)')
    parser.add_argument('--output', '-o', default='aggregated_results.csv',
                       help='Output file name (default: aggregated_results.csv)')
    parser.add_argument('--pivot', action='store_true',
                       help='Create pivot tables for each measure')
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
                        print(f"  {csv_file.name}")
                        print(f"    Model: {parsed['model']}, k: {parsed['k']}, Init: {parsed['initialization']}")
        return
    
    print(f"Processing CSV files from: {args.csv_dir}")
    
    # Create summary table
    results_df = aggregator.create_summary_table(args.output)
    
    if results_df is not None:
        # Print statistics
        aggregator.print_summary_statistics(results_df)
        
        # Create pivot tables if requested
        if args.pivot:
            print("\nCreating pivot tables...")
            aggregator.create_pivot_tables(results_df)
        
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
# Basic usage
python iwae_data_aggregator.py

# Custom CSV directory
python iwae_data_aggregator.py /path/to/csv/files

# Custom output file and create pivot tables
python iwae_data_aggregator.py --output my_results.csv --pivot

# Debug filename parsing
python iwae_data_aggregator.py --debug

# Full example with all options
python iwae_data_aggregator.py runs/Second_Run/csv --output final_results.csv --pivot
"""