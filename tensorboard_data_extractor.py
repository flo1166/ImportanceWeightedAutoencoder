"""
Hierarchical IWAE TensorBoard Exporter
Processes TensorBoard data from nested runs/Second_Run/[seed]/[model]/[run] structure
and organizes it in runs/Second_Run/csv with seed and model information preserved.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from datetime import datetime
import shutil
from PIL import Image
import io

class HierarchicalIWAETensorBoardExporter:
    def __init__(self, base_run_dir):
        self.base_run_dir = Path(base_run_dir)
        self.output_base_dir = self.base_run_dir / "csv"
        
        # Seeds and models to process
        self.seeds = [10, 32, 135, 630, 924]
        self.models = ["VAE", "IWAE"]
        
        # Define the mapping from panel names to output structure
        self.scalar_mappings = {
            # Activity mappings
            "EPOCH_DIM/Active Latent Dimensions - Test": "test_activity",
            "EPOCH_DIM/Active Latent Dimensions - Training": "training_activity",
            
            # Test loss mappings
            "EPOCH_LOSS/Avg. Test Loss (KL Div)": "test_loss/KL",
            "EPOCH_LOSS/Avg. Test Loss (NLL)": "test_loss/NLL", 
            "EPOCH_LOSS/Avg. Test Loss (NLL + KL Div)": "test_loss/NLL+KL",
            "EPOCH_LOSS/Standard Error Test Loss (KL Div)": "test_loss/SE_KL",
            "EPOCH_LOSS/Standard Error Test Loss (NLL)": "test_loss/SE_NLL",
            "EPOCH_LOSS/Standard Error Test Loss (NLL + KL Div)": "test_loss/SE_NLL+KL",
            
            # Training loss mappings
            "EPOCH_LOSS/Avg. Training Loss (KL Div)": "training_loss/KL",
            "EPOCH_LOSS/Avg. Training Loss (NLL)": "training_loss/NLL",
            "EPOCH_LOSS/Avg. Training Loss (NLL + KL Div)": "training_loss/NLL+KL",
            "EPOCH_LOSS/Standard Error Training Loss (KL Div)": "training_loss/SE_KL",
            "EPOCH_LOSS/Standard Error Training Loss (NLL)": "training_loss/SE_NLL",
            "EPOCH_LOSS/Standard Error Training Loss (NLL + KL Div)": "training_loss/SE_NLL+KL",
        }
        
        # Create output directory structure
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create the required directory structure."""
        directories = [
            "test_activity",
            "training_activity", 
            "test_loss",
            "training_loss",
            "reconstruction"
        ]
        
        for directory in directories:
            (self.output_base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def find_run_directories(self):
        """Find all run directories following the hierarchy: seed/model/run."""
        run_dirs = []
        
        for seed in self.seeds:
            seed_dir = self.base_run_dir / str(seed)
            if not seed_dir.exists():
                print(f"Warning: Seed directory {seed} not found")
                continue
                
            for model in self.models:
                model_dir = seed_dir / model
                if not model_dir.exists():
                    print(f"Warning: Model directory {seed}/{model} not found")
                    continue
                
                # Find all run directories in this model directory
                for run_dir in model_dir.iterdir():
                    if run_dir.is_dir() and any(run_dir.glob('events.out.tfevents.*')):
                        run_info = {
                            'path': run_dir,
                            'seed': seed,
                            'model': model,
                            'run_name': run_dir.name,
                            'full_identifier': run_dir.name  # Just use the directory name
                        }
                        run_dirs.append(run_info)
        
        return run_dirs
    
    def extract_run_data(self, run_info):
        """Extract scalar and image data from a single TensorBoard run."""
        logdir = run_info['path']
        identifier = run_info['full_identifier']
        
        event_acc = EventAccumulator(str(logdir), size_guidance={
            'scalars': 0,  # Load all scalars
            'images': 0,   # Load all images
        })
        event_acc.Reload()
        
        # Get available tags
        scalar_tags = event_acc.Tags().get('scalars', [])
        image_tags = event_acc.Tags().get('images', [])
        
        print(f"  Found {len(scalar_tags)} scalar tags and {len(image_tags)} image tags")
        
        # Process scalar data
        self.process_scalar_data(event_acc, scalar_tags, run_info)
        
        # Process image data (reconstruction)
        self.process_image_data(event_acc, image_tags, run_info)
    
    def process_scalar_data(self, event_acc, scalar_tags, run_info):
        """Process scalar data according to the mapping."""
        identifier = run_info['full_identifier']
        seed = run_info['seed']
        model = run_info['model']
        
        for panel_name, output_path in self.scalar_mappings.items():
            if panel_name in scalar_tags:
                # Extract scalar data
                scalars = event_acc.Scalars(panel_name)
                
                if scalars:
                    # Create DataFrame with additional metadata
                    data = []
                    for scalar in scalars:
                        data.append({
                            'step': scalar.step,
                            'value': scalar.value,
                            'wall_time': scalar.wall_time,
                            'timestamp': datetime.fromtimestamp(scalar.wall_time),
                            'seed': seed,
                            'model': model,
                            'run_name': run_info['run_name']
                        })
                    
                    df = pd.DataFrame(data)
                    
                    # Determine output file path
                    if "/" in output_path:
                        # Create subdirectory if needed
                        subdir, metric = output_path.split("/", 1)
                        output_dir = self.output_base_dir / subdir
                        output_dir.mkdir(exist_ok=True)
                        csv_path = output_dir / f"{identifier}_{metric.replace(' ', '_')}.csv"
                    else:
                        csv_path = self.output_base_dir / output_path / f"{identifier}.csv"
                    
                    # Save CSV
                    df.to_csv(csv_path, index=False)
                    print(f"    Saved: {csv_path.relative_to(self.base_run_dir)}")
                else:
                    print(f"    No data found for: {panel_name}")
            else:
                print(f"    Missing panel: {panel_name}")
    
    def process_image_data(self, event_acc, image_tags, run_info):
        """Process image data for reconstruction."""
        identifier = run_info['full_identifier']
        
        reconstruction_tags = [tag for tag in image_tags if 'reconstruction' in tag.lower()]
        
        if not reconstruction_tags:
            print(f"    No reconstruction images found for {identifier}")
            return
        
        # Create run-specific reconstruction directory
        recon_dir = self.output_base_dir / "reconstruction" / identifier
        recon_dir.mkdir(parents=True, exist_ok=True)
        
        for tag in reconstruction_tags:
            images = event_acc.Images(tag)
            
            for i, image_event in enumerate(images):
                # Decode image
                image_data = image_event.encoded_image_string
                image = Image.open(io.BytesIO(image_data))
                
                # Create filename
                step = image_event.step
                safe_tag = tag.replace('/', '_').replace('\\', '_')
                image_filename = f"epoch_{step:04d}_{safe_tag}.png"
                image_path = recon_dir / image_filename
                
                # Save image
                image.save(image_path)
            
            print(f"    Saved {len(images)} reconstruction images for {tag}")
    
    def export_all_runs(self):
        """Export all runs found in the hierarchy."""
        run_dirs = self.find_run_directories()
        
        if not run_dirs:
            print("No run directories found!")
            return
        
        print(f"Found {len(run_dirs)} runs to process:")
        
        # Group by seed and model for summary
        summary = {}
        for run_info in run_dirs:
            key = f"seed_{run_info['seed']}_{run_info['model']}"
            if key not in summary:
                summary[key] = []
            summary[key].append(run_info['run_name'])
        
        for key, runs in summary.items():
            print(f"  {key}: {len(runs)} runs")
        
        print("\nProcessing runs...")
        
        successful = 0
        failed = 0
        
        for run_info in run_dirs:
            print(f"\nProcessing: {run_info['run_name']} (seed_{run_info['seed']}_{run_info['model']})")
            print(f"  Path: {run_info['path'].relative_to(self.base_run_dir)}")
            
            try:
                self.extract_run_data(run_info)
                print(f"  ✓ Successfully processed")
                successful += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1
        
        print(f"\n=== Summary ===")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(run_dirs)}")
    
    def create_combined_csvs(self):
        """Create combined CSV files for each metric across all seeds and models."""
        print("\nCreating combined CSV files...")
        
        # Define metric categories
        metric_categories = {
            'test_activity': self.output_base_dir / 'test_activity',
            'training_activity': self.output_base_dir / 'training_activity',
            'test_loss': self.output_base_dir / 'test_loss',
            'training_loss': self.output_base_dir / 'training_loss'
        }
        
        for category, category_dir in metric_categories.items():
            if not category_dir.exists():
                continue
                
            # Group files by metric type
            metric_files = {}
            for csv_file in category_dir.glob('*.csv'):
                if category in ['test_loss', 'training_loss']:
                    # Extract metric name from filename (everything after the run name)
                    if '_' in csv_file.stem:
                        parts = csv_file.stem.split('_')
                        # Find the last part as the metric
                        metric = parts[-1]
                    else:
                        metric = 'unknown'
                else:
                    metric = 'activity'
                
                if metric not in metric_files:
                    metric_files[metric] = []
                metric_files[metric].append(csv_file)
            
            # Combine files for each metric
            for metric, files in metric_files.items():
                if len(files) <= 1:
                    continue
                    
                combined_data = []
                for file_path in files:
                    df = pd.read_csv(file_path)
                    combined_data.append(df)
                
                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    combined_file = category_dir / f"combined_{metric}.csv"
                    combined_df.to_csv(combined_file, index=False)
                    print(f"  Created: {combined_file.relative_to(self.base_run_dir)}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        summary_file = self.output_base_dir / "export_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Hierarchical IWAE TensorBoard Export Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base directory: {self.base_run_dir}\n")
            f.write("=" * 60 + "\n\n")
            
            # Count files by seed and model
            f.write("Files by Category:\n")
            f.write("-" * 20 + "\n")
            
            for subdir in self.output_base_dir.iterdir():
                if subdir.is_dir():
                    if subdir.name == "reconstruction":
                        # Count reconstruction directories
                        recon_dirs = [d for d in subdir.iterdir() if d.is_dir()]
                        f.write(f"{subdir.name}/: {len(recon_dirs)} run directories\n")
                        
                        # Group by seed and model
                        seed_model_count = {}
                        for recon_dir in recon_dirs:
                            # Just count by directory name since we're not prefixing anymore
                            seed_model_count[recon_dir.name] = seed_model_count.get(recon_dir.name, 0) + 1
                        
                        for dir_name, count in sorted(seed_model_count.items()):
                            images = list((subdir / dir_name).glob("*.png"))
                            f.write(f"  - {dir_name}: {len(images)} images\n")
                    else:
                        csv_files = list(subdir.glob('*.csv'))
                        combined_files = [f for f in csv_files if f.name.startswith('combined_')]
                        individual_files = [f for f in csv_files if not f.name.startswith('combined_')]
                        
                        f.write(f"{subdir.name}/: {len(individual_files)} individual CSV files")
                        if combined_files:
                            f.write(f", {len(combined_files)} combined files")
                        f.write("\n")
            
            f.write(f"\nExpected structure processed:\n")
            for seed in self.seeds:
                for model in self.models:
                    f.write(f"  - {seed}/{model}/\n")
        
        print(f"Summary report saved to: {summary_file.relative_to(self.base_run_dir)}")

def main():
    parser = argparse.ArgumentParser(description='Export hierarchical IWAE TensorBoard data')
    parser.add_argument('run_dir', nargs='?', default='runs/Second_Run',
                       help='Base run directory (default: runs/Second_Run)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 32, 135, 630, 924],
                       help='Seeds to process (default: 10 32 135 630 924)')
    parser.add_argument('--models', nargs='+', default=['VAE', 'IWAE'],
                       help='Models to process (default: VAE IWAE)')
    parser.add_argument('--no-combined', action='store_true',
                       help='Skip creating combined CSV files')
    parser.add_argument('--list-structure', action='store_true',
                       help='List the directory structure without processing')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = HierarchicalIWAETensorBoardExporter(args.run_dir)
    exporter.seeds = args.seeds
    exporter.models = args.models
    
    if args.list_structure:
        run_dirs = exporter.find_run_directories()
        print(f"Found directory structure in {args.run_dir}:")
        print("=" * 50)
        
        current_seed_model = None
        for run_info in run_dirs:
            seed_model = f"{run_info['seed']}/{run_info['model']}"
            if seed_model != current_seed_model:
                print(f"\n{seed_model}/")
                current_seed_model = seed_model
            print(f"  └── {run_info['run_name']}")
        
        return
    
    print(f"Processing TensorBoard data from: {exporter.base_run_dir}")
    print(f"Output directory: {exporter.output_base_dir}")
    print(f"Seeds: {exporter.seeds}")
    print(f"Models: {exporter.models}")
    
    # Process all runs
    exporter.export_all_runs()
    
    # Create combined files unless disabled
    if not args.no_combined:
        exporter.create_combined_csvs()
    
    # Create summary
    exporter.create_summary_report()
    
    print(f"\n✓ Export completed! All data saved to: {exporter.output_base_dir}")
    print("\nFinal directory structure:")
    print("runs/Second_Run/csv/")
    print("├── test_activity/")
    print("│   ├── [run_name].csv")
    print("│   ├── [another_run_name].csv")
    print("│   ├── combined_activity.csv")
    print("│   └── ...")
    print("├── training_activity/")
    print("├── test_loss/")
    print("│   ├── [run_name]_KL.csv")
    print("│   ├── combined_KL.csv")
    print("│   └── ...")
    print("├── training_loss/")
    print("├── reconstruction/")
    print("│   ├── [run_name]/")
    print("│   │   └── epoch_*.png")
    print("│   └── ...")
    print("└── export_summary.txt")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Process default structure (runs/Second_Run)
python hierarchical_iwae_exporter.py

# Process custom directory
python hierarchical_iwae_exporter.py /path/to/your/runs/Second_Run

# Process only specific seeds and models
python hierarchical_iwae_exporter.py --seeds 10 32 --models VAE

# List directory structure without processing
python hierarchical_iwae_exporter.py --list-structure

# Skip creating combined CSV files
python hierarchical_iwae_exporter.py --no-combined
"""