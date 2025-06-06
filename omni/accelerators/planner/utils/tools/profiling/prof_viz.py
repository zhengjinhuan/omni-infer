import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import os
import glob
import re
from pathlib import Path
import numpy as np
import sys

def find_csv_files(main_path, file_pattern):
    """
    Find all CSV files matching the pattern in the main path.
    Supports both glob patterns and regex patterns (when prefixed with 'regex:').
    """
    if file_pattern.startswith('regex:'):
        # Use regex pattern
        pattern = file_pattern[6:]  # Remove 'regex:' prefix
        regex = re.compile(pattern)
        
        # Walk through all files in the directory
        matched_files = []
        for root, dirs, files in os.walk(main_path):
            for file in files:
                # Get relative path from main_path
                rel_path = os.path.relpath(os.path.join(root, file), main_path)
                if regex.search(rel_path):
                    matched_files.append(os.path.join(root, file))
        return matched_files
    else:
        # Use glob pattern
        pattern_path = os.path.join(main_path, file_pattern)
        files = glob.glob(pattern_path, recursive=True)
        # Filter to ensure the pattern *_step_<number>.csv
        filtered_files = []
        for f in files:
            if f.endswith('.csv') and re.search(r'_step_\d+\.csv$', f):
                filtered_files.append(f)
        return filtered_files

def get_operator_name(file_path):
    """
    Extract operator name from file path (text before first underscore).
    E.g., "GroupedMatmul_Duration(us)_step_1.csv" -> "GroupedMatmul"
    """
    basename = os.path.basename(file_path)
    return basename.split('_')[0]

def load_csv_data(file_path):
    """
    Load CSV data and prepare it for heatmap generation.
    Returns the data matrix and labels (if the first column is non-numeric).
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: CSV file is empty: {file_path}")
            return None, None
        
        # Check if the first column is numeric
        first_col_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 0])
        
        if first_col_numeric:
            # First column is numeric, treat all columns as data
            data = df
            labels = None
        else:
            # First column is not numeric, use it as labels
            labels = df.iloc[:, 0]
            data = df.iloc[:, 1:]
            
            if data.empty:
                print(f"Warning: No data columns in {file_path}")
                return None, None
                
        return data, labels
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

def calculate_max_values(csv_files, custom_rescale):
    """
    Calculate maximum values for rescaling based on the custom_rescale option.
    Returns a dictionary mapping file paths to their respective max values.
    """
    max_values = {}
    
    if custom_rescale == "false":
        # Each file uses its own maximum
        for file_path in csv_files:
            data, _ = load_csv_data(file_path)
            if data is not None:
                max_values[file_path] = data.max().max()
    
    elif custom_rescale == "by-operator":
        # Group by operator and find max across each group
        operator_groups = {}
        for file_path in csv_files:
            operator = get_operator_name(file_path)
            if operator not in operator_groups:
                operator_groups[operator] = []
            operator_groups[operator].append(file_path)
        
        # Find max for each operator group
        for operator, files in operator_groups.items():
            group_max = 0
            for file_path in files:
                data, _ = load_csv_data(file_path)
                if data is not None:
                    file_max = data.max().max()
                    group_max = max(group_max, file_max)
            
            # Assign the group max to all files in the group
            for file_path in files:
                max_values[file_path] = group_max
    
    return max_values

def generate_heatmap(data, labels, output_file, vmax=None):
    """
    Generate a heatmap using matplotlib's imshow.
    """
    # Calculate figure dimensions with size limits
    dpi = 100
    pixels_per_row = 30
    max_pixels = 65535  # Maximum allowed by matplotlib (2^16 - 1)
    
    # Calculate desired height
    desired_height_pixels = len(data) * pixels_per_row
    
    # If too tall, adjust pixels per row
    if desired_height_pixels > max_pixels:
        pixels_per_row = max_pixels // len(data)
        # Ensure at least 1 pixel per row
        pixels_per_row = max(1, pixels_per_row)
        desired_height_pixels = len(data) * pixels_per_row
        print(f"Warning: Large dataset ({len(data)} rows). Adjusting row height to {pixels_per_row} pixels.")
    
    height_in_inches = desired_height_pixels / dpi
    width_in_inches = max(10, min(30, data.shape[1] * 0.8))  # Dynamic width with limits
    
    # Further limit the total size to prevent memory issues
    max_height_inches = 100  # Maximum height in inches
    if height_in_inches > max_height_inches:
        height_in_inches = max_height_inches
        print(f"Warning: Further limiting height to {max_height_inches} inches to prevent memory issues.")
    
    fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    
    # Convert data to numpy array for plotting
    data_array = data.values
    
    # Create the heatmap using imshow
    im = ax.imshow(data_array, aspect='auto', cmap='YlOrRd', vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Duration (us)')
    
    # Set ticks and labels
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f'Rank_{i}' for i in range(data.shape[1])], rotation=45, ha='right')
    
    # For very large datasets, reduce the number of y-axis labels
    if len(data) > 100:
        # Show every nth label
        n = max(1, len(data) // 50)  # Show approximately 50 labels
        y_positions = range(0, len(data), n)
        if labels is not None:
            y_labels = [labels.iloc[i] for i in y_positions]
        else:
            y_labels = [str(i) for i in y_positions]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
    else:
        if labels is not None:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
        else:
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(range(len(data)))
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Duration' if labels is not None else 'Row')
    ax.set_title(f'Duration Heatmap - {os.path.basename(output_file)}')
    
    # Add grid for better readability (only for smaller datasets)
    if len(data) < 100:
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(data)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        # Try with lower DPI
        try:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"Heatmap saved as {output_file} (reduced quality)")
        except Exception as e2:
            print(f"Failed to save {output_file}: {e2}")
    finally:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps for CSV files with dynamic scaling.')
    parser.add_argument('--main-path', required=True, help='Main directory path to search for files')
    parser.add_argument('--file-pattern', default='regex:round_*/analysis/*_step_*.csv', 
                       help='File pattern to match. For regex, prefix with "regex:" (default: round_*/analysis/*_step_*.csv)')
    parser.add_argument('--custom-rescale', choices=['false', 'by-operator', 'user-defined'], 
                       default='false', help='Rescaling method for heatmaps')
    parser.add_argument('--rescale-factor', type=float, help='Rescale factor (required when custom-rescale is user-defined)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.custom_rescale == 'user-defined' and args.rescale_factor is None:
        print("Error: --rescale-factor is required when --custom-rescale is 'user-defined'")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = find_csv_files(args.main_path, args.file_pattern)
    if not csv_files:
        print(f"No CSV files found matching pattern: {args.file_pattern} in {args.main_path}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Calculate max values based on rescaling option
    if args.custom_rescale == 'user-defined':
        max_values = {f: args.rescale_factor for f in csv_files}
    else:
        max_values = calculate_max_values(csv_files, args.custom_rescale)
    
    # Generate heatmaps
    for file_path in csv_files:
        data, labels = load_csv_data(file_path)
        if data is None:
            continue
        
        # Create output path
        rel_path = os.path.relpath(file_path, args.main_path)
        output_path = os.path.join(args.main_path, 'heatmaps', rel_path.replace('.csv', '.png'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate heatmap
        vmax = max_values.get(file_path, None)
        generate_heatmap(data, labels, output_path, vmax=vmax)

if __name__ == '__main__':
    main()