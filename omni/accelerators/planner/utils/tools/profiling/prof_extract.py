import pandas as pd
import argparse
import sys
import os
import glob
import re

def extract_columns(file_pattern, columns=None, destination=None):
    """
    Extract specified columns from CSV files matching a pattern and concatenate them horizontally.
    Columns in parentheses are extracted from the first CSV only; others from all CSVs.
    
    Parameters:
        file_pattern (str): Glob pattern for CSV files (e.g., 'data/prof/*/kernel_details.csv')
        columns (tuple): (first_csv_columns, all_csv_columns) where each is a list of names/indices
        destination (str): Path to save the concatenated columns as a CSV
    
    Returns:
        None (saves to destination)
    """
    try:
        # Find all CSV files matching the pattern recursively
        csv_files = glob.glob(file_pattern, recursive=True)
        if not csv_files:
            print(f"Error: No CSV files found matching pattern '{file_pattern}'.")
            return None
        
        # Read the first CSV to validate columns
        first_df = pd.read_csv(csv_files[0])
        all_columns = first_df.columns.tolist()
        
        # Split columns into first_csv_columns and all_csv_columns
        first_csv_columns, all_csv_columns = columns
        
        # Validate requested columns
        selected_columns = first_csv_columns + all_csv_columns
        valid_columns = []
        invalid_columns = []
        
        for col in selected_columns:
            if isinstance(col, int):
                if 0 <= col < len(all_columns):
                    valid_columns.append(all_columns[col])
                else:
                    invalid_columns.append(f"Index {col}")
            elif isinstance(col, str):
                if col in all_columns:
                    valid_columns.append(col)
                else:
                    invalid_columns.append(f"Name '{col}'")
            else:
                invalid_columns.append(str(col))
        
        # Report any invalid columns
        if invalid_columns:
            print(f"Error: The following columns were invalid: {', '.join(invalid_columns)}")
            print("Available columns (based on first CSV):")
            for idx, col in enumerate(all_columns):
                print(f"{idx}: {col}")
            return None
        
        first_csv_valid_cols = [valid_columns[i] for i in range(len(first_csv_columns))]
        all_csv_valid_cols = [valid_columns[i] for i in range(len(first_csv_columns), len(valid_columns))]
        
        # Extract and concatenate columns from all CSVs
        concatenated_dfs = []
        for counter, csv_file in enumerate(csv_files, start=1):
            try:
                df = pd.read_csv(csv_file)
                # Verify column consistency
                if not all(col in df.columns for col in valid_columns):
                    print(f"Error: CSV '{csv_file}' is missing some requested columns.")
                    return None
                
                # Extract columns based on whether it's the first CSV
                if counter == 1:
                    # First CSV: both first_csv_columns and all_csv_columns
                    extracted_df = df[first_csv_valid_cols + all_csv_valid_cols]
                    extracted_df.columns = (
                        [f"{col}_{counter}" for col in first_csv_valid_cols] +
                        [f"{col}_{counter}" for col in all_csv_valid_cols]
                    )
                else:
                    # Other CSVs: only all_csv_columns
                    extracted_df = df[all_csv_valid_cols]
                    extracted_df.columns = [f"{col}_{counter}" for col in all_csv_valid_cols]
                
                concatenated_dfs.append(extracted_df)
            except Exception as e:
                print(f"Error reading '{csv_file}': {str(e)}")
                return None
        
        # Concatenate horizontally
        result_df = pd.concat(concatenated_dfs, axis=1)
        
        # Save to destination CSV with explicit newline after each row
        try:
            result_df.to_csv(destination, index=False, lineterminator='\n')
            print(f"Concatenated columns from {len(csv_files)} CSV(s) saved to '{destination}'")
        except Exception as e:
            print(f"Error: Failed to save to '{destination}': {str(e)}")
            return None
        
        return True
    
    except FileNotFoundError:
        print(f"Error: No files found for pattern '{file_pattern}'.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return None

def list_columns(file_pattern):
    """
    List all column names and indices from the first CSV matching the pattern.
    
    Parameters:
        file_pattern (str): Glob pattern for CSV files
    """
    try:
        # Find CSV files
        csv_files = glob.glob(file_pattern, recursive=True)
        if not csv_files:
            print(f"Error: No CSV files found matching pattern '{file_pattern}'.")
            return False
        
        # Read only the header of the first CSV
        df = pd.read_csv(csv_files[0], nrows=0)
        all_columns = df.columns.tolist()
        print("Available columns (based on first CSV):")
        for idx, col in enumerate(all_columns):
            print(f"{idx}: {col}")
        return True
    except FileNotFoundError:
        print(f"Error: No files found for pattern '{file_pattern}'.")
        return False
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return False

def parse_columns(columns_str):
    """
    Parse a string of column names or indices into a tuple of lists.
    Format: '(col1,col2),col3,col4' where (col1,col2) are for first CSV only, col3,col4 for all CSVs.
    
    Parameters:
        columns_str (str): String of columns, e.g., '(1,2),3,4'
    
    Returns:
        Tuple: (first_csv_columns, all_csv_columns)
    """
    try:
        # Match the pattern: optional (first_columns), followed by all_columns
        match = re.match(r'^\((.*?)\)(?:,(.*))?$', columns_str)
        if not match:
            print("Error: Invalid columns format. Use '(col1,col2),col3,col4' or 'col1,col2,col3'")
            print("Example: '(1,2),3,4' or '3,8,6'")
            return None
        
        first_part, second_part = match.groups()
        
        # Parse first CSV columns (inside parentheses)
        first_csv_columns = []
        if first_part:
            first_items = [item.strip() for item in first_part.split(',') if item.strip()]
            for item in first_items:
                try:
                    first_csv_columns.append(int(item))
                except ValueError:
                    first_csv_columns.append(item)
        
        # Parse all CSV columns (outside parentheses)
        all_csv_columns = []
        if second_part:
            all_items = [item.strip() for item in second_part.split(',') if item.strip()]
            for item in all_items:
                try:
                    all_csv_columns.append(int(item))
                except ValueError:
                    all_csv_columns.append(item)
        
        # If no parentheses, treat all as all_csv_columns (backward compatibility)
        if not first_part and not match.group(1):
            all_items = [item.strip() for item in columns_str.split(',') if item.strip()]
            for item in all_items:
                try:
                    all_csv_columns.append(int(item))
                except ValueError:
                    all_csv_columns.append(item)
            first_csv_columns = []
        
        return (first_csv_columns, all_csv_columns)
    except Exception as e:
        print(f"Error parsing columns: {str(e)}")
        print("Expected format: '(col1,col2),col3,col4' or 'col1,col2,col3'")
        print("Example: '(1,2),3,4' or '3,8,6'")
        return None

def print_usage():
    """
    Print usage instructions for the script.
    """
    print("Usage: python prof_extract.py -i INPUT_PATTERN [-l | -c COLUMNS -d DESTINATION]")
    print("Extract and concatenate columns from CSV files matching a pattern, or list available columns.")
    print("With -c '(col1,col2),col3,col4', col1,col2 are extracted from the first CSV only; col3,col4 from all CSVs.")
    print("\nOptions:")
    print("  -i, --input INPUT_PATTERN  Glob pattern for CSV files (e.g., 'data/prof/*/kernel_details.csv')")
    print("  -l, --list                 List all column names and indices from the first matching CSV, then exit")
    print("  -c, --columns COLUMNS      Columns to extract, e.g., '(1,2),3,4' or '3,8,6' (no parentheses: all from all CSVs)")
    print("  -d, --destination DESTINATION  Save concatenated columns to a CSV file (required with -c)")
    print("\nExamples:")
    print("  python prof_extract.py -i 'data/prof/*/kernel_details.csv' -l                            # List columns from first CSV")
    print("  python prof_extract.py -i 'data/prof/*/kernel_details.csv' -c '(1,2),3,4' -d output.csv  # 1,2 from first, 3,4 from all")
    print("  python prof_extract.py -i 'data/prof/*/kernel_details.csv' -c '(Task ID,Stream ID),Name' -d output.csv  # By names")
    print("  python prof_extract.py -i 'data/prof/*/kernel_details.csv' -c '1,2,3' -d output.csv      # All from all CSVs")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and concatenate columns from CSV files matching a pattern.", add_help=False)
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Glob pattern for CSV files (e.g., 'data/prof/*/kernel_details.csv')"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all column names and indices from the first matching CSV, then exit"
    )
    parser.add_argument(
        "-c", "--columns",
        default=None,
        help="Columns to extract, e.g., '(1,2),3,4' (first CSV: 1,2; all CSVs: 3,4) or '3,8,6' (all from all)"
    )
    parser.add_argument(
        "-d", "--destination",
        default=None,
        help="Save concatenated columns to a CSV file"
    )
    
    # Check if no arguments were provided
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Check if input pattern is provided
    if not args.input:
        print("Error: Input file pattern is required.")
        print_usage()
        sys.exit(1)
    
    # Handle list option
    if args.list:
        list_columns(args.input)
        sys.exit(0)
    
    # Handle extraction
    if not args.columns:
        print("Error: Columns are required unless listing with -l.")
        print_usage()
        sys.exit(1)
    
    if not args.destination:
        print("Error: Destination file is required when extracting columns.")
        print_usage()
        sys.exit(1)
    
    columns_to_extract = parse_columns(args.columns)
    if columns_to_extract is None:
        sys.exit(1)
    
    print(f"Extracting columns: {columns_to_extract}")
    extract_columns(args.input, columns_to_extract, args.destination)