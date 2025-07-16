#!/usr/bin/env python3
"""
Parquet to CSV Converter
Usage: python convert_parquet_to_csv.py <input.parquet> [output.csv]
"""

import sys
import os
from pathlib import Path

# Try different backends
try:
    import pandas as pd
    BACKEND = 'pandas'
except ImportError:
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        BACKEND = 'pyarrow'
    except ImportError:
        print("❌ Error: Neither pandas nor pyarrow is available")
        sys.exit(1)

def convert_parquet_to_csv(input_file, output_file=None):
    """Convert parquet file to CSV"""
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.csv'))
    
    try:
        print(f"📊 Loading parquet file: {input_file} (using {BACKEND})")
        
        if BACKEND == 'pandas':
            df = pd.read_parquet(input_file)
            print(f"✅ Loaded successfully: {df.shape} (rows, columns)")
            print(f"📋 Columns: {list(df.columns[:10])}")
            if len(df.columns) > 10:
                print(f"    ... and {len(df.columns) - 10} more columns")
            
            print(f"💾 Saving to CSV: {output_file}")
            df.to_csv(output_file, index=False)
            
        elif BACKEND == 'pyarrow':
            table = pq.read_table(input_file)
            print(f"✅ Loaded successfully: {table.num_rows} rows, {table.num_columns} columns")
            print(f"📋 Columns: {table.column_names[:10]}")
            if table.num_columns > 10:
                print(f"    ... and {table.num_columns - 10} more columns")
            
            print(f"💾 Saving to CSV: {output_file}")
            # Convert to pandas DataFrame for CSV writing
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(table.column_names)
                # Write data row by row
                for batch in table.to_batches():
                    pandas_df = batch.to_pandas()
                    for _, row in pandas_df.iterrows():
                        writer.writerow(row.values)
        
        print(f"✅ Conversion completed!")
        print(f"   Input:  {input_file} ({os.path.getsize(input_file):,} bytes)")
        print(f"   Output: {output_file} ({os.path.getsize(output_file):,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_parquet_to_csv.py <input.parquet> [output.csv]")
        print("Example: python convert_parquet_to_csv.py data.parquet")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("🔄 PARQUET TO CSV CONVERTER")
    print("=" * 60)
    
    success = convert_parquet_to_csv(input_file, output_file)
    
    if success:
        print("\n🎉 Conversion successful!")
        if output_file is None:
            output_file = str(Path(input_file).with_suffix('.csv'))
        print(f"\nYou can now use: julia universal_feature_selector.jl {output_file}")
    else:
        print("\n💥 Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()