#!/usr/bin/env python3
import sys
import os

print("=== PYTHON PARQUET DIAGNOSTIC ===")

file_path = "competitions/Titanic/export/titanic_train_features.parquet"
print(f"File: {file_path}")
print(f"Size: {os.path.getsize(file_path):,} bytes")

# Test 1: PyArrow
try:
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    print(f"✅ PyArrow: {table.num_rows} rows, {table.num_columns} columns")
    print(f"   Schema: {table.schema}")
except Exception as e:
    print(f"❌ PyArrow failed: {e}")

# Test 2: Pandas
try:
    import pandas as pd
    df = pd.read_parquet(file_path)
    print(f"✅ Pandas: {df.shape} (rows, cols)")
    print(f"   Columns: {list(df.columns[:5])}")
    if len(df.columns) > 5:
        print(f"   ... and {len(df.columns)-5} more")
except Exception as e:
    print(f"❌ Pandas failed: {e}")

# Test 3: Raw file info
try:
    with open(file_path, 'rb') as f:
        header = f.read(50)
        print(f"✅ File header: {header[:20]}")
        print(f"   Magic bytes: {header[:4]}")
except Exception as e:
    print(f"❌ File read failed: {e}")