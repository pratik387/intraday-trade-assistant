#!/usr/bin/env python
"""
Split consolidated preagg cache into monthly files for Lambda.
This avoids Lambda memory issues and allows downloading only needed months.
"""
import pandas as pd
import boto3
from pathlib import Path

s3 = boto3.client('s3', region_name='ap-south-1')
bucket = 'backtest-runs-pratikhegde'
config_hash = 'beae8f50'

print('Downloading consolidated cache from S3...')
s3.download_file(bucket, f'cache/{config_hash}/preagg_1m.feather', 'cache/preagg_temp.feather')

print('Loading cache...')
df = pd.read_feather('cache/preagg_temp.feather')
print(f'Loaded {len(df):,} rows')

# Parse dates
df['date_parsed'] = pd.to_datetime(df['date'])
if df['date_parsed'].dt.tz is not None:
    df['date_parsed'] = df['date_parsed'].dt.tz_localize(None)

# Get unique year-month combinations
df['year_month'] = df['date_parsed'].dt.to_period('M')
unique_months = sorted(df['year_month'].unique())

print(f'\nFound {len(unique_months)} unique months: {[str(m) for m in unique_months]}')

# Create monthly files
output_dir = Path('cache/monthly')
output_dir.mkdir(exist_ok=True)

for ym in unique_months:
    print(f'\nProcessing {ym}...')

    # Filter to this month
    mask = df['year_month'] == ym
    df_month = df[mask].copy()

    # Drop temp columns
    df_month = df_month.drop(columns=['date_parsed', 'year_month'])

    # Save as monthly file (uncompressed for Lambda compatibility)
    year, month = ym.year, ym.month
    filename = f'{year}_{month:02d}_1m.feather'
    filepath = output_dir / filename

    df_month.to_feather(filepath, compression='uncompressed')

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f'  Created {filename}: {len(df_month):,} rows, {size_mb:.1f} MB')

    # Upload to S3
    s3_key = f'cache/{config_hash}/monthly/{filename}'
    s3.upload_file(str(filepath), bucket, s3_key)
    print(f'  Uploaded to s3://{bucket}/{s3_key}')

print(f'\n✓ Split into {len(unique_months)} monthly files and uploaded to S3!')

# Cleanup
import os
os.remove('cache/preagg_temp.feather')
import shutil
shutil.rmtree(output_dir)
print('Temp files cleaned up')
