# Upload Regime Cache Files to OCI

## Current Status
- ✅ June 2024 (2024_06_1m.feather) - Already in OCI
- ✅ October 2024 (2024_10_1m.feather) - Already in OCI
- ❌ December 2023 (2023_12_1m.feather) - 260 MB - Need to upload
- ❌ January 2024 (2024_01_1m.feather) - 346 MB - Need to upload
- ❌ February 2025 (2025_02_1m.feather) - 276 MB - Need to upload
- ❌ July 2025 (2025_07_1m.feather) - 490 MB - Need to upload

## Method 1: Upload via OCI Cloud Shell (Easiest)

### Step 1: Compress cache files locally
```bash
# Windows (run from project root)
cd cache/preaggregate
tar -czf regime_caches.tar.gz 2023_12_1m.feather 2024_01_1m.feather 2025_02_1m.feather 2025_07_1m.feather
```

This creates a ~1.3 GB archive.

### Step 2: Upload to OCI Object Storage (temporary bucket)
```bash
# Upload to a temporary bucket first
oci os object put \
  --bucket-name backtest-cache \
  --name temp/regime_caches.tar.gz \
  --file cache/preaggregate/regime_caches.tar.gz
```

### Step 3: In OCI Cloud Shell, download and extract
```bash
# Download from temp location
cd ~
oci os object get \
  --bucket-name backtest-cache \
  --name temp/regime_caches.tar.gz \
  --file regime_caches.tar.gz

# Extract
tar -xzf regime_caches.tar.gz

# Upload to proper locations
python3 ~/intraday-trade-assistant/oci/tools/upload_regime_caches_from_local.py
```

## Method 2: Direct Upload from Local (if you install OCI CLI on Windows)

### Install OCI CLI on Windows
```bash
# PowerShell (as Administrator)
Set-ExecutionPolicy RemoteSigned
bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
```

### Configure OCI CLI
```bash
oci setup config
```

### Upload cache files
```bash
cd C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant
python oci/tools/upload_regime_caches.py
```

## Method 3: Manual Upload via OCI Console (Slowest but guaranteed to work)

1. Open OCI Console: https://cloud.oracle.com
2. Navigate to: Storage → Buckets → backtest-cache
3. Click "Upload"
4. For each file:
   - Object Name Prefix: `monthly/d0635bb8/`
   - Files to upload:
     - `2023_12_1m.feather`
     - `2024_01_1m.feather`
     - `2025_02_1m.feather`
     - `2025_07_1m.feather`

Note: This will take a while (4 files × ~300 MB average = ~1.2 GB total upload)

## Verify Upload

After uploading, verify all files are present:

```bash
# In OCI Cloud Shell or local (if OCI CLI installed)
oci os object list \
  --bucket-name backtest-cache \
  --prefix monthly/d0635bb8/ \
  --query 'data[].name' \
  --output table
```

You should see:
- monthly/d0635bb8/2023_12_1m.feather
- monthly/d0635bb8/2024_01_1m.feather
- monthly/d0635bb8/2024_06_1m.feather ✓
- monthly/d0635bb8/2024_10_1m.feather ✓
- monthly/d0635bb8/2025_02_1m.feather
- monthly/d0635bb8/2025_07_1m.feather

## After Upload Complete

Run the regime backtest:

```bash
# Test with 2 regimes first (the ones already cached)
python oci/tools/submit_regime_backtest.py \
  --regimes Event_Driven_HighVol Correction_RiskOff \
  --max-parallel 50

# After cache upload is complete, run all 6 regimes
python oci/tools/submit_regime_backtest.py --max-parallel 50
```
