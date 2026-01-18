#!/bin/bash
#
# Oracle Cloud Setup Script
# ==========================
#
# Complete automated setup for OCI parallel backtesting.
# Run this on a laptop with Docker installed (not Windows machine).
#
# Usage:
#   bash oci_cloud/tools/setup_oci.sh
#

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Oracle Cloud Parallel Backtesting - Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check prerequisites
echo "[1/8] Checking prerequisites..."

if ! command -v oci &> /dev/null; then
    echo "❌ OCI CLI not found"
    echo "Install: bash -c \"\$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)\""
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    echo "Install: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found"
    echo "Install: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo "  ✓ OCI CLI installed"
echo "  ✓ Docker installed"
echo "  ✓ kubectl installed"

# Get compartment ID
echo
echo "[2/8] Getting OCI configuration..."

export COMPARTMENT_ID=$(oci iam compartment list --all --query 'data[0].id' --raw-output 2>/dev/null)
export TENANCY_NAMESPACE=$(oci os ns get --query data --raw-output 2>/dev/null)

if [ -z "$COMPARTMENT_ID" ]; then
    echo "❌ Could not get compartment ID"
    echo "Make sure OCI CLI is configured: oci setup config"
    exit 1
fi

echo "  ✓ Compartment ID: ${COMPARTMENT_ID:0:20}..."
echo "  ✓ Tenancy namespace: $TENANCY_NAMESPACE"

# Create Object Storage buckets
echo
echo "[3/8] Creating Object Storage buckets..."

for bucket in backtest-cache backtest-code backtest-results; do
    if oci os bucket get --name $bucket &> /dev/null; then
        echo "  ✓ Bucket exists: $bucket"
    else
        oci os bucket create \
            --compartment-id $COMPARTMENT_ID \
            --name $bucket \
            --storage-tier Standard \
            --public-access-type NoPublicAccess
        echo "  ✓ Created bucket: $bucket"
    fi
done

# Migrate cache from AWS (interactive)
echo
echo "[4/8] Migrating cache from AWS to OCI..."
echo
read -p "Do you want to migrate cache from AWS S3 now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Downloading cache from AWS S3..."
    mkdir -p ~/cache_temp
    aws s3 sync s3://backtest-runs-pratikhegde/cache/ ~/cache_temp/

    echo "  Uploading cache to OCI Object Storage..."
    oci os object bulk-upload \
        --bucket-name backtest-cache \
        --src-dir ~/cache_temp/ \
        --parallel-upload-count 10

    echo "  ✓ Cache migrated"

    rm -rf ~/cache_temp
else
    echo "  ⊘ Skipped cache migration (you can do this later)"
fi

# Create OKE cluster (interactive)
echo
echo "[5/8] Creating OKE cluster..."
echo "  This requires manual setup in OCI Console"
echo
echo "  Steps:"
echo "  1. Go to: https://cloud.oracle.com"
echo "  2. Developer Services → Kubernetes Clusters (OKE)"
echo "  3. Click 'Create Cluster' → 'Quick Create'"
echo "  4. Name: backtest-cluster"
echo "  5. Node pool: 15 nodes × VM.Standard.E4.Flex (16 OCPU, 64 GB)"
echo "  6. ✅ Enable Spot Instances"
echo "  7. Click Create (wait 10-15 minutes)"
echo
read -p "Have you created the cluster? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Please create OKE cluster first, then re-run this script"
    exit 1
fi

# Configure kubectl
echo
echo "[6/8] Configuring kubectl..."

CLUSTER_ID=$(oci ce cluster list \
    --compartment-id $COMPARTMENT_ID \
    --name backtest-cluster \
    --query 'data[0].id' --raw-output 2>/dev/null)

if [ -z "$CLUSTER_ID" ]; then
    echo "❌ Cluster 'backtest-cluster' not found"
    exit 1
fi

mkdir -p ~/.kube

oci ce cluster create-kubeconfig \
    --cluster-id $CLUSTER_ID \
    --file ~/.kube/config \
    --region ap-mumbai-1 \
    --token-version 2.0.0 \
    --kube-endpoint PUBLIC_ENDPOINT

echo "  ✓ kubectl configured"

# Test kubectl
if kubectl get nodes &> /dev/null; then
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    echo "  ✓ Connected to cluster ($NODE_COUNT nodes)"
else
    echo "❌ Could not connect to cluster"
    exit 1
fi

# Create Container Registry repository
echo
echo "[7/8] Creating Container Registry repository..."

if oci artifacts container repository get \
    --repository-id "$(oci artifacts container repository list --compartment-id $COMPARTMENT_ID --display-name backtest-worker --query 'data.items[0].id' --raw-output 2>/dev/null)" \
    &> /dev/null; then
    echo "  ✓ Repository exists: backtest-worker"
else
    oci artifacts container repository create \
        --compartment-id $COMPARTMENT_ID \
        --display-name backtest-worker \
        --is-public false
    echo "  ✓ Created repository: backtest-worker"
fi

# Build and push Docker image
echo
echo "[8/8] Building and pushing Docker image..."
echo
echo "  ⚠️  You need an auth token to push images"
echo "  Generate one at: Profile → Auth Tokens → Generate Token"
echo
read -p "Do you have an auth token? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "  Please generate auth token first, then continue"
    echo "  URL: https://cloud.oracle.com/identity/domains/my-profile/auth-tokens"
    echo
    read -p "Press enter when ready..."
fi

# Docker login
echo "  Logging in to OCIR..."
echo "  Username: ${TENANCY_NAMESPACE}/oracleidentitycloudservice/<your-email>"
echo

docker login bom.ocir.io

# Build image
echo "  Building Docker image..."
cd $(dirname $0)/../..
docker build -f oci_cloud/docker/Dockerfile -t backtest-worker:latest .

# Tag for OCIR
docker tag backtest-worker:latest \
    bom.ocir.io/$TENANCY_NAMESPACE/backtest-worker:latest

# Push
echo "  Pushing to OCIR..."
docker push bom.ocir.io/$TENANCY_NAMESPACE/backtest-worker:latest

echo "  ✓ Docker image pushed"

# Summary
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Resources created:"
echo "  ✓ Object Storage buckets: backtest-cache, backtest-code, backtest-results"
echo "  ✓ OKE cluster: backtest-cluster (240 OCPU)"
echo "  ✓ Container image: bom.ocir.io/$TENANCY_NAMESPACE/backtest-worker:latest"
echo
echo "Next steps:"
echo "  1. On Windows machine: pip install oci-cli"
echo "  2. Copy ~/.oci/config to Windows machine"
echo "  3. Run: python tools/submit_oci_backtest.py --start 2024-05-01 --end 2024-10-31"
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
