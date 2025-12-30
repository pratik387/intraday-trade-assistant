#!/bin/bash
# Scale down your OCI node pool to 0 after backtest completes

# Basic cluster node pool (free control plane)
NODE_POOL_ID="ocid1.nodepool.oc1.ap-mumbai-1.aaaaaaaaqs7a4f5jyyhcy3dsmedknnzbmhpdmdj6dqkastv5cnaehilq5g3q"

echo "Checking for running backtest jobs..."

# Check if any backtest jobs are running
RUNNING_JOBS=$(kubectl get jobs -l app=backtest -o json 2>/dev/null | jq '[.items[] | select(.status.active > 0 or ((.status.succeeded // 0) + (.status.failed // 0)) < (.spec.completions // 0))] | length')

if [ -z "$RUNNING_JOBS" ]; then
    RUNNING_JOBS=0
fi

echo "Running jobs: $RUNNING_JOBS"

if [ "$RUNNING_JOBS" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "No jobs running. Scaling node pool to 0..."
    echo "============================================================"

    oci ce node-pool update \
        --node-pool-id "$NODE_POOL_ID" \
        --size 0 \
        --force

    echo ""
    echo "✓ Node pool scaled to 0"
    echo "✓ Nodes will terminate in ~5 minutes"
    echo "✓ Billing will stop once nodes are terminated"
    echo ""
else
    echo ""
    echo "Jobs still running ($RUNNING_JOBS). Not scaling down yet."
    echo "Run this script again after jobs complete."
    echo ""
fi
