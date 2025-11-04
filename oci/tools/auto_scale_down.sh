#!/bin/bash
# Auto-scale down nodes after backtest completion

CLUSTER_ID="YOUR_CLUSTER_ID"
NODE_POOL_ID="YOUR_NODE_POOL_ID"
COMPARTMENT_ID="YOUR_COMPARTMENT_ID"

echo "Checking for running jobs..."

# Check if any backtest jobs are running
RUNNING_JOBS=$(kubectl get jobs -o json | jq '[.items[] | select(.status.active > 0 or (.status.succeeded // 0) < (.spec.completions // 0))] | length')

if [ "$RUNNING_JOBS" -eq 0 ]; then
    echo "No jobs running. Scaling node pool to 0..."

    oci ce node-pool update \
        --node-pool-id "$NODE_POOL_ID" \
        --size 0 \
        --force

    echo "Node pool scaled to 0. Nodes will terminate in ~5 minutes."
else
    echo "Jobs still running ($RUNNING_JOBS). Not scaling down."
fi
