#!/usr/bin/env python3
"""
Create OKE Basic Cluster (Free Control Plane)
==============================================

Creates a new Basic cluster alongside your existing Enhanced cluster.
Once validated, you can delete the old Enhanced cluster to save $72/month.

Usage:
    python oci/tools/create_basic_cluster.py --create-cluster
    python oci/tools/create_basic_cluster.py --create-nodepool
    python oci/tools/create_basic_cluster.py --create-secrets
    python oci/tools/create_basic_cluster.py --status

Steps:
    1. Run with --create-cluster (wait ~10 mins for cluster to be ACTIVE)
    2. Run with --create-nodepool (wait ~5 mins for nodes)
    3. Run with --create-secrets
    4. Update node_pool_id in submit scripts
    5. Test a backtest run
    6. If working, delete old Enhanced cluster from OCI Console
"""

import argparse
import subprocess
import json
import sys
import time
from pathlib import Path


# =============================================================================
# CONFIGURATION - Update these values from your existing cluster
# =============================================================================

# Compartment (your tenancy root)
COMPARTMENT_ID = "ocid1.tenancy.oc1..aaaaaaaasysw4isgo5u2wxxskjvkue4ofgo6ysqmc4bdc5zjexgw6icw57ma"

# Get these from your existing cluster (OCI Console > Kubernetes > backtest-cluster > Cluster Details)
# Or run: oci ce cluster get --cluster-id <your-cluster-ocid>
VCN_ID = None  # Will be fetched from existing cluster
KUBERNETES_VERSION = "v1.30.1"  # Use same as existing or latest

# Subnet OCIDs - Get from OCI Console > Networking > VCN > Subnets
# These should be the same subnets your existing cluster uses
ENDPOINT_SUBNET_ID = None  # Kubernetes API endpoint subnet (public or private)
NODE_SUBNET_ID = None  # Worker node subnet
POD_SUBNET_ID = None  # Pod networking subnet (can be same as node subnet)
SERVICE_LB_SUBNET_ID = None  # Load balancer subnet (if using services)

# Node pool configuration (match your existing)
NODE_SHAPE = "VM.Standard.E4.Flex"  # Or VM.Standard.A1.Flex for ARM
NODE_OCPUS = 2
NODE_MEMORY_GB = 32
NODE_IMAGE_ID = None  # Will use latest OKE image
INITIAL_NODE_COUNT = 0  # Start with 0, scale up when needed

# New cluster name
NEW_CLUSTER_NAME = "backtest-cluster-basic"

# =============================================================================


def run_oci_command(cmd, parse_json=True):
    """Run OCI CLI command and return output"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if parse_json and result.stdout.strip():
            return json.loads(result.stdout)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None


def get_existing_cluster_config():
    """Fetch configuration from existing Enhanced cluster"""
    print("Fetching existing cluster configuration...")

    # List clusters
    clusters = run_oci_command([
        'oci', 'ce', 'cluster', 'list',
        '--compartment-id', COMPARTMENT_ID,
        '--lifecycle-state', 'ACTIVE'
    ])

    if not clusters or not clusters.get('data'):
        print("No active clusters found")
        return None

    # Find the Enhanced cluster
    enhanced_cluster = None
    for cluster in clusters['data']:
        if cluster.get('type') == 'ENHANCED_CLUSTER':
            enhanced_cluster = cluster
            break

    if not enhanced_cluster:
        print("No Enhanced cluster found")
        return None

    cluster_id = enhanced_cluster['id']
    print(f"Found Enhanced cluster: {enhanced_cluster['name']} ({cluster_id})")

    # Get full cluster details
    cluster_details = run_oci_command([
        'oci', 'ce', 'cluster', 'get',
        '--cluster-id', cluster_id
    ])

    if not cluster_details:
        return None

    cluster_data = cluster_details['data']

    # Get node pool details
    node_pools = run_oci_command([
        'oci', 'ce', 'node-pool', 'list',
        '--compartment-id', COMPARTMENT_ID,
        '--cluster-id', cluster_id
    ])

    node_pool_data = None
    if node_pools and node_pools.get('data'):
        node_pool_id = node_pools['data'][0]['id']
        node_pool_details = run_oci_command([
            'oci', 'ce', 'node-pool', 'get',
            '--node-pool-id', node_pool_id
        ])
        if node_pool_details:
            node_pool_data = node_pool_details['data']

    return {
        'cluster': cluster_data,
        'node_pool': node_pool_data
    }


def print_existing_config(config):
    """Print existing cluster configuration"""
    if not config:
        print("Could not fetch existing configuration")
        return

    cluster = config['cluster']
    node_pool = config.get('node_pool')

    print()
    print("=" * 70)
    print("EXISTING ENHANCED CLUSTER CONFIGURATION")
    print("=" * 70)
    print()
    print(f"Cluster Name:      {cluster['name']}")
    print(f"Cluster ID:        {cluster['id']}")
    print(f"Kubernetes Ver:    {cluster['kubernetes-version']}")
    print(f"VCN ID:            {cluster['vcn-id']}")
    print()

    # Endpoint config
    endpoint = cluster.get('endpoint-config', {})
    print(f"Endpoint Subnet:   {endpoint.get('subnet-id', 'N/A')}")
    print(f"Is Public:         {endpoint.get('is-public-ip-enabled', False)}")
    print()

    # Cluster options
    options = cluster.get('options', {})
    print(f"Service LB Subnet: {options.get('service-lb-subnet-ids', ['N/A'])[0] if options.get('service-lb-subnet-ids') else 'N/A'}")
    print()

    if node_pool:
        print("NODE POOL:")
        print(f"  Name:            {node_pool['name']}")
        print(f"  Node Pool ID:    {node_pool['id']}")
        print(f"  Shape:           {node_pool['node-shape']}")

        shape_config = node_pool.get('node-shape-config', {})
        print(f"  OCPUs:           {shape_config.get('ocpus', 'N/A')}")
        print(f"  Memory GB:       {shape_config.get('memory-in-gbs', 'N/A')}")

        node_config = node_pool.get('node-config-details', {})
        print(f"  Current Size:    {node_config.get('size', 0)}")

        # Subnet
        placement = node_config.get('placement-configs', [{}])[0]
        print(f"  Node Subnet:     {placement.get('subnet-id', 'N/A')}")

        # Node source
        source = node_pool.get('node-source', {})
        print(f"  Image ID:        {source.get('image-id', 'N/A')}")

    print()
    print("=" * 70)
    print()
    print("Copy these values to update the configuration section above,")
    print("then run: python oci/tools/create_basic_cluster.py --create-cluster")
    print()


def check_status():
    """Check status of both clusters"""
    print("Checking cluster status...")
    print()

    clusters = run_oci_command([
        'oci', 'ce', 'cluster', 'list',
        '--compartment-id', COMPARTMENT_ID,
        '--all'
    ])

    if not clusters or not clusters.get('data'):
        print("No clusters found")
        return

    print(f"{'Name':<30} {'Type':<20} {'State':<15} {'K8s Version'}")
    print("-" * 80)

    for cluster in clusters['data']:
        name = cluster.get('name', 'N/A')[:30]
        ctype = cluster.get('type', 'N/A')
        state = cluster.get('lifecycle-state', 'N/A')
        version = cluster.get('kubernetes-version', 'N/A')
        print(f"{name:<30} {ctype:<20} {state:<15} {version}")

    print()

    # List node pools
    print("Node Pools:")
    print("-" * 80)

    for cluster in clusters['data']:
        if cluster.get('lifecycle-state') != 'ACTIVE':
            continue

        node_pools = run_oci_command([
            'oci', 'ce', 'node-pool', 'list',
            '--compartment-id', COMPARTMENT_ID,
            '--cluster-id', cluster['id']
        ])

        if node_pools and node_pools.get('data'):
            for np in node_pools['data']:
                np_details = run_oci_command([
                    'oci', 'ce', 'node-pool', 'get',
                    '--node-pool-id', np['id']
                ])
                if np_details:
                    np_data = np_details['data']
                    size = np_data.get('node-config-details', {}).get('size', 0)
                    print(f"  {cluster['name']}: {np['name']} (size={size}, id={np['id'][:50]}...)")


def create_basic_cluster():
    """Create new Basic cluster"""

    # First fetch existing config if not set
    global VCN_ID, ENDPOINT_SUBNET_ID, NODE_SUBNET_ID, SERVICE_LB_SUBNET_ID

    if not VCN_ID:
        print("Fetching configuration from existing cluster...")
        config = get_existing_cluster_config()
        if not config:
            print("ERROR: Could not fetch existing cluster config")
            print("Please set VCN_ID, ENDPOINT_SUBNET_ID, etc. manually in this script")
            return

        cluster = config['cluster']
        node_pool = config.get('node_pool')

        VCN_ID = cluster['vcn-id']
        ENDPOINT_SUBNET_ID = cluster.get('endpoint-config', {}).get('subnet-id')

        options = cluster.get('options', {})
        if options.get('service-lb-subnet-ids'):
            SERVICE_LB_SUBNET_ID = options['service-lb-subnet-ids'][0]

        if node_pool:
            placement = node_pool.get('node-config-details', {}).get('placement-configs', [{}])[0]
            NODE_SUBNET_ID = placement.get('subnet-id')

    print()
    print("=" * 70)
    print("CREATING BASIC CLUSTER (Free Control Plane)")
    print("=" * 70)
    print()
    print(f"Cluster Name:      {NEW_CLUSTER_NAME}")
    print(f"Kubernetes Ver:    {KUBERNETES_VERSION}")
    print(f"VCN ID:            {VCN_ID}")
    print(f"Endpoint Subnet:   {ENDPOINT_SUBNET_ID}")
    print()

    # Build cluster create command
    cmd = [
        'oci', 'ce', 'cluster', 'create',
        '--compartment-id', COMPARTMENT_ID,
        '--name', NEW_CLUSTER_NAME,
        '--vcn-id', VCN_ID,
        '--kubernetes-version', KUBERNETES_VERSION,
        '--type', 'BASIC_CLUSTER',
        '--endpoint-subnet-id', ENDPOINT_SUBNET_ID,
        '--endpoint-public-ip-enabled', 'true'
    ]

    if SERVICE_LB_SUBNET_ID:
        cmd.extend(['--service-lb-subnet-ids', f'["{SERVICE_LB_SUBNET_ID}"]'])

    print("Creating cluster...")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = run_oci_command(cmd)

    if result:
        print("✅ Cluster creation initiated!")
        print()
        print("⏳ Cluster will take ~10 minutes to become ACTIVE")
        print()
        print("Check status with:")
        print("  python oci/tools/create_basic_cluster.py --status")
        print()
        print("Once ACTIVE, create node pool with:")
        print("  python oci/tools/create_basic_cluster.py --create-nodepool")
    else:
        print("❌ Cluster creation failed")


def create_node_pool():
    """Create node pool for the Basic cluster"""

    # Find the Basic cluster
    clusters = run_oci_command([
        'oci', 'ce', 'cluster', 'list',
        '--compartment-id', COMPARTMENT_ID,
        '--name', NEW_CLUSTER_NAME,
        '--lifecycle-state', 'ACTIVE'
    ])

    if not clusters or not clusters.get('data'):
        print(f"ERROR: Basic cluster '{NEW_CLUSTER_NAME}' not found or not ACTIVE")
        print("Create it first with: python oci/tools/create_basic_cluster.py --create-cluster")
        return

    basic_cluster = clusters['data'][0]
    cluster_id = basic_cluster['id']

    print()
    print("=" * 70)
    print("CREATING NODE POOL")
    print("=" * 70)
    print()
    print(f"Cluster:           {NEW_CLUSTER_NAME}")
    print(f"Cluster ID:        {cluster_id}")
    print()

    # Get node subnet from existing cluster if not set
    global NODE_SUBNET_ID, NODE_IMAGE_ID

    if not NODE_SUBNET_ID:
        config = get_existing_cluster_config()
        if config and config.get('node_pool'):
            node_pool = config['node_pool']
            placement = node_pool.get('node-config-details', {}).get('placement-configs', [{}])[0]
            NODE_SUBNET_ID = placement.get('subnet-id')

            # Get image ID
            source = node_pool.get('node-source', {})
            NODE_IMAGE_ID = source.get('image-id')

    if not NODE_SUBNET_ID:
        print("ERROR: NODE_SUBNET_ID not set. Please configure it in this script.")
        return

    # Get availability domain
    ad_result = run_oci_command([
        'oci', 'iam', 'availability-domain', 'list',
        '--compartment-id', COMPARTMENT_ID
    ])

    if not ad_result or not ad_result.get('data'):
        print("ERROR: Could not fetch availability domains")
        return

    ad_name = ad_result['data'][0]['name']
    print(f"Availability Domain: {ad_name}")
    print(f"Node Subnet:         {NODE_SUBNET_ID}")
    print(f"Node Shape:          {NODE_SHAPE}")
    print(f"OCPUs:               {NODE_OCPUS}")
    print(f"Memory GB:           {NODE_MEMORY_GB}")
    print(f"Initial Size:        {INITIAL_NODE_COUNT}")
    print()

    # Build node config
    placement_config = {
        "availabilityDomain": ad_name,
        "subnetId": NODE_SUBNET_ID
    }

    node_config = {
        "size": INITIAL_NODE_COUNT,
        "placementConfigs": [placement_config]
    }

    shape_config = {
        "ocpus": NODE_OCPUS,
        "memoryInGBs": NODE_MEMORY_GB
    }

    # Node source - use OKE image
    node_source = {
        "sourceType": "IMAGE",
        "imageId": NODE_IMAGE_ID
    } if NODE_IMAGE_ID else None

    cmd = [
        'oci', 'ce', 'node-pool', 'create',
        '--compartment-id', COMPARTMENT_ID,
        '--cluster-id', cluster_id,
        '--name', 'backtest-workers',
        '--node-shape', NODE_SHAPE,
        '--kubernetes-version', KUBERNETES_VERSION,
        '--node-config-details', json.dumps(node_config),
        '--node-shape-config', json.dumps(shape_config)
    ]

    if node_source:
        cmd.extend(['--node-source-details', json.dumps(node_source)])

    print("Creating node pool...")
    result = run_oci_command(cmd)

    if result:
        # Get the new node pool ID
        node_pool_id = result.get('data', {}).get('id', 'N/A')
        print()
        print("✅ Node pool creation initiated!")
        print()
        print(f"Node Pool ID: {node_pool_id}")
        print()
        print("⏳ Node pool will take ~5 minutes to be ready")
        print()
        print("IMPORTANT: Update this ID in your submit scripts:")
        print(f"  oci/tools/submit_oci_backtest.py")
        print(f"  oci/tools/cleanup_and_download_backtest.py")
        print()
        print("Look for: self.node_pool_id = \"...\"")
        print(f"Replace with: self.node_pool_id = \"{node_pool_id}\"")
        print()
        print("Then create secrets with:")
        print("  python oci/tools/create_basic_cluster.py --create-secrets")
    else:
        print("❌ Node pool creation failed")


def create_secrets():
    """Create Kubernetes secrets for OCI access"""

    # First, get kubeconfig for the new cluster
    clusters = run_oci_command([
        'oci', 'ce', 'cluster', 'list',
        '--compartment-id', COMPARTMENT_ID,
        '--name', NEW_CLUSTER_NAME,
        '--lifecycle-state', 'ACTIVE'
    ])

    if not clusters or not clusters.get('data'):
        print(f"ERROR: Basic cluster '{NEW_CLUSTER_NAME}' not found or not ACTIVE")
        return

    cluster_id = clusters['data'][0]['id']

    print()
    print("=" * 70)
    print("CREATING KUBERNETES SECRETS")
    print("=" * 70)
    print()

    # Create kubeconfig for new cluster
    print("Step 1: Creating kubeconfig for new cluster...")
    kubeconfig_path = Path.home() / '.kube' / 'config-basic'

    subprocess.run([
        'oci', 'ce', 'cluster', 'create-kubeconfig',
        '--cluster-id', cluster_id,
        '--file', str(kubeconfig_path),
        '--region', 'ap-mumbai-1',
        '--token-version', '2.0.0',
        '--kube-endpoint', 'PUBLIC_ENDPOINT'
    ], check=True)

    print(f"✅ Kubeconfig created: {kubeconfig_path}")
    print()

    # Instructions for creating secrets
    print("Step 2: Create secrets using the new kubeconfig")
    print()
    print("Run these commands in OCI Cloud Shell (or locally with kubectl):")
    print()
    print(f"# Set KUBECONFIG to use new cluster")
    print(f"export KUBECONFIG={kubeconfig_path}")
    print()
    print("# Verify you're connected to the new cluster")
    print("kubectl cluster-info")
    print()
    print("# Create OCI config secret (same as existing)")
    print("kubectl create secret generic oci-config \\")
    print("  --from-file=config=$HOME/.oci/config \\")
    print("  --from-file=key=$HOME/.oci/oci_api_key.pem")
    print()
    print("# Create OCIR pull secret (same as existing)")
    print("kubectl create secret docker-registry ocir-secret \\")
    print("  --docker-server=bom.ocir.io \\")
    print("  --docker-username='<tenancy-namespace>/oracleidentitycloudservice/<your-email>' \\")
    print("  --docker-password='<auth-token>'")
    print()
    print("=" * 70)
    print()
    print("After creating secrets, test with:")
    print(f"  KUBECONFIG={kubeconfig_path} kubectl get secrets")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Create OKE Basic Cluster (Free Control Plane)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check existing cluster config
  python oci/tools/create_basic_cluster.py --show-config

  # Check status of all clusters
  python oci/tools/create_basic_cluster.py --status

  # Step 1: Create Basic cluster
  python oci/tools/create_basic_cluster.py --create-cluster

  # Step 2: Create node pool (after cluster is ACTIVE)
  python oci/tools/create_basic_cluster.py --create-nodepool

  # Step 3: Create secrets
  python oci/tools/create_basic_cluster.py --create-secrets
"""
    )

    parser.add_argument('--show-config', action='store_true',
                        help='Show existing cluster configuration')
    parser.add_argument('--status', action='store_true',
                        help='Check status of all clusters')
    parser.add_argument('--create-cluster', action='store_true',
                        help='Create new Basic cluster')
    parser.add_argument('--create-nodepool', action='store_true',
                        help='Create node pool for Basic cluster')
    parser.add_argument('--create-secrets', action='store_true',
                        help='Instructions for creating Kubernetes secrets')

    args = parser.parse_args()

    if args.show_config:
        config = get_existing_cluster_config()
        print_existing_config(config)
    elif args.status:
        check_status()
    elif args.create_cluster:
        create_basic_cluster()
    elif args.create_nodepool:
        create_node_pool()
    elif args.create_secrets:
        create_secrets()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
