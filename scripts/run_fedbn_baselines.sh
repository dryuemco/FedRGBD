#!/bin/bash
# =============================================================================
# FedRGBD — FedBN + Baseline Experiment Runner
# =============================================================================
# Run this from: ~/FedRGBD on Node A (server + client node)
# Prerequisites:
#   - source ~/fedrgbd_venv/bin/activate
#   - All 4 new files deployed (fedbn_strategy.py, server.py, client.py, 
#     train_centralized.py, train_local.py)
#   - pkill -f python3 on all nodes before starting
#
# IMPORTANT: FedBN experiments require manual coordination (3 terminal windows,
# one per node). Use the commands below. Centralized and Local experiments
# can run on a single node.
# =============================================================================

set -e
cd ~/FedRGBD
source ~/fedrgbd_venv/bin/activate

echo "=============================================="
echo "  FedRGBD — FedBN + Baseline Experiments"
echo "  $(date)"
echo "=============================================="

# =============================================================================
# PART 1: FedBN EXPERIMENTS (FL — requires 3 nodes, ~9 hours total)
# =============================================================================
# These need manual multi-terminal coordination, same as FedAvg/FedProx.
# Run each seed × distribution combo separately.

echo ""
echo "================================================="
echo "  PART 1: FedBN Experiments (manual multi-node)"
echo "================================================="
echo ""
echo "For each experiment below, open 3 terminals and run:"
echo ""

# --- FedBN IID ---
for SEED in 42 123 456; do
    DIR="results/3n_iid_fedbn_seed${SEED}"
    echo "--- FedBN IID seed=${SEED} ---"
    echo "  Node A (server):  python3 src/fl/server.py --strategy fedbn --rounds 3 --seed ${SEED} --output_dir ${DIR}"
    echo "  Node A (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_a --batch_size 8 --seed ${SEED}"
    echo "  Node B (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_b --batch_size 8 --seed ${SEED}"
    echo "  Node C (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/iid/node_c --batch_size 8 --seed ${SEED}"
    echo "  Expected time: ~90 min (similar to FedAvg)"
    echo ""
done

# --- FedBN Non-IID ---
for SEED in 42 123 456; do
    DIR="results/3n_noniid_fedbn_seed${SEED}"
    echo "--- FedBN Non-IID seed=${SEED} ---"
    echo "  Node A (server):  python3 src/fl/server.py --strategy fedbn --rounds 3 --seed ${SEED} --output_dir ${DIR}"
    echo "  Node A (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/non_iid_label/node_a --batch_size 8 --seed ${SEED}"
    echo "  Node B (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/non_iid_label/node_b --batch_size 8 --seed ${SEED}"
    echo "  Node C (client):  python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir data/processed/non_iid_label/node_c --batch_size 8 --seed ${SEED}"
    echo "  Expected time: ~90 min"
    echo ""
done

echo "FedBN total: 6 runs × ~90 min = ~9 hours"
echo ""

# =============================================================================
# PART 2: CENTRALIZED BASELINE (single node, ~4 hours total)
# =============================================================================
# Can run on any single node. Node A recommended.
# Run these SEQUENTIALLY — one at a time.

echo "================================================="
echo "  PART 2: Centralized Baseline (single node)"
echo "================================================="
echo ""

# Uncomment to auto-run, or copy commands manually:

# --- Centralized IID ---
for SEED in 42 123 456; do
    echo "--- Centralized IID seed=${SEED} ---"
    echo "  python3 scripts/train_centralized.py \\"
    echo "    --data_dirs data/processed/iid/node_a data/processed/iid/node_b data/processed/iid/node_c \\"
    echo "    --epochs 15 --batch_size 8 --seed ${SEED} \\"
    echo "    --output_dir results/centralized_iid_seed${SEED}"
    echo "  Expected time: ~40 min"
    echo ""
done

# --- Centralized Non-IID ---
for SEED in 42 123 456; do
    echo "--- Centralized Non-IID seed=${SEED} ---"
    echo "  python3 scripts/train_centralized.py \\"
    echo "    --data_dirs data/processed/non_iid_label/node_a data/processed/non_iid_label/node_b data/processed/non_iid_label/node_c \\"
    echo "    --epochs 15 --batch_size 8 --seed ${SEED} \\"
    echo "    --output_dir results/centralized_noniid_seed${SEED}"
    echo "  Expected time: ~40 min"
    echo ""
done

echo "Centralized total: 6 runs × ~40 min = ~4 hours"
echo ""

# =============================================================================
# PART 3: LOCAL-ONLY BASELINE (single node batch mode, ~4.5 hours total)
# =============================================================================
# Batch mode trains all 3 nodes sequentially on one device.
# --cross_eval flag tests each model on other nodes' test sets.

echo "================================================="
echo "  PART 3: Local-Only Baseline (batch mode)"
echo "================================================="
echo ""

# --- Local IID ---
for SEED in 42 123 456; do
    echo "--- Local-Only IID seed=${SEED} ---"
    echo "  python3 scripts/train_local.py --batch --cross_eval \\"
    echo "    --data_dirs data/processed/iid/node_a data/processed/iid/node_b data/processed/iid/node_c \\"
    echo "    --epochs 15 --batch_size 8 --seed ${SEED} \\"
    echo "    --output_dir results/local_iid_seed${SEED}"
    echo "  Expected time: ~45 min (3 nodes × ~15 min each)"
    echo ""
done

# --- Local Non-IID ---
for SEED in 42 123 456; do
    echo "--- Local-Only Non-IID seed=${SEED} ---"
    echo "  python3 scripts/train_local.py --batch --cross_eval \\"
    echo "    --data_dirs data/processed/non_iid_label/node_a data/processed/non_iid_label/node_b data/processed/non_iid_label/node_c \\"
    echo "    --epochs 15 --batch_size 8 --seed ${SEED} \\"
    echo "    --output_dir results/local_noniid_seed${SEED}"
    echo "  Expected time: ~45 min"
    echo ""
done

echo "Local-Only total: 6 runs × ~45 min = ~4.5 hours"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "================================================="
echo "  TOTAL EXPERIMENT PLAN"
echo "================================================="
echo ""
echo "  FedBN (FL, 3 nodes):        6 runs × ~90 min  = ~9 hours"
echo "  Centralized (single node):  6 runs × ~40 min  = ~4 hours"
echo "  Local-Only (single node):   6 runs × ~45 min  = ~4.5 hours"
echo "  ─────────────────────────────────────────────────────────"
echo "  Grand Total:                18 runs            ≈ 17.5 hours"
echo ""
echo "  Recommended order:"
echo "    1. Centralized (no network needed, ~4h)"
echo "    2. Local-Only (no network needed, ~4.5h)"  
echo "    3. FedBN (needs all 3 nodes, ~9h)"
echo ""
echo "  TIP: Run centralized + local overnight, FedBN next day."
echo "  TIP: pkill -f python3 on all nodes between experiments!"
echo "================================================="
