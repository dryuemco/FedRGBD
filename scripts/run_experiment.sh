#!/bin/bash
# FedRGBD — Experiment Runner
# Usage: bash run_experiment.sh <strategy> <distribution> <seeds...>
# Example: bash run_experiment.sh fedavg iid 42 123 456
# Example: bash run_experiment.sh fedprox_0.01 non_iid 42 123 456
#
# Run this on the SERVER node (Node A).
# Client nodes must be started separately (see printed instructions).

STRATEGY=${1:-fedavg}
DIST=${2:-iid}
shift 2
SEEDS=${@:-42}

# Data paths
if [ "$DIST" = "iid" ]; then
    DATA_A="data/processed/iid/node_a"
    DATA_B="data/processed/iid/node_b"
    DATA_C="data/processed/iid/node_c"
else
    DATA_A="data/processed/non_iid_label/node_a"
    DATA_B="data/processed/non_iid_label/node_b"
    DATA_C="data/processed/non_iid_label/node_c"
fi

echo "=============================================="
echo "  FedRGBD Experiment Runner"
echo "  Strategy: $STRATEGY"
echo "  Distribution: $DIST"
echo "  Seeds: $SEEDS"
echo "=============================================="

for SEED in $SEEDS; do
    OUTPUT_DIR="results/3node_${DIST}_${STRATEGY}_seed${SEED}"

    # Skip if already completed
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo "[SKIP] $OUTPUT_DIR already exists"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "  Running: $STRATEGY / $DIST / seed=$SEED"
    echo "  Output: $OUTPUT_DIR"
    echo "=============================================="
    echo ""
    echo ">>> START CLIENTS ON OTHER NODES:"
    echo ""
    echo "  Node A (this node):"
    echo "    python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir $DATA_A --batch_size 8 --seed $SEED"
    echo ""
    echo "  Node B (192.168.1.5):"
    echo "    python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir $DATA_B --batch_size 8 --seed $SEED"
    echo ""
    echo "  Node C (192.168.1.3):"
    echo "    python3 src/fl/client.py --server 192.168.1.4:8080 --data_dir $DATA_C --batch_size 8 --seed $SEED"
    echo ""
    echo ">>> Starting server in 10 seconds..."
    echo "   (Start clients on other nodes NOW)"
    sleep 10

    python3 src/fl/server.py \
        --strategy "$STRATEGY" \
        --rounds 3 \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED" \
        --min_clients 3

    echo ""
    echo "[DONE] Seed $SEED complete. Results: $OUTPUT_DIR/results.json"
    echo ""
    echo ">>> Waiting 30s before next seed (let clients restart)..."
    sleep 30
done

echo ""
echo "=============================================="
echo "  All seeds completed for $STRATEGY / $DIST"
echo "=============================================="
