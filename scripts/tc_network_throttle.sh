#!/bin/bash
# =============================================================================
# FedRGBD — Network Bandwidth Throttling for Experiment 5
# =============================================================================
# Uses Linux Traffic Control (tc) to simulate bandwidth constraints.
# Run on ALL 3 Jetson nodes before starting FL experiments.
#
# Usage:
#   sudo ./tc_network_throttle.sh <condition> [interface]
#
# Conditions:
#   baseline    — Remove all limits
#   10mbps      — Limit to 10 Mbps
#   1mbps       — Limit to 1 Mbps
#   1mbps_lossy — 1 Mbps + 5% packet loss
#   status      — Show current tc rules
# =============================================================================

set -e

CONDITION="${1:-status}"
IFACE="${2:-}"

# Auto-detect network interface (WiFi first, then Ethernet)
if [ -z "$IFACE" ]; then
    # Try WiFi interfaces first (current setup uses WiFi)
    IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^wl|^wifi' | head -1)

    # Fallback to Ethernet
    if [ -z "$IFACE" ]; then
        IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^eth|^enp|^eno' | head -1)
    fi

    if [ -z "$IFACE" ]; then
        echo "ERROR: No network interface found."
        echo "Specify manually: sudo $0 <condition> <interface>"
        exit 1
    fi
    echo "Auto-detected interface: $IFACE"
fi

case "$CONDITION" in
    baseline)
        echo "Removing all traffic control rules on $IFACE..."
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        echo "Done. Unrestricted network."
        ;;

    10mbps)
        echo "Setting $IFACE to 10 Mbps..."
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        sudo tc qdisc add dev "$IFACE" root tbf rate 10mbit burst 32kbit latency 400ms
        echo "Done. Bandwidth limited to 10 Mbps."
        ;;

    1mbps)
        echo "Setting $IFACE to 1 Mbps..."
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        sudo tc qdisc add dev "$IFACE" root tbf rate 1mbit burst 32kbit latency 400ms
        echo "Done. Bandwidth limited to 1 Mbps."
        ;;

    1mbps_lossy)
        echo "Setting $IFACE to 1 Mbps + 5% packet loss..."
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        sudo tc qdisc add dev "$IFACE" root handle 1: netem loss 5%
        sudo tc qdisc add dev "$IFACE" parent 1:1 handle 10: tbf rate 1mbit burst 32kbit latency 400ms
        echo "Done. Bandwidth limited to 1 Mbps with 5% packet loss."
        ;;

    status)
        echo "Current tc rules on $IFACE:"
        sudo tc qdisc show dev "$IFACE"
        ;;

    *)
        echo "Usage: sudo $0 <condition> [interface]"
        echo ""
        echo "Conditions:"
        echo "  baseline    — Remove all limits"
        echo "  10mbps      — 10 Mbps limit"
        echo "  1mbps       — 1 Mbps limit"
        echo "  1mbps_lossy — 1 Mbps + 5% packet loss"
        echo "  status      — Show current rules"
        exit 1
        ;;
esac
