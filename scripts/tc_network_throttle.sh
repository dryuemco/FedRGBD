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
# Interface: the wired FL link. Default enP8p1s0 (Jetson Orin Nano onboard
# Ethernet); override with the second argument or FEDRGBD_IFACE. The script
# never auto-detects: it aborts if the interface does not exist or is a WiFi
# interface, so it cannot silently throttle the wrong link. (v1 of the paper
# ran over WiFi; the revision testbed is wired Gigabit Ethernet.)
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

IFACE="${IFACE:-${FEDRGBD_IFACE:-enP8p1s0}}"

if [ ! -e "/sys/class/net/$IFACE" ]; then
    echo "ERROR: network interface '$IFACE' not found on $(hostname)." >&2
    echo "Available interfaces: $(ls /sys/class/net | tr '\n' ' ')" >&2
    echo "Pass the wired FL interface explicitly: sudo $0 $CONDITION <interface>" >&2
    exit 1
fi
if [ -d "/sys/class/net/$IFACE/wireless" ] || [ -e "/sys/class/net/$IFACE/phy80211" ]; then
    echo "ERROR: '$IFACE' is a wireless interface; the FL link is wired Gigabit Ethernet." >&2
    exit 1
fi
echo "Interface: $IFACE"

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
