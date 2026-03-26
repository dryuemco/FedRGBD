# FedRGBD — Hardware Setup Guide

## Network Topology

```
Node A (FL Server + Client)          Node B (FL Client)
┌─────────────────────┐              ┌─────────────────────┐
│ Jetson Orin Nano 8GB │              │ Jetson Orin Nano 8GB │
│ IP: 192.168.1.8     │              │ IP: 192.168.1.5     │
│                      │              │                      │
│ USB3: D435if camera  │              │ USB3: D435i camera   │
│ ETH: Gigabit LAN     │──── Switch ──│ ETH: Gigabit LAN     │
│ Power: 15W mode      │              │ Power: 15W mode      │
└─────────────────────┘              └─────────────────────┘
```

## Static IP Configuration

On Node A:
```bash
sudo nmcli connection modify 'Wired connection 1' \
  ipv4.addresses 192.168.1.8/24 \
  ipv4.method manual
sudo nmcli connection up 'Wired connection 1'
```

On Node B:
```bash
sudo nmcli connection modify 'Wired connection 1' \
  ipv4.addresses 192.168.1.5/24 \
  ipv4.method manual
sudo nmcli connection up 'Wired connection 1'
```

Verify: `ping 192.168.1.5` from Node A.

## Power Mode

Set both Jetsons to 15W mode (default for Orin Nano 8GB):
```bash
sudo nvpmodel -m 0    # 15W mode
sudo jetson_clocks     # Max clocks within power budget
```

Check current mode:
```bash
sudo nvpmodel -q
```

## USB3 Camera Connection

- D435i → Node A USB 3.2 Gen2 port (the blue one)
- D455 → Node B USB 3.2 Gen2 port

**Important**: Use short, high-quality USB-C cables. Long or low-quality cables 
cause bandwidth issues with depth streaming.

Verify camera detection:
```bash
lsusb | grep Intel
rs-enumerate-devices | head -20
```

## Monitoring During Experiments

Terminal 1 — jtop (visual monitoring):
```bash
jtop
```

Terminal 2 — tegrastats logging:
```bash
tegrastats --interval 100 --logfile tegrastats_exp1_run1.txt
```
