# FedRGBD — Hardware Setup Guide

## Network Topology

```
Node A (FL Server + Client)       Node B (FL Client)            Node C (FL Client)
┌──────────────────────┐        ┌──────────────────────┐      ┌──────────────────────┐
│ Jetson Orin Nano Super│        │ Jetson Orin Nano Super│      │ Jetson Orin Nano Super│
│ IP: 192.168.1.10     │        │ IP: 192.168.1.7      │      │ IP: 192.168.1.6      │
│ Hostname: fedrgbd-a  │        │ Hostname: fedrgbd-b  │      │ Hostname: fedrgbd-c  │
│                       │        │                       │      │                       │
│ USB3: D435if camera  │        │ USB3: D435i camera   │      │ USB3: ZED 2i camera  │
│ Eth: 1 GbE (wired)   │        │ Eth: 1 GbE (wired)   │      │ Eth: 1 GbE (wired)   │
│ Power: 15W mode      │        │ Power: 15W mode      │      │ Power: 15W mode      │
└──────────┬───────────┘        └──────────┬───────────┘      └──────────┬───────────┘
           │                               │                             │
           └───────────────────┐           │           ┌─────────────────┘
                          ┌────┴───────────┴───────────┴────┐
                          │     Gigabit Ethernet switch     │
                          └─────────────────────────────────┘
```

The revision experiments run on wired Gigabit Ethernet with static addresses (link 1000 Mb/s;
mean RTT to the gateway 0.68 ms over 5 pings, 0% loss). The testbed was reassembled between
submission and revision: **the v1 experiments (`results/3node_*` etc.) ran over WiFi
(IEEE 802.11ac)**, so v1 and revision wall-clock times are not comparable.

The FL link is the onboard Ethernet interface `enP8p1s0`; WiFi is down on all three nodes.

### Raw link measurements (source of the numbers in the paper, Section III-A)

The paper states "links negotiated 1000 Mb/s" and "mean round-trip time to the gateway
0.68 ms over five pings, no packet loss". These come from the commands below.

> **TODO (author):** paste the verbatim output here, and say which node it was run on and
> when. Do not retype or summarise the numbers.

```bash
sudo ethtool enP8p1s0 | grep -E 'Speed|Duplex|Link detected'
ping -c 5 <gateway-ip>
```

```text
<paste ethtool output here>
```

```text
<paste the ping summary here (the "packet loss" and "rtt min/avg/max/mdev" lines)>
```

## Node Details

| Property | Node A | Node B | Node C |
|----------|--------|--------|--------|
| Hostname | fedrgbd-a | fedrgbd-b | fedrgbd-c |
| IP | 192.168.1.10 | 192.168.1.7 | 192.168.1.6 |
| Network | Wired GbE (v1: WiFi) | Wired GbE (v1: WiFi) | Wired GbE (v1: WiFi) |
| Camera | Intel RealSense D435if | Intel RealSense D435i | Stereolabs ZED 2i |
| Camera S/N | 239722070442 | 405622076256 | 32608934 |
| Camera FW | 5.13.0.55 | 5.17.0.10 | 1523 |
| Camera SDK | librealsense 2.55.1 | librealsense 2.55.1 | ZED SDK 5.2.3 |
| FL Role | Server + Client | Client | Client |
| JetPack | 6.2 | 6.2 | 6.2 |
| CUDA | 12.6 | 12.6 | 12.6 |

## Verify Network Connectivity

```bash
# From Node A
ping -c 3 192.168.1.7   # Node B
ping -c 3 192.168.1.6   # Node C

# From Node B
ping -c 3 192.168.1.10   # Node A

# From Node C
ping -c 3 192.168.1.10   # Node A
```

## Power Mode

Set all Jetsons to 15W mode:
```bash
sudo nvpmodel -m 0    # 15W mode
sudo jetson_clocks     # Max clocks within power budget
sudo nvpmodel -q       # Verify
```

## Camera Connection

- D435if → Node A USB 3.2 Gen2 port
- D435i → Node B USB 3.2 Gen2 port
- ZED 2i → Node C USB 3.2 Gen2 port

**Important**: Use short, high-quality USB-C cables.

Verify camera detection:
```bash
# Node A / Node B (RealSense)
rs-enumerate-devices | head -20

# Node C (ZED)
python3 -c "import pyzed.sl as sl; cam = sl.Camera(); print(sl.Camera.get_device_list())"
```

## Virtual Environment

All nodes use the same venv:
```bash
source ~/fedrgbd_venv/bin/activate
```

For RealSense nodes (A, B), pyrealsense2 needs PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:~/librealsense/build/Release:~/librealsense/build/wrappers/python
```

## Monitoring During Experiments

```bash
# Terminal 1 — jtop (visual monitoring)
jtop

# Terminal 2 — tegrastats logging
tegrastats --interval 100 --logfile tegrastats_exp_round.txt
```

## Known Memory Issues

- `pin_memory=True` causes OOM → use `pin_memory=False`
- Node A needs `batch_size=8` when running server + client (shared GPU)
- Node C: kill ZED background processes before FL training
- `numpy` must be 1.26.4 — numpy 2.x breaks PyTorch
- NvMapMemAllocInternalTagged errors → clear GPU cache, reduce batch size
