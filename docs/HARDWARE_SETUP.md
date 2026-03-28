# FedRGBD — Hardware Setup Guide

## Network Topology

```
Node A (FL Server + Client)       Node B (FL Client)            Node C (FL Client)
┌──────────────────────┐        ┌──────────────────────┐      ┌──────────────────────┐
│ Jetson Orin Nano Super│        │ Jetson Orin Nano Super│      │ Jetson Orin Nano Super│
│ IP: 192.168.1.4      │        │ IP: 192.168.1.5      │      │ IP: 192.168.1.3      │
│ Hostname: fedrgbd-a  │        │ Hostname: fedrgbd-b  │      │ Hostname: fedrgbd-c  │
│                       │        │                       │      │                       │
│ USB3: D435if camera  │        │ USB3: D435i camera   │      │ USB3: ZED 2i camera  │
│ WiFi: 802.11ac       │───WiFi──│ WiFi: 802.11ac       │──────│ WiFi: 802.11ac       │
│ Power: 15W mode      │        │ Power: 15W mode      │      │ Power: 15W mode      │
└──────────────────────┘        └──────────────────────┘      └──────────────────────┘
```

## Node Details

| Property | Node A | Node B | Node C |
|----------|--------|--------|--------|
| Hostname | fedrgbd-a | fedrgbd-b | fedrgbd-c |
| IP | 192.168.1.4 | 192.168.1.5 | 192.168.1.3 |
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
ping -c 3 192.168.1.5   # Node B
ping -c 3 192.168.1.3   # Node C

# From Node B
ping -c 3 192.168.1.4   # Node A

# From Node C
ping -c 3 192.168.1.4   # Node A
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
