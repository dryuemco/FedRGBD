# Auxiliary tilt measurement (NOT e5 data): D435i accelerometer, ~2 s at rest, uncalibrated.
# RealSense IMU frame (librealsense docs, same axes as the depth frame): +X right, +Y down,
# +Z forward out of the lens.  At rest the accelerometer reports the specific force, i.e.
# +g pointing UP, so a level camera reads about (0, -9.81, 0).
#   pitch_down_deg = atan2(-az, -ay): positive = optical axis tilted DOWN (nose down).
#     Nose down by t: the up vector has component -sin(t) on +Z, so az = -g sin(t) < 0.
#   roll_deg = atan2(-ax, -ay): positive = camera rotated clockwise as seen from behind
#     (right side down): up has component -sin(r) on +X, so ax = -g sin(r) < 0.
import json, math, time
import pyrealsense2 as rs

SERIAL = "405622076256"
import threading
dev = [d for d in rs.context().query_devices() if d.get_info(rs.camera_info.serial_number) == SERIAL][0]
motion = [s for s in dev.query_sensors() if s.get_info(rs.camera_info.name) == "Motion Module"][0]
prof = [p for p in motion.get_stream_profiles()
        if p.stream_type() == rs.stream.accel and p.fps() == 100][0]
raw, lock = [], threading.Lock()
def cb(f):
    d = f.as_motion_frame().get_motion_data()
    with lock:
        raw.append((f.get_timestamp(), d.x, d.y, d.z))
motion.open(prof)
try:
    motion.start(cb)
    time.sleep(4.0)
    motion.stop()
finally:
    motion.close()                                # release the device for the capture script
fps = prof.fps()
if not raw:
    raise SystemExit("no accel frame in 4 s")
t0 = raw[0][0]
xs = [(x, y, z) for t, x, y, z in raw if 500 <= t - t0 < 2500]   # 2 s after a 0.5 s settle
n = len(xs)
ax, ay, az = (sum(v[i] for v in xs) / n for i in range(3))
sd = [math.sqrt(sum((v[i] - m) ** 2 for v in xs) / (n - 1)) for i, m in enumerate((ax, ay, az))]
print(json.dumps({
    "auxiliary_not_e5_data": True, "tilt_source": "D435i IMU accelerometer, uncalibrated",
    "serial": SERIAL, "accel_stream_fps": fps, "n_samples": n, "window_s": 2.0,
    "accel_mean_m_s2": [round(ax, 4), round(ay, 4), round(az, 4)],
    "accel_sd_m_s2": [round(s, 4) for s in sd],
    "norm_m_s2": round(math.sqrt(ax * ax + ay * ay + az * az), 4),
    "pitch_down_deg": round(math.degrees(math.atan2(-az, -ay)), 2),
    "roll_deg": round(math.degrees(math.atan2(-ax, -ay)), 2),
    "device_released": True}))
