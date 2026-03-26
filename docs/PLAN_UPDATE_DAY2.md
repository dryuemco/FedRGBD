# FedRGBD — Plan Update: Camera Configuration

**Date:** Day 2 (March 2026)
**Reason:** D455 camera unavailable. Actual cameras: D435i + D435if

---

## Current Hardware

| Node | Camera | Key Difference |
|------|--------|---------------|
| Node A | Intel RealSense D435if | IR cut filter on RGB sensor |
| Node B | Intel RealSense D435i | No IR cut filter on RGB |

Both cameras share: 50mm baseline, 0.3-3m range, 1280×720 depth, 1920×1080 RGB.

## Impact on Research Gaps

### Gap 2: Sensor Heterogeneity (REVISED)

**Original plan:** D435i (50mm) vs D455 (95mm) — large structural differences in depth maps.

**Revised plan:** D435i vs D435if — heterogeneity sources:
1. **IR filter difference:** D435if's RGB sensor filters IR light, producing different color response especially under mixed lighting (indoor fluorescent, sunlight with IR component). This creates measurable feature distribution shift in the RGB channel.
2. **Manufacturing variance:** Even same-model cameras have slightly different calibration, lens alignment, and sensor noise profiles. Each camera's intrinsics (fx, fy, ppx, ppy, distortion coefficients) differ.
3. **Serial-specific noise:** Different noise floors in depth and IR channels due to manufacturing tolerances.

**Argument strength:** Moderate. This represents a *realistic* deployment scenario where identical camera models still produce non-IID data due to manufacturing variance — a practical concern that has not been studied in FL literature.

### Strengthened Framing

Instead of "different camera models produce different data," the argument becomes:
> "Even nominally identical sensors produce non-IID data in federated settings due to manufacturing variance, calibration differences, and hardware configuration (IR filter). This is a more realistic scenario than artificial non-IID partitioning, as real-world deployments typically use the same camera model across nodes."

## Experiment 3: Cross-Sensor Generalization (REVISED)

| Config | Train | Test | What it measures |
|--------|-------|------|-----------------|
| A→B | D435if data | D435i data | IR filter impact on generalization |
| B→A | D435i data | D435if data | Reverse direction |
| FL(A+B) | Both via FL | Both | Does FL bridge the gap? |

**New analysis added:**
- Quantify RGB channel difference (histogram comparison, SSIM) between D435i and D435if under same scene
- Measure depth noise profile difference using flat wall capture
- Report camera intrinsics difference as non-IID evidence

## Future Enhancement: ZED 2i Integration

**Status:** ZED 2i camera may become available during or after the study.

**If available, adds Experiment 6:**
- D435i (active IR stereo) vs ZED 2i (passive stereo + neural depth)
- Completely different depth technology = strong heterogeneity argument
- Different SDK integration (librealsense vs ZED SDK)
- 50mm vs 120mm baseline
- 0.3-3m vs 0.3-20m range

**Impact on paper:** Would significantly strengthen Gap 2 contribution. Can be added as extension experiment without changing the core methodology.

## Updated Manuscript Framing

### Contribution 2 (revised):
"Sensor heterogeneity as natural non-IID: We quantify the impact of real-world sensor variance — including manufacturing differences, hardware configuration (IR filter), and calibration offsets — on FL convergence. Unlike artificial non-IID partitioning, this represents the actual data heterogeneity encountered when deploying identical camera models across distributed edge nodes."

---

*This update does not change the core methodology, FL protocol, or other experiments. Only Experiment 3 framing and the sensor heterogeneity argument are adjusted.*
