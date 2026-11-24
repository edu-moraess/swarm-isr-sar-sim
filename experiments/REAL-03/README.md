# REAL-03 — Video perception experiment

Status: experimental offline validation checkpoint. This record documents the REAL-03.5 video run and the REAL-03.9 dashboard validation. It does not claim operational swarm capability.

## Model artifact

- Architecture: YOLO11n, trained on VisDrone.
- Artifact role: visual detector used by the REAL-03.5 video inference/tracking diagnostic.
- SHA-256 (best.pt): `2705051bbecb35493faf0d6d4e4c00de1062616a0f678696b568749c9b76dba5`
- Artifact storage: preserved in the user's Google Drive backup; the binary weights are not included in this Git commit.
- Training run: 50 epochs, imgsz=1280, batch=2, Tesla T4.
- Validation metrics recorded for the trained model: precision=0.557958, recall=0.445809, mAP50=0.452222, mAP50-95=0.274647.

## REAL-03.5 video diagnostic

- Input: OpenCV `vtest.avi`
- Input SHA-256: `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`
- Frames: 795; source rate: 10 FPS; dimensions: 768x576; duration: 79.5 s.
- Runtime device: CPU.
- Detector settings: confidence=0.25, IoU=0.7, imgsz=1280, max_det=317.
- Detections: 6,391.
- Unique track IDs: 71.
- Mean inference latency: 714.171 ms; P95: 974.625 ms; mean inference throughput: 1.4002 FPS.
- Wall-clock duration: 567.766 s.

This is an inference/tracking diagnostic only. The input video has no independent ground-truth labels; therefore, no video detection accuracy or official MOT score is claimed.

## Semantic stability analysis

- Tracks: 71; observations: 6,391.
- Stable: 35.
- Uncertain: 36 (8 LOW_CONFIDENCE + 28 INSUFFICIENT_OBSERVATIONS).
- Consolidated stable classes: pedestrian=33, car=1, van=1.

Stable/uncertain are outcomes of configured semantic rules, not ground-truth classification accuracy.

## Dashboard validation

Dashboard build: REAL-03.9.3, offline artifacts, no live inference.

- Streamlit runtime smoke test: PASS; zero app exceptions.
- Browser interface loaded and rendered persisted metrics.
- H.264 video playback: manually confirmed by the user.
- REAL-03.9.6 class filter: PASS.
- REAL-03.9.6 semantic status filter: PASS.
- REAL-03.9.6.1 track selection: PASS after string-normalized AppTest verification; the earlier FAIL was a type-sensitive comparison false negative.
- AppTest emitted non-fatal ScriptRunContext and Streamlit `use_container_width` deprecation warnings.

## Evidence and limitations

Observed: video frames, detector outputs, track IDs, confidence scores, recorded coordinates and class segments.

Derived: consolidated semantic classes, stability states, trajectories and class distributions.

Not established: ground-truth accuracy for the video, official MOT performance, synchronized multi-sensor fusion performance, physical swarm operation, autonomous navigation or real-world autonomous coordination.

The dashboard reads persisted artifacts. It does not perform live inference.

## Artifact integrity references

- Trained best.pt SHA-256: `2705051bbecb35493faf0d6d4e4c00de1062616a0f678696b568749c9b76dba5`
- Annotated H.264 video SHA-256: `a1c511d1538f31a56ae3c7eab9e2a3f85a9cea63b4545975c6a98c793ae8e474`
- Dashboard app SHA-256 at REAL-03.9.5.3: `5dfb976d3e04639f5d0cc475a84959b5a710f63340b193e803c04d74ba3d9242`

The model and experiment artifacts remain backed up in Google Drive. This repository commit records metadata and validation evidence, not the binary model weights or full video.
