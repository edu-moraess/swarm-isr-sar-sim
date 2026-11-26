# Frontend — SWARM ISR/SAR Operational Workspace

## Scope

The Streamlit application is a presentation layer for recorded experiment artifacts. It does not modify model weights, run inference, or claim live sensor connectivity. The interface is designed around a primary recorded observation feed, selected-contact context, image-plane track geometry, and a track ledger.

## Current data contract

Default experiment root:
`/content/drive/MyDrive/SWARM_ISR_SAR_BACKUPS/DOMAIN_COMPARISON_01/demo_run_01`

Expected files under `inference_01/`:
- `swarm_visdrone_bytetrack_h264.mp4` (preferred when present)
- `swarm_visdrone_bytetrack.mp4` (original fallback)
- `detections.csv`
- `inference_report.json`

The CSV adapter accepts aliases for frame, track ID, class, confidence, and bounding-box coordinates. If required fields are absent, the UI reports the mismatch rather than fabricating data. OpenCV is used for still-frame inspection at the selected frame; the recorded video player remains a separate replay control.

## Interaction model

- Select a tracked contact to highlight its trajectory and inspect its recorded history.
- Select a frame to inspect detections and extract the corresponding still image from the video.
- Review current-frame contacts and the full track ledger.

The Streamlit video element does not provide frame-accurate bidirectional synchronization with the frame selector. The selector-driven still image is the deterministic frame inspection surface.

## Geospatial integrity

The current VisDrone video and detection export provide image-plane pixel coordinates, not geographic coordinates. Track geometry is therefore shown in image pixels. A geographic basemap must only be enabled after a dataset supplies a documented georeferencing transform or valid geographic coordinates with provenance.

## Run

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

For a different experiment directory, set `SWARM_DEMO_ROOT` or edit the directory in the sidebar.
