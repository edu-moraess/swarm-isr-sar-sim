# Frontend — Tactical Geospatial Command

## Scope

The Streamlit application is a presentation layer for recorded experiment artifacts.
It does not modify model weights, run inference, or claim live sensor connectivity.

## Current data contract

Default experiment root:
`/content/drive/MyDrive/SWARM_ISR_SAR_BACKUPS/DOMAIN_COMPARISON_01/demo_run_01`

Expected files under `inference_01/`:
- `swarm_visdrone_bytetrack.mp4`
- `detections.csv`
- `inference_report.json`

The CSV adapter accepts common aliases for frame, track ID, class, confidence,
and bounding-box coordinates. If the required fields are absent, the UI reports
the mismatch instead of fabricating data.

## Geospatial integrity

The current VisDrone video and detection export provide image-plane pixel
coordinates, not geographic coordinates. The primary spatial panel therefore
plots image-plane track centroids. A geographic basemap must only be enabled
after a dataset supplies a documented georeferencing transform or valid
coordinates with provenance.

## Run

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

For a different experiment directory, set `SWARM_DEMO_ROOT` or edit the
directory in the sidebar.
