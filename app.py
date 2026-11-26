from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_ROOT = Path(os.getenv(
    "SWARM_DEMO_ROOT",
    "/content/drive/MyDrive/SWARM_ISR_SAR_BACKUPS/DOMAIN_COMPARISON_01/demo_run_01",
))
COLORS = ["#4cc9f0", "#a3e635", "#fbbf24", "#f472b6", "#a78bfa", "#fb7185", "#2dd4bf"]

st.set_page_config(
    page_title="SWARM ISR/SAR | Operational Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg:#080d13; --surface:#0d151e; --surface2:#111d28; --line:#233342;
  --text:#e5edf4; --muted:#8293a3; --cyan:#56c8e8; --green:#9bcf62;
  --amber:#e6b45c; --red:#e47777;
}
html, body, [class*="css"] { font-family:Inter, sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,13,19,.94); }
[data-testid="stSidebar"] { background:#0b131c; border-right:1px solid var(--line); }
.block-container { max-width:1600px; padding:1.05rem 1.35rem 2rem; }
#MainMenu, footer { visibility:hidden; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line);
  padding:10px 13px; border-radius:3px; }
div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.68rem; letter-spacing:.11em; }
div[data-testid="stMetricValue"] { color:var(--text); font-family:'IBM Plex Mono',monospace; }
.stButton button, .stDownloadButton button { border-radius:3px; border:1px solid var(--line);
  background:#142331; color:var(--text); }
.stSelectbox label, .stSlider label { color:var(--muted); font-size:.72rem; letter-spacing:.08em; }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
hr { border-color:var(--line); }
.swarm-head { display:flex; justify-content:space-between; align-items:center;
  gap:16px; padding:4px 0 15px; border-bottom:1px solid var(--line); margin-bottom:14px; }
.brand { font-size:1.12rem; font-weight:800; letter-spacing:.19em; color:#f2f7fb; }
.brand span { color:var(--cyan); }
.subbrand { color:var(--muted); font:500 .62rem 'IBM Plex Mono',monospace;
  letter-spacing:.13em; margin-top:4px; }
.status { border:1px solid #536d3b; color:var(--green); padding:6px 9px;
  border-radius:2px; font:600 .62rem 'IBM Plex Mono',monospace; letter-spacing:.08em; white-space:nowrap; }
.section-label { color:var(--cyan); font:600 .66rem 'IBM Plex Mono',monospace;
  letter-spacing:.14em; text-transform:uppercase; margin:2px 0 9px; }
.panel { background:var(--surface); border:1px solid var(--line); padding:12px; border-radius:3px; }
.panel-title { color:#dbe7ef; font-size:.78rem; font-weight:700; letter-spacing:.08em; }
.panel-meta { color:var(--muted); font: .62rem 'IBM Plex Mono',monospace; }
.kv { display:flex; justify-content:space-between; gap:10px; padding:7px 0;
  border-bottom:1px solid #1b2a37; font-size:.72rem; }
.kv:last-child { border-bottom:0; }
.kv .k { color:var(--muted); }
.kv .v { color:var(--text); font-family:'IBM Plex Mono',monospace; text-align:right; }
.track-chip { display:inline-block; color:var(--cyan); border:1px solid #31556a;
  padding:3px 7px; font:600 .65rem 'IBM Plex Mono',monospace; border-radius:2px; }
.note { color:var(--muted); font-size:.68rem; line-height:1.5; }
@media (max-width: 700px) {
  .block-container { padding:.65rem .65rem 1.5rem; }
  .swarm-head { flex-wrap:wrap; align-items:flex-start; gap:9px; }
  .brand { font-size:.98rem; }
  .subbrand { font-size:.54rem; line-height:1.5; }
  .status { font-size:.55rem; padding:5px 6px; white-space:normal; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.65rem; }
  [data-testid="column"] { flex:1 1 100% !important; min-width:0 !important; }
  [data-testid="stDataFrame"] { overflow-x:auto; }
  .section-label { line-height:1.5; }
}
</style>
""", unsafe_allow_html=True)


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


@st.cache_data(show_spinner=False)
def load_csv(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    path = Path(path_text)
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report(path_text: str, modified: float) -> dict:
    del modified
    path = Path(path_text)
    if not path.is_file():
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def read_video_frame(path_text: str, frame_number: int, modified: float):
    del modified
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(path_text)
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    aliases = {
        "frame": ("frame", "frame_id", "frame_idx", "frame_number"),
        "track_id": ("track_id", "track", "id", "tracker_id"),
        "class_name": ("class_name", "class", "label", "name"),
        "confidence": ("confidence", "conf", "score"),
        "x1": ("x1", "xmin", "left", "bbox_x1"),
        "y1": ("y1", "ymin", "top", "bbox_y1"),
        "x2": ("x2", "xmax", "right", "bbox_x2"),
        "y2": ("y2", "ymax", "bottom", "bbox_y2"),
    }
    out = pd.DataFrame(index=raw.index)
    for target, names in aliases.items():
        col = find_column(raw, names)
        if col is not None:
            out[target] = raw[col]
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    for col in ("frame", "track_id", "confidence", "x1", "y1", "x2", "y2"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "class_name" not in out:
        out["class_name"] = "unknown"
    out = out.dropna(subset=["frame", "track_id", "x1", "y1", "x2", "y2"]).copy()
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    out["cx"] = (out["x1"] + out["x2"]) / 2
    out["cy"] = (out["y1"] + out["y2"]) / 2
    return out


def track_figure(data: pd.DataFrame, selected_id: int | None, current_frame: int) -> go.Figure:
    fig = go.Figure()
    if data.empty:
        fig.add_annotation(text="NO TRACK OBSERVATIONS", x=.5, y=.5, xref="paper",
                           yref="paper", showarrow=False, font=dict(color="#8293a3"))
    else:
        ids = sorted(data["track_id"].unique())
        for index, track_id in enumerate(ids):
            part = data[data["track_id"] == track_id].sort_values("frame")
            active = selected_id == int(track_id)
            fig.add_trace(go.Scatter(
                x=part["cx"], y=part["cy"], mode="lines+markers",
                name=f"TRK-{int(track_id):03d}",
                line=dict(width=2.4 if active else 1, color=COLORS[index % len(COLORS)]),
                marker=dict(size=5 if active else 3),
                opacity=1 if active else .28,
                hovertemplate=f"TRACK {int(track_id)}<br>FRAME %{{customdata}}<br>X %{{x:.0f}} px<br>Y %{{y:.0f}} px<extra></extra>",
                customdata=part["frame"],
                showlegend=active or selected_id is None,
            ))
        at_frame = data[data["frame"] == current_frame]
        if not at_frame.empty:
            fig.add_trace(go.Scatter(
                x=at_frame["cx"], y=at_frame["cy"], mode="markers",
                name="FRAME CONTACTS",
                marker=dict(size=10, color="#f5c46a", symbol="circle-open", line=dict(width=2)),
                text=at_frame["track_id"].map(lambda value: f"TRK-{int(value):03d}"),
                hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#0d151e", plot_bgcolor="#0a121a",
        font=dict(color="#b9c8d4", size=9),
        legend=dict(orientation="h", y=-.23, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="IMAGE X / PX", gridcolor="#1d2d3b", zeroline=False),
        yaxis=dict(title="IMAGE Y / PX", gridcolor="#1d2d3b", zeroline=False,
                   autorange="reversed", scaleanchor="x", scaleratio=1),
    )
    return fig


root_text = str(DEFAULT_ROOT)
with st.sidebar:
    st.markdown("### SWARM / DATA SOURCE")
    root_text = st.text_input("Experiment directory", root_text)
    st.caption("Recorded experiment artifacts only.")
root = Path(root_text).expanduser()
run_dir = root / "inference_01"
h264 = run_dir / "swarm_visdrone_bytetrack_h264.mp4"
original = run_dir / "swarm_visdrone_bytetrack.mp4"
video = h264 if h264.is_file() else original
csv_path = run_dir / "detections.csv"
report_path = run_dir / "inference_report.json"

st.markdown("""
<div class="swarm-head">
  <div><div class="brand">SWARM <span>/</span> ISR·SAR</div>
  <div class="subbrand">PERCEPTION SYSTEMS &nbsp; / &nbsp; RECORDED EXPERIMENT WORKSPACE</div></div>
  <div class="status">● REPLAY MODE &nbsp; / &nbsp; OFFLINE DATA</div>
</div>
""", unsafe_allow_html=True)

raw = load_csv(str(csv_path), csv_path.stat().st_mtime if csv_path.exists() else 0)
data = normalise(raw)
report = load_report(str(report_path), report_path.stat().st_mtime if report_path.exists() else 0)

if data.empty:
    st.error("Detection data unavailable or incompatible. Verify the experiment directory and CSV schema.")
    st.stop()

min_frame, max_frame = int(data.frame.min()), int(data.frame.max())
all_ids = sorted(int(value) for value in data.track_id.unique())
if "swarm_selected_track" not in st.session_state or st.session_state.swarm_selected_track not in all_ids:
    st.session_state.swarm_selected_track = all_ids[0]

toolbar_a, toolbar_b, toolbar_c = st.columns([1.25, 1.25, 2.5])
with toolbar_a:
    selected_id = st.selectbox("SELECTED CONTACT", all_ids,
        index=all_ids.index(st.session_state.swarm_selected_track),
        format_func=lambda value: f"TRK-{value:03d}", key="swarm_selected_track")
with toolbar_b:
    selected_frame = st.select_slider("FRAME", options=list(range(min_frame, max_frame + 1)),
                                      value=min_frame, key="swarm_frame")
with toolbar_c:
    st.caption("EXPERIMENT / VISDRONE-VID · IMAGE-PLANE COORDINATES · NOT GEOREFERENCED")

current = data[data.frame == selected_frame]
selected_track = data[data.track_id == selected_id].sort_values("frame")
selected_now = current[current.track_id == selected_id]
track_first = int(selected_track.frame.min())
track_last = int(selected_track.frame.max())
track_observations = len(selected_track)
class_counts = selected_track["class_name"].fillna("unknown").astype(str).str.lower().value_counts()
dominant_class = str(class_counts.index[0]).upper() if not class_counts.empty else "UNKNOWN"
class_variants = class_counts[class_counts.index != class_counts.index[0]] if len(class_counts) > 1 else pd.Series(dtype=int)
class_distribution = " / ".join(
    f"{str(name).upper()} {int(count)}" for name, count in class_counts.items()
)
current_class = (
    str(selected_now.iloc[0]["class_name"]).upper()
    if not selected_now.empty else "NOT OBSERVED"
)
span = max(1, track_last - track_first + 1)
continuity = track_observations / span

m1, m2, m3, m4 = st.columns(4)
m1.metric("DETECTIONS", f"{len(data):,}")
m2.metric("TRACKED CONTACTS", f"{len(all_ids):02d}")
m3.metric("VIDEO FRAMES", f"{data.frame.nunique():02d}")
m4.metric("FRAME CONTACTS", f"{len(current):02d}")

left, right = st.columns([2.15, 1], gap="medium")
with left:
    st.markdown('<div class="section-label">01 / Primary observation feed</div>', unsafe_allow_html=True)
    if video.is_file():
        st.video(str(video))
        st.caption("RECORDED VIDEO PLAYBACK · use FRAME inspection below for deterministic frame selection.")
        with st.expander(f"INSPECT SELECTED FRAME · {selected_frame:03d} / {max_frame:03d}", expanded=False):
            if cv2 is not None:
                frame = read_video_frame(str(video), int(selected_frame), video.stat().st_mtime)
                if frame is not None:
                    st.image(frame, use_container_width=True)
                    st.caption("Still frame decoded independently from the recorded video; it does not control the video player's playback position.")
                else:
                    st.warning("Selected frame could not be decoded.")
            else:
                st.caption("Install opencv-python-headless to enable frame-by-frame inspection.")
    else:
        st.warning("Recorded video file not found.")
with right:
    st.markdown('<div class="section-label">02 / Contact intelligence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="track-chip">TRK-{selected_id:03d}</div><div style="height:9px"></div><div class="panel-title">DOMINANT CLASS · {dominant_class}</div><div class="panel-meta">TRACK HISTORY / RECORDED OBSERVATIONS</div><div style="height:8px"></div><div class="kv"><span class="k">Class distribution</span><span class="v">{class_distribution}</span></div><div class="kv"><span class="k">Current frame class</span><span class="v">{current_class}</span></div><div class="kv"><span class="k">First observation</span><span class="v">F{track_first:03d}</span></div><div class="kv"><span class="k">Last observation</span><span class="v">F{track_last:03d}</span></div><div class="kv"><span class="k">Observations</span><span class="v">{track_observations}</span></div><div class="kv"><span class="k">Temporal coverage*</span><span class="v">{continuity:.1%}</span></div></div>', unsafe_allow_html=True)
    st.caption("*Observations divided by frame span; descriptive, not a tracking-quality probability.")
    if not selected_now.empty:
        row = selected_now.iloc[0]
        conf = f"{float(row.confidence):.3f}" if "confidence" in row and pd.notna(row.confidence) else "N/A"
        st.markdown('<div class="section-label">Current frame observation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><div class="kv"><span class="k">Detection confidence</span><span class="v">{conf}</span></div><div class="kv"><span class="k">Centroid X</span><span class="v">{row.cx:.1f} px</span></div><div class="kv"><span class="k">Centroid Y</span><span class="v">{row.cy:.1f} px</span></div></div>', unsafe_allow_html=True)
    else:
        st.info("Selected contact has no observation in this frame.")

st.markdown('<div class="section-label">03 / Track geometry & temporal history</div>', unsafe_allow_html=True)
geometry_col, ledger_col = st.columns([1.55, 1], gap="medium")
with geometry_col:
    st.plotly_chart(track_figure(data, selected_id, int(selected_frame)),
                    use_container_width=True, config={"displaylogo": False, "responsive": True},
                    key="swarm_track_geometry_v2")
    st.caption("Centroid trajectories in image pixels. Highlighted line = selected contact.")
with ledger_col:
    st.markdown('<div class="panel-title">CONTACT REGISTER / CURRENT FRAME</div>', unsafe_allow_html=True)
    if not current.empty:
        view = current[["track_id", "class_name", "confidence", "cx", "cy"]].copy()
        view.columns = ["TRACK", "FRAME CLASS", "CONF", "X / PX", "Y / PX"]
        st.dataframe(view, use_container_width=True, hide_index=True, height=245)
    else:
        st.info("No detections in selected frame.")

st.markdown('<div class="section-label">04 / Track ledger</div>', unsafe_allow_html=True)
ledger = data.groupby("track_id").agg(
    first_frame=("frame", "min"), last_frame=("frame", "max"),
    observations=("frame", "count"),
).reset_index()
dominant = (
    data.assign(class_name=data["class_name"].fillna("unknown").astype(str).str.lower())
        .groupby(["track_id", "class_name"]).size()
        .rename("count").reset_index()
        .sort_values(["track_id", "count", "class_name"], ascending=[True, False, True])
        .drop_duplicates("track_id")
        .set_index("track_id")["class_name"]
)
ledger["dominant_class"] = ledger["track_id"].map(dominant).str.upper()
ledger["class_variants"] = ledger["track_id"].map(
    data.assign(class_name=data["class_name"].fillna("unknown").astype(str).str.lower())
        .groupby("track_id")["class_name"]
        .agg(lambda values: " / ".join(
            f"{name.upper()} {count}" for name, count in values.value_counts().items()
        ))
)
ledger["coverage"] = (ledger.observations / (ledger.last_frame - ledger.first_frame + 1).clip(lower=1)).round(3)
ledger = ledger[["track_id", "first_frame", "last_frame", "observations", "dominant_class", "class_variants", "coverage"]]
ledger.columns = ["TRACK ID", "FIRST FRAME", "LAST FRAME", "OBSERVATIONS", "DOMINANT CLASS", "CLASS DISTRIBUTION", "FRAME COVERAGE"]
st.dataframe(ledger, use_container_width=True, hide_index=True)
st.download_button("EXPORT TRACK LEDGER", ledger.to_csv(index=False).encode("utf-8"),
                   "swarm_track_ledger.csv", "text/csv")

st.markdown("---")
st.markdown('<div class="note">SWARM ISR/SAR / EXPERIMENTAL WORKSPACE &nbsp; · &nbsp; RECORDED DATA ONLY &nbsp; · &nbsp; NO LIVE SENSOR LINK &nbsp; · &nbsp; NO GEOGRAPHIC COORDINATES IN THIS DATASET</div>', unsafe_allow_html=True)
