from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEFAULT_ROOT = Path(
    os.getenv(
        "SWARM_DEMO_ROOT",
        "/content/drive/MyDrive/SWARM_ISR_SAR_BACKUPS/"
        "DOMAIN_COMPARISON_01/demo_run_01",
    )
)

st.set_page_config(
    page_title="SWARM | Tactical Geospatial Command",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --bg:#07111b; --panel:#0d1b29; --line:#1c3447;
            --cyan:#31c5e8; --muted:#7893a8; --green:#a8d84b; }
    .stApp { background:var(--bg); color:#dce8f1; }
    [data-testid="stHeader"] { background:rgba(7,17,27,.92); }
    [data-testid="stSidebar"] { background:#091521; border-right:1px solid var(--line); }
    .block-container { padding-top:1.1rem; max-width:100%; }
    .swarm-top { display:flex; justify-content:space-between; align-items:center;
      border-bottom:1px solid var(--line); padding:0 0 14px 0; margin-bottom:18px; }
    .swarm-brand { font-size:1.15rem; font-weight:800; letter-spacing:.18em; color:#eef7ff; }
    .swarm-sub { color:var(--muted); font-size:.67rem; letter-spacing:.16em; }
    .swarm-status { color:var(--green); border:1px solid #48652b; padding:5px 10px;
      font-size:.68rem; letter-spacing:.12em; border-radius:3px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:5px;
      padding:12px 14px; }
    .eyebrow { color:var(--cyan); font-size:.68rem; letter-spacing:.16em; font-weight:700; }
    .small-muted { color:var(--muted); font-size:.76rem; }
    div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line);
      padding:10px 12px; border-radius:4px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { color:#edf7ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None

def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def load_detections(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Não foi possível ler o CSV: {exc}")
        return pd.DataFrame()

def normalise_detections(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
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
    out = out.dropna(subset=["frame", "track_id", "x1", "y1", "x2", "y2"])
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    out["cx"] = (out["x1"] + out["x2"]) / 2
    out["cy"] = (out["y1"] + out["y2"]) / 2
    return out

def make_track_figure(data: pd.DataFrame, selected_frame: int) -> go.Figure:
    fig = go.Figure()
    if data.empty:
        fig.add_annotation(
            text="TRACK DATA UNAVAILABLE",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#7893a8", size=15),
        )
    else:
        for track_id, group in data.sort_values("frame").groupby("track_id"):
            fig.add_trace(go.Scatter(
                x=group["cx"], y=group["cy"], mode="lines+markers",
                name=f"TRK-{track_id:03d}",
                line=dict(width=1.5), marker=dict(size=4),
                opacity=0.72,
                hovertemplate=(
                    f"TRACK {track_id}<br>Frame %{{customdata}}"
                    "<br>X %{x:.0f} px<br>Y %{y:.0f} px<extra></extra>"
                ),
                customdata=group["frame"],
            ))
        current = data[data["frame"] == selected_frame]
        if not current.empty:
            fig.add_trace(go.Scatter(
                x=current["cx"], y=current["cy"], mode="markers",
                name="CURRENT FRAME",
                marker=dict(size=12, color="#f3b84b", symbol="circle-open",
                            line=dict(width=2)),
                text=current["track_id"].map(lambda x: f"TRK-{x:03d}"),
                hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(
        paper_bgcolor="#0d1b29", plot_bgcolor="#091521",
        font=dict(color="#c8d8e5", size=10),
        margin=dict(l=8, r=8, t=8, b=8), height=490,
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="IMAGE X (PIXELS)", gridcolor="#1c3447", zeroline=False),
        yaxis=dict(title="IMAGE Y (PIXELS)", gridcolor="#1c3447",
                   zeroline=False, autorange="reversed", scaleanchor="x",
                   scaleratio=1),
    )
    return fig

with st.sidebar:
    st.markdown("### ◈ SWARM CONTROL")
    st.caption("DATA SOURCE / LOCAL ARTIFACTS")
    root_text = st.text_input("Experiment directory", str(DEFAULT_ROOT))
    root = Path(root_text).expanduser()
    run_name = st.selectbox("Dataset", ["inference_01"], index=0)
    run_dir = root / run_name
    video_path = run_dir / "swarm_visdrone_bytetrack.mp4"
    csv_path = run_dir / "detections.csv"
    report_path = run_dir / "inference_report.json"
    st.divider()
    st.caption("DISPLAY")
    show_tracks = st.checkbox("Show track trajectories", value=True)
    st.caption("This view is a recorded experiment, not a live sensor feed.")

st.markdown(
    """
    <div class="swarm-top">
      <div><div class="swarm-brand">SWARM <span style="color:#31c5e8">/</span> ISR·SAR</div>
      <div class="swarm-sub">TACTICAL GEOSPATIAL COMMAND · PERCEPTION & TRACKING</div></div>
      <div class="swarm-status">● RECORDED DATA / REPLAY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

raw = load_detections(csv_path)
data = normalise_detections(raw)
report = load_json(report_path)

if raw.empty:
    st.warning("DETECTION CSV NÃO ENCONTRADO. Confira o diretório do experimento na barra lateral.")
elif data.empty:
    st.error(
        "O CSV foi localizado, mas seu esquema não contém os campos necessários "
        "(frame, track_id e coordenadas x1/y1/x2/y2). Nenhum dado foi inventado."
    )

if not data.empty:
    max_frame = int(data["frame"].max())
    min_frame = int(data["frame"].min())
    selected_frame = st.slider("FRAME INSPECTION", min_frame, max_frame, min_frame)
    current = data[data["frame"] == selected_frame]
    total_detections = len(data)
    total_tracks = int(data["track_id"].nunique())
    fps = report.get("model_call_fps", report.get("fps", None))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DETECTIONS", f"{total_detections:,}")
    col2.metric("TRACK IDs", f"{total_tracks:,}")
    col3.metric("FRAMES", f"{data['frame'].nunique():,}")
    col4.metric("CURRENT FRAME", f"{len(current):,}")

    left, right = st.columns([2.25, 1], gap="medium")
    with left:
        st.markdown('<div class="eyebrow">01 / SPATIAL TRACK VIEW</div>', unsafe_allow_html=True)
        st.caption(
            "Image-plane coordinates (pixels). Georeferencing is not present in this dataset."
        )
        plot_data = data if show_tracks else current
        st.plotly_chart(
            make_track_figure(plot_data, selected_frame),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
    with right:
        st.markdown('<div class="eyebrow">02 / VIDEO INTELLIGENCE FEED</div>', unsafe_allow_html=True)
        if video_path.is_file():
            st.video(str(video_path))
        else:
            st.warning(f"Vídeo não encontrado: {video_path}")
        st.markdown('<div class="eyebrow">03 / CONTACT REGISTER</div>', unsafe_allow_html=True)
        if not current.empty:
            cols = [c for c in ["track_id", "class_name", "confidence", "cx", "cy"] if c in current]
            view = current[cols].copy()
            view = view.rename(columns={
                "track_id": "TRACK", "class_name": "CLASS",
                "confidence": "CONF", "cx": "X(px)", "cy": "Y(px)"
            })
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum contato registrado neste frame.")

    st.markdown('<div class="eyebrow">04 / TRACK HISTORY</div>', unsafe_allow_html=True)
    summary = data.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        observations=("frame", "count"),
        class_name=("class_name", "last"),
    ).reset_index()
    summary["span_frames"] = summary["last_frame"] - summary["first_frame"] + 1
    summary["continuity"] = (
        summary["observations"] / summary["span_frames"].clip(lower=1)
    ).round(3)
    st.dataframe(summary.rename(columns={
        "track_id": "TRACK ID", "first_frame": "FIRST",
        "last_frame": "LAST", "observations": "OBSERVATIONS",
        "class_name": "CLASS", "span_frames": "SPAN",
        "continuity": "CONTINUITY",
    }), use_container_width=True, hide_index=True)

    st.download_button(
        "EXPORT CURRENT TRACK REGISTER",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="swarm_track_register.csv",
        mime="text/csv",
    )
else:
    st.info("Aguardando artefatos válidos do experimento para preencher a estação.")

st.markdown("---")
st.caption(
    "SWARM ISR/SAR · Experimental visualization layer · "
    "Recorded observations only · No operational targeting or engagement functions"
)
