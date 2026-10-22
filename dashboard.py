"""
dashboard.py
============
Smart Temple Crowd Management Dashboard (Streamlit)

Features:
  1. Live Video Processing — pick a video, set congested threshold,
     Normal/Moderate auto-calculated, real-time graph + alerts
  2. Crowd Analysis & EDA — dataset exploration
  3. LSTM Model Performance — model evaluation & predictions
  4. Area Configuration — manage temple zones & thresholds
  5. Alerts & Notifications — alert history and delivery log
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

# Ensure working directory is project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from area_config import AreaConfig, AreaManager, CrowdClassifier
from notification_manager import (
    NotificationManager, MockNotificationProvider, AlertRecord,
)
from lstm_model import predict_future_crowd, load_trained_model

DATASET_CSV = os.path.join("output", "crowd_dataset.csv")
EVAL_JSON   = os.path.join("output", "lstm_evaluation.json")
SAMPLE_IMG  = os.path.join("output", "sample_annotated_detection.jpg")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Temple Crowd Management System",
    page_icon="🛕",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .status-normal {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white; padding: 8px 20px; border-radius: 25px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .status-moderate {
        background: linear-gradient(135deg, #ff9800, #ffb74d);
        color: white; padding: 8px 20px; border-radius: 25px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    .status-congested {
        background: linear-gradient(135deg, #f44336, #ef5350);
        color: white; padding: 8px 20px; border-radius: 25px;
        font-weight: bold; font-size: 18px; text-align: center;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3); }
        50% { box-shadow: 0 4px 30px rgba(244, 67, 54, 0.6); }
        100% { box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3); }
    }
    .alert-moderate {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 5px solid #ff9800; padding: 15px 20px;
        border-radius: 8px; margin: 10px 0;
        color: #e65100; font-weight: 500; font-size: 16px;
    }
    .alert-congested {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left: 5px solid #f44336; padding: 15px 20px;
        border-radius: 8px; margin: 10px 0;
        color: #b71c1c; font-weight: 600; font-size: 16px;
    }
    .alert-recovery {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #4caf50; padding: 15px 20px;
        border-radius: 8px; margin: 10px 0;
        color: #1b5e20; font-weight: 500; font-size: 16px;
    }
    .pipeline-banner {
        background-color: #1e222d; padding: 10px 18px; border-radius: 8px;
        margin-bottom: 18px; border: 1px solid #363b4e; text-align: center;
        font-size: 13px; color: #e0e0e0;
    }
    .big-count {
        font-size: 48px; font-weight: 800; text-align: center;
        line-height: 1.1; margin: 5px 0;
    }
    .count-normal { color: #00e676; }
    .count-moderate { color: #ffb74d; }
    .count-congested { color: #ef5350; }
    .notif-entry {
        background: #1a1a2e; border: 1px solid #363b4e;
        border-radius: 8px; padding: 10px 15px; margin: 5px 0;
        font-size: 13px; color: #e0e0e0;
    }
    .threshold-box {
        background: #1a1a2e; padding: 14px 20px; border-radius: 10px;
        border: 1px solid #363b4e; margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "video_processing" not in st.session_state:
    st.session_state.video_processing = False
if "video_chart_data" not in st.session_state:
    st.session_state.video_chart_data = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "classifier" not in st.session_state:
    st.session_state.classifier = CrowdClassifier(escalation_frames=3, deescalation_frames=5)
if "notification_mgr" not in st.session_state:
    st.session_state.notification_mgr = NotificationManager(
        provider=MockNotificationProvider(),
        cooldown_seconds=5,       # short cooldown for demo
        test_mode=True,
    )
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

if st.session_state.stop_requested:
    st.session_state.video_processing = False
    st.session_state.stop_requested = False

area_mgr  = AreaManager()
notif_mgr = st.session_state.notification_mgr

@st.cache_resource
def load_yolo_cached():
    try:
        from ultralytics import YOLO
        mp = os.path.join(BASE_DIR, "yolov8n.pt")
        if os.path.exists(mp):
            return YOLO(mp)
    except Exception as e:
        st.warning(f"Cannot load YOLO: {e}")
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🛕 Smart Temple Crowd Management System")
st.subheader("AI-Based Crowd Detection, Classification, Prediction & Alerts")

st.markdown(
    '<div class="pipeline-banner">'
    '<b>VIDEO INPUT</b> &nbsp;➜&nbsp;'
    '<b>YOLOv8 + BYTETRACK</b> &nbsp;➜&nbsp;'
    '<b>AUTO CLASSIFICATION</b> &nbsp;➜&nbsp;'
    '<b>LSTM PREDICTION</b> &nbsp;➜&nbsp;'
    '<b>REAL-TIME ALERTS</b> &nbsp;➜&nbsp;'
    '<b>TEAM NOTIFICATIONS</b>'
    '</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🛕 Control Center")

# ── 1. Video Selection ───────────────────────────────────────────────────────
st.sidebar.subheader("🎥 Select a Video to Test")

video_folder = os.path.join(BASE_DIR, "videos")
video_files = []
if os.path.isdir(video_folder):
    video_files = sorted([
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
    ])

if video_files:
    selected_video = st.sidebar.selectbox(
        "Choose a video file for testing",
        video_files,
        format_func=lambda x: os.path.basename(x),
        key="video_selector",
    )
    st.sidebar.caption(f"Selected: `{os.path.basename(selected_video)}`")
else:
    selected_video = None
    st.sidebar.warning("No videos in `videos/` folder.")

st.sidebar.divider()

# ── 2. Threshold — User only sets CONGESTED, rest auto-calculated ────────────
st.sidebar.subheader("🎯 Set Congested Threshold")
st.sidebar.caption(
    "Enter the **Congested** threshold only. "
    "Normal and Moderate thresholds are **auto-calculated**."
)

congested_threshold = st.sidebar.number_input(
    "🚨 Congested Threshold (people count)",
    min_value=6, max_value=1000, value=80,
    help="When crowd count >= this value, status = CONGESTED and notifications sent to ALL team members.",
    key="congested_threshold",
)

# Auto-calculate Normal and Moderate thresholds
normal_threshold = int(congested_threshold / 3)
moderate_threshold = int(congested_threshold * 2 / 3)

st.sidebar.markdown(
    f"**Auto-calculated thresholds:**\n"
    f"- 🟢 **Normal**: 0 - {normal_threshold}\n"
    f"- 🟠 **Moderate**: {normal_threshold + 1} - {congested_threshold - 1}\n"
    f"- 🔴 **Congested**: ≥ {congested_threshold}"
)

st.sidebar.divider()

# ── 3. Processing Controls ───────────────────────────────────────────────────
st.sidebar.subheader("⚙️ Processing")

frame_skip = st.sidebar.slider("Process every Nth frame", 1, 15, 3, key="frame_skip_slider")

col_start, col_stop = st.sidebar.columns(2)

if col_start.button(
    "▶ Start", use_container_width=True,
    disabled=st.session_state.video_processing or not selected_video,
):
    st.session_state.video_processing = True
    st.session_state.processing_complete = False
    st.session_state.video_chart_data = []
    st.session_state.alert_log = []
    st.session_state.classifier.reset()
    # Reset notification manager cooldowns for fresh run
    notif_mgr._last_alert_time.clear()
    notif_mgr._current_status.clear()
    st.rerun()

if col_stop.button(
    "⏹ Stop", use_container_width=True,
    disabled=not st.session_state.video_processing,
):
    st.session_state.stop_requested = True
    st.session_state.video_processing = False
    st.session_state.processing_complete = True

st.sidebar.divider()

# ── 4. Pipeline Management ───────────────────────────────────────────────────
st.sidebar.subheader("🔄 Pipeline")

if st.sidebar.button("📊 Process All Videos", use_container_width=True):
    with st.sidebar, st.spinner("Processing videos..."):
        try:
            proc = subprocess.run(
                [sys.executable, "-u", os.path.join(BASE_DIR, "process_videos.py")],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=1200,
            )
            if proc.returncode == 0:
                st.success("✅ Video processing completed!")
                st.rerun()
            else:
                st.error(f"Error:\n{proc.stderr[-500:]}")
        except Exception as e:
            st.error(f"Execution error: {e}")

if st.sidebar.button("🧠 Train / Retrain LSTM", use_container_width=True):
    with st.sidebar, st.spinner("Training LSTM model..."):
        try:
            proc = subprocess.run(
                [sys.executable, "-u", os.path.join(BASE_DIR, "train_lstm.py")],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                st.success("✅ LSTM training completed!")
                st.rerun()
            else:
                st.error(f"Error:\n{proc.stderr[-500:]}")
        except Exception as e:
            st.error(f"Execution error: {e}")

st.sidebar.divider()

# ── 5. Dataset Status ────────────────────────────────────────────────────────
if os.path.exists(DATASET_CSV):
    try:
        df_side = pd.read_csv(DATASET_CSV)
        st.sidebar.success("📊 Dataset Active")
        st.sidebar.metric("Videos Processed", df_side["Video_Name"].nunique())
        st.sidebar.metric("Total Observations", len(df_side))
    except Exception:
        st.sidebar.warning("Reading dataset...")
else:
    st.sidebar.warning("No dataset found.")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_crowd(count, norm_thresh, cong_thresh):
    """Classify: Normal / Moderate / Congested."""
    if count >= cong_thresh:
        return "CONGESTED"
    elif count > norm_thresh:
        return "MODERATE"
    return "NORMAL"


def status_display_name(status):
    return {"NORMAL": "Normal", "MODERATE": "Moderate", "CONGESTED": "Congested"}.get(status, status)


def get_status_color(status):
    return {"CONGESTED": "#f44336", "MODERATE": "#ff9800"}.get(status, "#00c853")


def load_dataset():
    if not os.path.exists(DATASET_CSV):
        return None
    try:
        df = pd.read_csv(DATASET_CSV)
        if df.empty or "People_Count" not in df.columns:
            return None
        if "Threshold" not in df.columns:
            df["Threshold"] = 20
        df["Exceeded"] = df["People_Count"] > df["Threshold"]
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def load_evaluation():
    if os.path.exists(EVAL_JSON):
        try:
            with open(EVAL_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "training_samples": 0, "test_samples": 0, "sequence_length": 10,
        "mae": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0,
        "predictions": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_video, tab_analysis, tab_lstm, tab_areas, tab_alerts = st.tabs([
    "🎥 Live Video Processing",
    "📊 Crowd Analysis & EDA",
    "🧠 LSTM Model Performance",
    "🗺️ Area Configuration",
    "🚨 Alerts & Notifications",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.header("🎥 Real-Time Crowd Monitoring & Alert System")

    # Show threshold classification info
    st.markdown(
        f'<div class="threshold-box">'
        f'<span style="color:#aaa;font-size:14px;">Auto-Calculated Thresholds (Congested = {congested_threshold}):</span><br>'
        f'<span style="color:#00e676;font-weight:bold;font-size:16px;">🟢 NORMAL: 0 - {normal_threshold}</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:#ffb74d;font-weight:bold;font-size:16px;">🟠 MODERATE: {normal_threshold + 1} - {congested_threshold - 1}</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:#ef5350;font-weight:bold;font-size:16px;">🔴 CONGESTED: ≥ {congested_threshold}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.video_processing and not st.session_state.processing_complete:
        st.info(
            "👈 **Step 1:** Select a video in the sidebar\n\n"
            "**Step 2:** Set your Congested threshold (Normal & Moderate are auto-calculated)\n\n"
            "**Step 3:** Click **▶ Start** to begin real-time crowd detection"
        )
        if os.path.exists(SAMPLE_IMG):
            with st.expander("👁 View YOLOv8 Detection Sample", expanded=False):
                st.image(SAMPLE_IMG, caption="Sample annotated frame", width='stretch')

    # ── Active processing loop ────────────────────────────────────────────────
    if st.session_state.video_processing:
        import cv2

        yolo_model = load_yolo_cached()
        if yolo_model is None:
            st.error("❌ YOLO model could not be loaded. Place `yolov8n.pt` in the project root.")
            st.session_state.video_processing = False
        elif selected_video is None or not os.path.exists(selected_video):
            st.error(f"❌ Video file not found: {selected_video}")
            st.session_state.video_processing = False
        else:
            cap = cv2.VideoCapture(selected_video)
            if not cap.isOpened():
                st.error(f"❌ Cannot open video: {selected_video}")
                st.session_state.video_processing = False
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_name = os.path.basename(selected_video)

                st.markdown(
                    f"**Testing Video:** `{vid_name}` | "
                    f"Normal ≤ {normal_threshold} | "
                    f"Moderate ≤ {congested_threshold - 1} | "
                    f"Congested ≥ {congested_threshold}"
                )

                # ── Layout ──
                col_frame, col_info = st.columns([3, 1])
                with col_frame:
                    frame_ph = st.empty()
                with col_info:
                    count_ph   = st.empty()
                    status_ph  = st.empty()
                    time_ph    = st.empty()
                    lstm_ph    = st.empty()

                # Alert area — two containers for persistent + current alerts
                current_alert_ph = st.empty()

                # Real-time chart
                st.subheader("📈 Real-Time Crowd Fluctuation Graph")
                chart_ph = st.empty()

                # Progress
                progress_ph = st.empty()

                # Alert log — grows during processing
                st.subheader("📋 Live Alert Log")
                alert_log_container = st.container()

                chart_data   = []
                frame_idx    = 0
                processed    = 0
                alert_log    = []
                prev_status  = "NORMAL"

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        if frame_idx % frame_skip != 0:
                            frame_idx += 1
                            continue

                        time_sec = round(frame_idx / fps, 1)

                        # ── YOLO detection ──
                        try:
                            results = yolo_model.track(
                                frame, conf=0.40, classes=[0],
                                persist=True, tracker="bytetrack.yaml",
                                verbose=False,
                            )
                            people_count = (
                                len(results[0].boxes)
                                if results[0].boxes is not None else 0
                            )
                            annotated = results[0].plot()
                        except Exception:
                            try:
                                results = yolo_model(
                                    frame, conf=0.40, classes=[0], verbose=False,
                                )
                                people_count = (
                                    len(results[0].boxes)
                                    if results[0].boxes is not None else 0
                                )
                                annotated = results[0].plot()
                            except Exception:
                                people_count = 0
                                annotated = frame

                        # ── Classify ──
                        current_status = classify_crowd(
                            people_count, normal_threshold, congested_threshold
                        )
                        status_color = get_status_color(current_status)

                        # ── Show frame ──
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_ph.image(
                            rgb,
                            caption=f"{vid_name} | t={time_sec}s | Count={people_count} | {status_display_name(current_status)}",
                            width='stretch',
                        )

                        # ── Metrics panel ──
                        count_class = f"count-{current_status.lower()}"
                        count_ph.markdown(
                            f'<div style="text-align:center">'
                            f'<div style="color:#aaa;font-size:12px;">PEOPLE COUNT</div>'
                            f'<div class="big-count {count_class}">{people_count}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        status_class = f"status-{current_status.lower()}"
                        status_ph.markdown(
                            f'<div class="{status_class}">'
                            f'{status_display_name(current_status).upper()}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        time_ph.metric("⏱️ Time", f"{time_sec}s")

                        # ── LSTM prediction ──
                        if len(chart_data) >= 10 and len(chart_data) % 5 == 0:
                            try:
                                recent = [d["Crowd Count"] for d in chart_data[-10:]]
                                pred = predict_future_crowd(recent)
                                pred_status = classify_crowd(pred, normal_threshold, congested_threshold)
                                pred_color = get_status_color(pred_status)
                                lstm_ph.markdown(
                                    f'<div style="text-align:center;margin-top:10px;">'
                                    f'<div style="color:#aaa;font-size:12px;">LSTM PREDICTION</div>'
                                    f'<div style="font-size:28px;font-weight:700;color:{pred_color};">{pred}</div>'
                                    f'<div style="font-size:11px;color:#888;">Next step forecast</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            except Exception:
                                pass

                        # ═══════════════════════════════════════════════════
                        # ALERT SYSTEM
                        # ═══════════════════════════════════════════════════

                        # ── MODERATE threshold crossed: MESSAGE TO YOU ──
                        if current_status == "MODERATE" and prev_status == "NORMAL":
                            alert_entry = {
                                "time": time_sec,
                                "type": "⚠️ MODERATE WARNING",
                                "count": people_count,
                                "message": (
                                    f"Crowd count ({people_count}) crossed Moderate "
                                    f"threshold ({normal_threshold}) at t={time_sec}s"
                                ),
                            }
                            alert_log.append(alert_entry)

                            current_alert_ph.markdown(
                                f'<div class="alert-moderate">'
                                f'⚠️ <b>MODERATE THRESHOLD CROSSED!</b><br>'
                                f'Crowd count: <b>{people_count}</b> '
                                f'(threshold: {normal_threshold + 1})<br>'
                                f'Time: {time_sec}s<br>'
                                f'📩 <b>Warning message sent to YOU (Administrator)</b>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Send via notification manager (MODERATE = admin only)
                            try:
                                notif_mgr.process_crowd_update(
                                    "Temple Area", people_count, "MODERATE", None,
                                )
                            except Exception:
                                pass

                            # Show toast popup
                            try:
                                st.toast(
                                    f"⚠️ MODERATE: Count={people_count} at t={time_sec}s",
                                    icon="⚠️",
                                )
                            except Exception:
                                pass

                        # ── CONGESTED threshold crossed: NOTIFY ALL TEAM ──
                        if current_status == "CONGESTED" and prev_status != "CONGESTED":
                            alert_entry = {
                                "time": time_sec,
                                "type": "🚨 CONGESTED ALERT",
                                "count": people_count,
                                "message": (
                                    f"Crowd count ({people_count}) crossed Congested "
                                    f"threshold ({congested_threshold}) at t={time_sec}s "
                                    f"— Notification sent to ALL team!"
                                ),
                            }
                            alert_log.append(alert_entry)

                            current_alert_ph.markdown(
                                f'<div class="alert-congested">'
                                f'🚨 <b>CONGESTED THRESHOLD CROSSED!</b><br>'
                                f'Crowd count: <b>{people_count}</b> '
                                f'(threshold: {congested_threshold})<br>'
                                f'Time: {time_sec}s<br><br>'
                                f'📩 <b>NOTIFICATION SENT TO ALL PROJECT TEAM MEMBERS:</b><br>'
                                f'&nbsp;&nbsp;• Temple Administrator (admin@temple.local)<br>'
                                f'&nbsp;&nbsp;• Security Head (security@temple.local)<br>'
                                f'&nbsp;&nbsp;• Crowd Control Officer (crowdcontrol@temple.local)'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # Send via notification manager (CONGESTED = ALL recipients)
                            try:
                                notif_mgr.process_crowd_update(
                                    "Temple Area", people_count, "CONGESTED", None,
                                )
                            except Exception:
                                pass

                            # Show toast popup
                            try:
                                st.toast(
                                    f"🚨 CONGESTED: Count={people_count} — Team notified!",
                                    icon="🚨",
                                )
                            except Exception:
                                pass

                        # ── RECOVERY: crowd back to normal ──
                        if current_status == "NORMAL" and prev_status in ("MODERATE", "CONGESTED"):
                            alert_entry = {
                                "time": time_sec,
                                "type": "✅ RECOVERY",
                                "count": people_count,
                                "message": (
                                    f"Crowd recovered to Normal ({people_count}) "
                                    f"at t={time_sec}s"
                                ),
                            }
                            alert_log.append(alert_entry)

                            current_alert_ph.markdown(
                                f'<div class="alert-recovery">'
                                f'✅ <b>CROWD RECOVERED!</b> '
                                f'Count: {people_count} — Status: Normal at t={time_sec}s'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            try:
                                notif_mgr.process_crowd_update(
                                    "Temple Area", people_count, "NORMAL", None,
                                )
                            except Exception:
                                pass

                            try:
                                st.toast(
                                    f"✅ Crowd recovered to Normal ({people_count})",
                                    icon="✅",
                                )
                            except Exception:
                                pass

                        prev_status = current_status

                        # ── Chart data ──
                        chart_data.append({
                            "Time (s)": time_sec,
                            "Crowd Count": people_count,
                            "Moderate Line": normal_threshold + 1,
                            "Congested Line": congested_threshold,
                        })

                        # ── Update chart ──
                        df_chart = pd.DataFrame(chart_data)
                        chart_ph.line_chart(
                            df_chart.set_index("Time (s)"),
                            height=350,
                            color=["#4fc3f7", "#ffb74d", "#ef5350"],
                        )

                        # ── Progress ──
                        pct = min(frame_idx / max(total_frames, 1), 1.0)
                        progress_ph.progress(
                            pct,
                            text=(
                                f"Processing: {int(pct*100)}% | "
                                f"Frame {frame_idx}/{total_frames} | "
                                f"Count: {people_count} | "
                                f"Status: {status_display_name(current_status)} | "
                                f"Alerts: {len(alert_log)}"
                            ),
                        )

                        # ── Update alert log ──
                        if alert_log:
                            log_html = ""
                            for entry in reversed(alert_log[-15:]):
                                log_html += (
                                    f'<div class="notif-entry">'
                                    f'<b>{entry["type"]}</b> | '
                                    f't={entry["time"]}s | '
                                    f'Count: {entry["count"]} | '
                                    f'{entry["message"]}'
                                    f'</div>'
                                )
                            alert_log_container.markdown(log_html, unsafe_allow_html=True)

                        processed += 1
                        frame_idx += 1

                finally:
                    cap.release()
                    st.session_state.video_processing = False
                    st.session_state.processing_complete = True
                    st.session_state.video_chart_data = chart_data
                    st.session_state.alert_log = alert_log

                st.success(
                    f"✅ Processing complete — {processed} frames from `{vid_name}` | "
                    f"{len(alert_log)} alerts triggered"
                )

    # ── Post-processing results ──────────────────────────────────────────────
    if st.session_state.processing_complete and st.session_state.video_chart_data:
        st.divider()
        st.subheader("📈 Final Crowd Fluctuation Chart")
        df_result = pd.DataFrame(st.session_state.video_chart_data)
        st.line_chart(
            df_result.set_index("Time (s)"),
            height=400,
            color=["#4fc3f7", "#ffb74d", "#ef5350"],
        )

        counts = [d["Crowd Count"] for d in st.session_state.video_chart_data]
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        col_s1.metric("Max Count", max(counts))
        col_s2.metric("Min Count", min(counts))
        col_s3.metric("Avg Count", f"{np.mean(counts):.1f}")
        col_s4.metric("Frames", len(counts))
        col_s5.metric("Alerts", len(st.session_state.alert_log))

        if st.session_state.alert_log:
            st.divider()
            st.subheader("📋 Complete Alert Summary")
            for entry in st.session_state.alert_log:
                if "CONGESTED" in entry["type"]:
                    st.error(f"**{entry['type']}** at t={entry['time']}s — Count: {entry['count']} — 📩 All team notified!")
                elif "MODERATE" in entry["type"]:
                    st.warning(f"**{entry['type']}** at t={entry['time']}s — Count: {entry['count']} — Message sent to you")
                else:
                    st.success(f"**{entry['type']}** at t={entry['time']}s — Count: {entry['count']}")

        with st.expander("📋 Raw Data Table"):
            st.dataframe(df_result, width='stretch', hide_index=True)
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "processing_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CROWD ANALYSIS & EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("📊 Dynamic Crowd Analysis & EDA")
    df_data = load_dataset()

    if df_data is None:
        st.warning("No crowd dataset found. Click 'Process All Videos' in the sidebar.")
    else:
        num_videos = df_data["Video_Name"].nunique()
        total_obs  = len(df_data)
        total_det  = int(df_data["People_Count"].sum())
        avg_count  = round(df_data["People_Count"].mean(), 2)
        alert_obs  = int(df_data["Exceeded"].sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Videos", num_videos)
        col2.metric("Observations", total_obs)
        col3.metric("Total Detections", f"{total_det:,}")
        col4.metric("Avg Count", f"{avg_count:.2f}")
        col5.metric("Alert Frames", alert_obs)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("A. People Count vs Time")
            pivoted = df_data.pivot_table(
                index="Time_Seconds", columns="Video_Name",
                values="People_Count", aggfunc="mean",
            ).sort_index().interpolate().ffill().bfill()
            st.line_chart(pivoted, height=320)
        with c2:
            st.subheader("B. Video Comparison")
            avg_per = df_data.groupby("Video_Name")["People_Count"].mean().reset_index()
            avg_per.columns = ["Video", "Avg Count"]
            st.bar_chart(avg_per.set_index("Video"), height=320)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("C. Min/Mean/Max per Video")
            vs = df_data.groupby("Video_Name")["People_Count"].agg(["mean","min","max"]).reset_index()
            vs.columns = ["Video", "Mean", "Min", "Max"]
            st.bar_chart(vs.set_index("Video"), height=320)
        with c4:
            st.subheader("D. Crowd Status Distribution")
            sd = df_data["Crowd_Status"].value_counts().reset_index()
            sd.columns = ["Status", "Count"]
            st.bar_chart(sd.set_index("Status"), height=320)

        st.divider()
        st.subheader("📋 Per-Video Summary")
        agg = {
            "Samples": ("People_Count","count"), "Min": ("People_Count","min"),
            "Max": ("People_Count","max"), "Avg": ("People_Count","mean"),
            "Threshold": ("Threshold","first"), "Alerts": ("Exceeded","sum"),
        }
        summary = df_data.groupby("Video_Name").agg(**agg).reset_index()
        summary["Avg"] = summary["Avg"].round(1)
        st.dataframe(summary, width='stretch', hide_index=True)

        with st.expander("📥 Download Full Dataset"):
            st.dataframe(df_data, width='stretch', hide_index=True)
            st.download_button("⬇️ Download CSV", df_data.to_csv(index=False).encode(), "crowd_dataset.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LSTM MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_lstm:
    st.header("🧠 LSTM Model Performance & Evaluation")
    ev = load_evaluation()

    st.info("Predictions below are LSTM model estimates, not measured counts.")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Train Seq.", ev.get("training_samples", 0))
    m2.metric("Val Seq.", ev.get("validation_samples", 0))
    m3.metric("Test Seq.", ev.get("test_samples", 0))
    m4.metric("MAE", f"{ev.get('mae', 0.0):.4f}")
    m5.metric("RMSE", f"{ev.get('rmse', 0.0):.4f}")
    m6.metric("R²", f"{ev.get('r2_score', 0.0):.4f}")
    m7.metric("Accuracy", f"{ev.get('accuracy', 0.0) * 100:.2f}%")

    tc = ev.get("training_loss_curve", [])
    vc = ev.get("validation_loss_curve", [])
    if tc and vc:
        st.divider()
        st.subheader("📉 Loss Curves")
        df_loss = pd.DataFrame({"Epoch": range(1,len(tc)+1), "Train":tc, "Val":vc})
        st.line_chart(df_loss.set_index("Epoch"), height=300)

    st.divider()
    st.subheader("📈 Actual vs Predicted")
    preds = ev.get("predictions", [])
    if preds:
        df_p = pd.DataFrame(preds)
        ct = "Target_Time" if "Target_Time" in df_p.columns else "Target Time"
        ca = "Actual_Count" if "Actual_Count" in df_p.columns else "Actual Count"
        cp = "Predicted_Count" if "Predicted_Count" in df_p.columns else "Predicted Count"
        ce = "Absolute_Error" if "Absolute_Error" in df_p.columns else "Absolute Error"
        df_p["Label"] = df_p["Video"] + " (" + df_p[ct].astype(str) + "s)"
        chart = df_p.set_index("Label")[[ca,cp]].rename(columns={ca:"Actual",cp:"Predicted"})
        st.line_chart(chart, height=350)

        c_tbl, c_cm = st.columns([3,2])
        with c_tbl:
            st.subheader("📋 Predictions Table")
            st.dataframe(df_p[["Video",ct,ca,cp,ce]], hide_index=True)
        with c_cm:
            st.subheader("🎯 Confusion Matrix")
            cm = ev.get("confusion_matrix", {})
            if cm:
                st.dataframe(pd.DataFrame(cm).T, width='stretch')
    else:
        st.info("No evaluation data. Train the LSTM model first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AREA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_areas:
    st.header("🗺️ Temple Area Configuration")

    areas = area_mgr.list_areas()
    if areas:
        rows = []
        for name in areas:
            a = area_mgr.get_area(name)
            rows.append({
                "Area": a.name,
                "Moderate Threshold": a.count_threshold_moderate,
                "Congested Threshold": a.count_threshold_congested,
                "Max Capacity": a.max_capacity,
                "Area (m²)": a.area_sqm or "N/A",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.divider()
    st.subheader("➕ Add or Edit Area")
    with st.form("area_form"):
        a_name = st.text_input("Area Name", value="Main Sanctum")
        a_mod  = st.number_input("Moderate Threshold", min_value=1, value=40)
        a_cong = st.number_input("Congested Threshold", min_value=2, value=80)
        a_cap  = st.number_input("Max Capacity", min_value=1, value=200)
        a_sqm  = st.number_input("Area (m²)", min_value=1.0, value=100.0)
        if st.form_submit_button("💾 Save Area"):
            try:
                new_area = AreaConfig(
                    name=a_name, count_threshold_moderate=a_mod,
                    count_threshold_congested=a_cong, max_capacity=a_cap, area_sqm=a_sqm,
                )
                area_mgr.add_area(new_area)
                st.success(f"✅ Area '{a_name}' saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALERTS & NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.header("🚨 Alerts & Notifications")

    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric("Mode", "Test Mode" if notif_mgr.test_mode else "Production")
    col_n2.metric("Cooldown", f"{notif_mgr.cooldown_seconds}s")
    col_n3.metric("Total Alerts", len(notif_mgr.get_recent_alerts(limit=100)))

    st.divider()
    st.subheader("🧪 Send Test Notification")
    with st.form("test_notif"):
        tc = st.number_input("Count", min_value=0, value=100)
        ts = st.selectbox("Status", ["NORMAL", "MODERATE", "CONGESTED"])
        if st.form_submit_button("🔔 Send Test Alert"):
            record = notif_mgr.process_crowd_update("Test Area", tc, ts, None)
            if record:
                st.success(f"✅ Alert sent — {ts}, Count={tc}, Notified: {', '.join(record.recipients_notified)}")
            else:
                st.info("No notification (cooldown active or Normal status).")

    st.divider()
    st.subheader("📜 Alert History")
    recent = notif_mgr.get_recent_alerts(limit=30)
    if recent:
        rows = []
        for r in reversed(recent):
            rows.append({
                "Time": r.timestamp, "Area": r.area_name, "Count": r.crowd_count,
                "Status": r.crowd_status, "Previous": r.previous_status,
                "Recovery": "✅" if r.is_recovery else "",
                "Notified": ", ".join(r.recipients_notified), "Reason": r.reason,
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        log_json = json.dumps([{"time":r.timestamp,"area":r.area_name,"count":r.crowd_count,
            "status":r.crowd_status,"notified":r.recipients_notified} for r in recent], indent=2).encode()
        st.download_button("⬇️ Download Log", log_json, "alert_log.json", "application/json")
    else:
        st.info("No alerts yet. Process a video or send a test alert.")

    live_path = os.path.join(BASE_DIR, "output", "live_state.json")
    if os.path.exists(live_path):
        try:
            with open(live_path) as f:
                ls = json.load(f)
            with st.expander("🔴 Live State"):
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Status", ls.get("status", "N/A"))
                l2.metric("Crowd", ls.get("current_crowd", "N/A"))
                l3.metric("Predicted", ls.get("predicted_crowd", "N/A"))
                l4.metric("Updated", ls.get("timestamp", "N/A"))
        except Exception:
            pass
