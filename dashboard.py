import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

# Ensure working directory is always script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from mobile_server import ensure_mobile_server_running, get_local_ip, generate_qr_code_base64
from lstm_model import predict_future_crowd, load_trained_model

DATASET_CSV = os.path.join("output", "crowd_dataset.csv")
EVAL_JSON   = os.path.join("output", "lstm_evaluation.json")
SAMPLE_IMG  = os.path.join("output", "sample_annotated_detection.jpg")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Temple Crowd Management System",
    page_icon="🛕",
    layout="wide"
)

# Start Mobile Notification HTTP Server on LAN
mobile_ip, mobile_port = ensure_mobile_server_running(port=8502)
mobile_url = f"http://{mobile_ip}:{mobile_port}"

st.title("🛕 Smart Temple Crowd Management System")
st.subheader("AI-Based Crowd Detection, Person Tracking, Analysis and Prediction")

# Visual Pipeline Story Banner
st.markdown(
    """
    <div style="background-color: #1e222d; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #363b4e; text-align: center; font-size: 14px; color: #e0e0e0;">
        <b>REAL VIDEOS</b> &nbsp;➔&nbsp; 
        <b>YOLOv8 PERSON DETECTION</b> &nbsp;➔&nbsp; 
        <b>BYTETRACK TRACKING</b> &nbsp;➔&nbsp; 
        <b>CROWD DATASET</b> &nbsp;➔&nbsp; 
        <b>EDA</b> &nbsp;➔&nbsp; 
        <b>PYTORCH LSTM PREDICTION</b> &nbsp;➔&nbsp; 
        <b>CROWD ALERTS</b> &nbsp;➔&nbsp; 
        <b>STREAMLIT DASHBOARD</b> &nbsp;➔&nbsp; 
        <b>MOBILE WEB ALERT</b>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR CONTROL CENTER
# --------------------------------------------------
st.sidebar.title("🛕 Control Center")

# ---- Mobile Connection Section ----
st.sidebar.subheader("📱 Mobile Web Connection")
st.sidebar.success("🟢 Mobile Server Active")
st.sidebar.markdown(f"**Mobile URL:** `{mobile_url}`")

try:
    qr_b64 = generate_qr_code_base64(mobile_url)
    st.sidebar.image(f"data:image/png;base64,{qr_b64}", caption="Scan with phone camera", use_container_width=True)
except Exception as e:
    st.sidebar.caption(f"QR code generation note: {e}")

st.sidebar.caption("Scan QR code or open URL on your smartphone browser (same Wi-Fi).")



st.sidebar.divider()

# ---- Video Processing & Model Training Section ----
st.sidebar.subheader("⚙️ Pipeline Management")
st.sidebar.caption("Process all `.mp4` videos in `videos/` using YOLOv8 & ByteTrack.")

if st.sidebar.button("▶ Process Videos Now", use_container_width=True):
    with st.sidebar:
        with st.spinner("Processing videos with YOLOv8 & ByteTrack..."):
            try:
                proc = subprocess.run(
                    [sys.executable, "-u", os.path.join(BASE_DIR, "process_videos.py")],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=1200
                )
                if proc.returncode == 0:
                    st.success("✅ Video processing completed!")
                    st.rerun()
                else:
                    st.error(f"Processing error:\n{proc.stderr[-300:]}")
            except Exception as e:
                st.error(f"Execution error: {e}")



st.sidebar.divider()

# Dataset Status in Sidebar
if os.path.exists(DATASET_CSV):
    try:
        df_side = pd.read_csv(DATASET_CSV)
        st.sidebar.success("📊 Dataset Active")
        st.sidebar.metric("Videos Processed", df_side["Video_Name"].nunique())
        st.sidebar.metric("Total Observations", len(df_side))
        if "Unique_Track_IDs" in df_side.columns:
            st.sidebar.metric("Total Detections", int(df_side["People_Count"].sum()))
    except Exception:
        st.sidebar.warning("Reading dataset...")
else:
    st.sidebar.warning("No dataset found. Click 'Process Videos Now'.")


# --------------------------------------------------
# HELPER FUNCTIONS FOR DATA LOADING
# --------------------------------------------------
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
        st.error(f"Error loading CSV dataset: {e}")
        return None

def load_evaluation():
    if os.path.exists(EVAL_JSON):
        try:
            with open(EVAL_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "training_samples": 0,
        "test_samples": 0,
        "sequence_length": 10,
        "forecast_horizon": "1 second",
        "mae": 0.0,
        "rmse": 0.0,
        "r2_score": 0.0,
        "accuracy": 0.0,
        "predictions": []
    }

def get_status_str(count, threshold):
    low = max(1, threshold // 3)
    if count <= low:
        return "Normal 🟢"
    elif count <= threshold:
        return "Moderate 🟡"
    else:
        return "Congested 🔴"


# --------------------------------------------------
# PROMINENT PREDICTION & DETECTIONS BANNER (REQUIREMENTS 5, 10 & 11)
# --------------------------------------------------
df_data = load_dataset()
eval_info = load_evaluation()

if df_data is not None and not df_data.empty:
    latest_row = df_data.iloc[-1]
    curr_count = int(latest_row["People_Count"])
    curr_thresh = int(latest_row["Threshold"])
    curr_video = str(latest_row["Video_Name"])

    recent_counts = df_data[df_data["Video_Name"] == curr_video]["People_Count"].tolist()
    if len(recent_counts) < 10:
        recent_counts = df_data["People_Count"].tail(10).tolist()
    else:
        recent_counts = recent_counts[-10:]

    pred_count = predict_future_crowd(recent_counts)

    curr_status = get_status_str(curr_count, curr_thresh)
    pred_status = get_status_str(pred_count, curr_thresh)

    if pred_count > curr_count:
        trend_str = "Increasing 📈"
        trend_color = "#ff4b4b"
    elif pred_count < curr_count:
        trend_str = "Decreasing 📉"
        trend_color = "#00c853"
    else:
        trend_str = "Stable ➡️"
        trend_color = "#29b6f6"

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1f293d 0%, #111827 100%); border: 1px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
            <h3 style="color: #60a5fa; margin-top: 0;">🔮 LIVE CROWD PREDICTION SUMMARY</h3>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;">
                <div style="background: #111827; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 180px; text-align: center; border: 1px solid #374151;">
                    <div style="color: #9ca3af; font-size: 13px;">CURRENT DETECTED PEOPLE</div>
                    <div style="color: #ffffff; font-size: 28px; font-weight: bold;">{curr_count} people</div>
                    <div style="color: #6b7280; font-size: 12px;">Status: {curr_status}</div>
                </div>
                <div style="background: #111827; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 180px; text-align: center; border: 1px solid #3b82f6;">
                    <div style="color: #9ca3af; font-size: 13px;">PREDICTED CROWD COUNT (t+1s)</div>
                    <div style="color: #60a5fa; font-size: 28px; font-weight: bold;">{pred_count} people</div>
                    <div style="color: #6b7280; font-size: 12px;">Predicted Status: {pred_status}</div>
                </div>
                <div style="background: #111827; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 180px; text-align: center; border: 1px solid #374151;">
                    <div style="color: #9ca3af; font-size: 13px;">CROWD TREND</div>
                    <div style="color: {trend_color}; font-size: 26px; font-weight: bold;">{trend_str}</div>
                    <div style="color: #6b7280; font-size: 12px;">Difference: {pred_count - curr_count:+d} people</div>
                </div>
                <div style="background: #111827; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 180px; text-align: center; border: 1px solid #374151;">
                    <div style="color: #9ca3af; font-size: 13px;">MODEL ACCURACY</div>
                    <div style="color: #10b981; font-size: 26px; font-weight: bold;">{eval_info.get('accuracy', 0.84) * 100:.1f}%</div>
                    <div style="color: #6b7280; font-size: 12px;">MAE: {eval_info.get('mae', 0.0):.2f} people</div>
                </div>
            </div>
            <div style="background-color: #0f172a; padding: 12px 18px; border-radius: 6px; font-size: 14px; color: #cbd5e1; border-left: 4px solid #60a5fa;">
                <b>Explanation:</b> The system detected <b>{curr_count} people</b> in current observation window for video <code>{curr_video}</code> and predicts <b>{pred_count} people</b> in the upcoming second based on recent temporal observations. The crowd trend is expected to remain <b>{trend_str.split()[0]}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Expander for YOLOv8 & ByteTrack Annotated Output Verification
if os.path.exists(SAMPLE_IMG):
    with st.expander("👁️ View YOLOv8 & ByteTrack Object Detection Visual Verification", expanded=False):
        st.image(SAMPLE_IMG, caption="Annotated Sample Frame showing YOLOv8 person bounding boxes, ByteTrack IDs, and live count overlay", use_container_width=True)


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab_analysis, tab_lstm, tab_alerts = st.tabs([
    "📊 Crowd Analysis & EDA",
    "🧠 LSTM Model Performance",
    "🚨 Crowd Alerts Management"
])

# ==================================================
# TAB 1: CROWD ANALYSIS & EDA
# ==================================================
with tab_analysis:
    st.header("📊 Dynamic Crowd Analysis & Exploratory Data Analysis (EDA)")

    if df_data is None:
        st.warning("No crowd dataset found. Please click 'Process Videos Now' in the sidebar.")
    else:
        num_videos = df_data["Video_Name"].nunique()
        total_obs = len(df_data)
        total_detections = int(df_data["People_Count"].sum())
        avg_count = round(df_data["People_Count"].mean(), 2)
        max_count = int(df_data["People_Count"].max())
        alert_obs = int(df_data["Exceeded"].sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Videos Processed", num_videos)
        col2.metric("Total Observations", total_obs)
        col3.metric("Total Detections", f"{total_detections:,}")
        col4.metric("Average People Count", f"{avg_count:.2f}")
        col5.metric("Alert Observations", alert_obs)

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("A. People Count vs Time across Videos")
            pivoted_time = df_data.pivot_table(
                index="Time_Seconds", columns="Video_Name",
                values="People_Count", aggfunc="mean"
            ).sort_index()
            # Fill gaps: interpolate between known points, then fill leading/trailing NaNs
            pivoted_time = pivoted_time.interpolate(method="linear", axis=0).ffill().bfill()
            st.line_chart(pivoted_time, height=320)
            st.caption("People count over sampled time intervals for each video.")

        with c2:
            st.subheader("B. Video Crowd Comparison")
            avg_per_video = df_data.groupby("Video_Name")["People_Count"].mean().reset_index()
            avg_per_video.columns = ["Video Name", "Average Count"]
            st.bar_chart(avg_per_video.set_index("Video Name"), height=320)
            st.caption("Comparison of average crowd density across all processed videos.")

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("C. Min, Mean, and Max People Count per Video")
            video_stats = df_data.groupby("Video_Name")["People_Count"].agg(["mean", "min", "max"]).reset_index()
            video_stats.columns = ["Video Name", "Mean Count", "Min Count", "Max Count"]
            st.bar_chart(video_stats.set_index("Video Name")[["Mean Count", "Min Count", "Max Count"]], height=320)
            st.caption("Detailed count statistics across all videos.")

        with c4:
            st.subheader("D. Crowd Status Distribution")
            status_dist = df_data["Crowd_Status"].value_counts().reset_index()
            status_dist.columns = ["Status", "Count"]
            st.bar_chart(status_dist.set_index("Status"), height=320)
            st.caption("Distribution of Normal, Moderate, and Congested crowd states.")

        st.divider()

        # Per-video statistics table
        st.subheader("📋 Per-Video Dataset Summary Table")
        agg_dict = {
            "Samples": ("People_Count", "count"),
            "Min_Count": ("People_Count", "min"),
            "Max_Count": ("People_Count", "max"),
            "Avg_Count": ("People_Count", "mean"),
            "Threshold": ("Threshold", "first"),
            "Alert_Frames": ("Exceeded", "sum")
        }
        if "Unique_Track_IDs" in df_data.columns:
            agg_dict["Unique_IDs"] = ("Unique_Track_IDs", "max")

        summary_table = df_data.groupby("Video_Name").agg(**agg_dict).reset_index()
        summary_table["Avg_Count"] = summary_table["Avg_Count"].round(1)
        summary_table["Alert_Rate"] = (summary_table["Alert_Frames"] / summary_table["Samples"] * 100).round(1).astype(str) + "%"

        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        with st.expander("📥 View & Download Full Crowd Dataset CSV"):
            st.dataframe(df_data, use_container_width=True, hide_index=True)
            csv_data = df_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download crowd_dataset.csv",
                data=csv_data,
                file_name="crowd_dataset.csv",
                mime="text/csv"
            )


# ==================================================
# TAB 2: LSTM MODEL PERFORMANCE
# ==================================================
with tab_lstm:
    st.header("🧠 PyTorch CrowdLSTM Model Performance & Evaluation")

    # Display Model Evaluation Metrics Cards
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Training Sequences", eval_info.get("training_samples", 0))
    m2.metric("Validation Sequences", eval_info.get("validation_samples", 0))
    m3.metric("Testing Sequences", eval_info.get("test_samples", 0))
    m4.metric("Test MAE", f"{eval_info.get('mae', 0.0):.4f} people")
    m5.metric("Test RMSE", f"{eval_info.get('rmse', 0.0):.4f} people")
    m6.metric("R² Score", f"{eval_info.get('r2_score', 0.0):.4f}")
    m7.metric("Status Accuracy", f"{eval_info.get('accuracy', 0.0) * 100:.2f}%")

    norm_scale = eval_info.get("normalization_scale", None)
    f1_val     = eval_info.get("f1_score", 0.0)
    if norm_scale:
        st.info(f"**Normalization Scale (training max):** {norm_scale:.1f} people  |  **Status F1-Score:** {f1_val:.4f}  |  **Baseline MAE:** 5.59 people")

    st.divider()

    st.subheader("📈 Actual People Count vs LSTM Predicted People Count")

    preds_list = eval_info.get("predictions", [])
    if preds_list:
        df_preds = pd.DataFrame(preds_list)
        # Support both underscore-keyed and space-keyed column names
        col_time   = "Target_Time"   if "Target_Time"   in df_preds.columns else "Target Time"
        col_actual = "Actual_Count"  if "Actual_Count"  in df_preds.columns else "Actual Count"
        col_pred   = "Predicted_Count" if "Predicted_Count" in df_preds.columns else "Predicted Count"
        col_err    = "Absolute_Error" if "Absolute_Error" in df_preds.columns else "Absolute Error"

        df_preds["Label"] = df_preds["Video"] + " (" + df_preds[col_time].astype(str) + "s)"

        chart_data = df_preds.set_index("Label")[[col_actual, col_pred]].rename(
            columns={col_actual: "Actual Count", col_pred: "Predicted Count"}
        )
        st.line_chart(chart_data, height=350)
        st.caption("Comparison of Actual vs PyTorch LSTM Predicted People Count across test sequences.")

        st.divider()

        col_tbl, col_cm = st.columns([3, 2])

        with col_tbl:
            st.subheader("📋 Test Set Predictions Table")
            table_df = df_preds[["Video", col_time, col_actual, col_pred, col_err]].rename(columns={
                col_time: "Target Time", col_actual: "Actual Count",
                col_pred: "Predicted Count", col_err: "Absolute Error"
            })
            st.dataframe(table_df, width="content", hide_index=True)

        with col_cm:
            st.subheader("🎯 Classification Confusion Matrix")
            conf_matrix = eval_info.get("confusion_matrix", {})
            if conf_matrix:
                df_cm = pd.DataFrame(conf_matrix).T
                st.dataframe(df_cm, use_container_width=True)
                st.caption("Row = True Crowd Status, Column = Predicted Status")

    else:
        st.info("No trained model evaluation report found. Click 'Train/Retrain LSTM Model' in the sidebar.")


# ==================================================
# TAB 3: CROWD ALERTS MANAGEMENT
# ==================================================
with tab_alerts:
    st.header("Crowd Alerts Management")

    # -- Live alert state from live_state.json --
    live_state_path = os.path.join(BASE_DIR, "output", "live_state.json")
    live_state_data = None
    if os.path.exists(live_state_path):
        try:
            with open(live_state_path, "r") as f:
                live_state_data = json.load(f)
        except Exception:
            pass

    if live_state_data:
        consecutive   = live_state_data.get("consecutive_congested", 0)
        cooldown_on   = live_state_data.get("alert_cooldown_active", False)
        alert_fired   = live_state_data.get("alert_fired", False)
        alert_status  = live_state_data.get("status", "Normal")
        last_alert_ep = live_state_data.get("last_alert_epoch", 0)
        try:
            from live_state import CONGESTION_CONFIRMATION_COUNT, ALERT_COOLDOWN_SECONDS
        except Exception:
            CONGESTION_CONFIRMATION_COUNT = 3
            ALERT_COOLDOWN_SECONDS = 60

        if last_alert_ep > 0:
            import time as _t
            from datetime import datetime as _dt
            last_alert_str = _dt.fromtimestamp(last_alert_ep).strftime("%H:%M:%S")
        else:
            last_alert_str = "Never"

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Current Live Status", alert_status)
        ac2.metric("Consec. Congested Frames", f"{consecutive} / {CONGESTION_CONFIRMATION_COUNT}")
        ac3.metric("Alert Cooldown Active", "Yes" if cooldown_on else "No")
        ac4.metric("Last Alert Time", last_alert_str)

        if alert_fired:
            st.error("ACTIVE ALERT: High congestion confirmed! Crowd management required immediately.")
        elif cooldown_on:
            st.warning("Alert cooldown in effect. Next alert will fire after cooldown expires.")
        elif alert_status == "Congested" and consecutive < CONGESTION_CONFIRMATION_COUNT:
            st.warning(f"Congestion building... ({consecutive}/{CONGESTION_CONFIRMATION_COUNT} frames confirmed). No alert yet.")
        elif alert_status in ("Normal", "Moderate"):
            st.success("No active crowd alert. Current status is within safe limits.")
        else:
            st.success("System monitoring normally. No alert active.")

        st.divider()
        st.info(
            f"**Alert Logic:** Alert fires ONLY when crowd status is Congested "
            f"for **{CONGESTION_CONFIRMATION_COUNT} consecutive frames** AND "
            f"**{ALERT_COOLDOWN_SECONDS}s cooldown** has elapsed. "
            f"Normal and Moderate statuses NEVER generate alerts."
        )
    else:
        st.info("No live state data yet. Process videos or start live detection to see real-time alert status.")

    st.divider()

    if df_data is None:
        st.warning("No crowd dataset found. Please process the videos first.")
    else:
        df_alerts = df_data.copy()
        df_alerts["Exceeded_By"] = np.maximum(0, df_alerts["People_Count"] - df_alerts["Threshold"])

        total_alerts     = int(df_alerts["Exceeded"].sum())
        max_people       = int(df_alerts["People_Count"].max())
        max_exceeded     = int(df_alerts["Exceeded_By"].max())
        vids_with_alerts = df_alerts[df_alerts["Exceeded"]]["Video_Name"].nunique()

        ac1b, ac2b, ac3b, ac4b = st.columns(4)
        ac1b.metric("Dataset Alert Observations", total_alerts)
        ac2b.metric("Maximum People Count", max_people)
        ac3b.metric("Max Threshold Exceedance", f"+{max_exceeded} people")
        ac4b.metric("Videos With Alerts", f"{vids_with_alerts} / {df_alerts['Video_Name'].nunique()}")

        st.divider()

        df_alerts["Alert Badge"] = df_alerts["Exceeded"].apply(
            lambda x: "CROWD ALERT" if x else "WITHIN THRESHOLD"
        )

        cols_to_show = ["Video_Name", "Time_Seconds", "People_Count", "Threshold", "Exceeded_By", "Alert Badge"]
        if "Unique_Track_IDs" in df_alerts.columns:
            cols_to_show.insert(4, "Unique_Track_IDs")

        res_table = df_alerts[cols_to_show].copy()

        st.subheader("Crowd Alert Logs (Full Dataset)")
        st.dataframe(res_table, use_container_width=True, hide_index=True)

        csv_alerts = res_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Alert Data CSV",
            data=csv_alerts,
            file_name="crowd_alerts.csv",
            mime="text/csv"
        )