import streamlit as st
import pandas as pd
import numpy as np
import os
from lstm_model import predict_future_crowd, load_trained_model

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Temple Crowd Management",
    page_icon="🛕",
    layout="wide"
)

st.title("🛕 Smart Temple Crowd Management")
st.caption("Staff Real-Time Monitoring & PyTorch LSTM Crowd Forecasting Dashboard")


# --------------------------------------------------
# MODEL LOADING CACHE & SIDEBAR CONTROLS
# --------------------------------------------------

@st.cache_resource
def get_lstm_model():
    return load_trained_model()

# Sidebar Control Center
st.sidebar.title("🛕 Control Center")

stop_signal_file = "output/stop_camera.signal"

if os.path.exists(stop_signal_file):
    st.sidebar.warning("🛑 Camera Status: STOPPED")
    st.sidebar.caption("To restart camera, run `python detect.py` in terminal.")
else:
    st.sidebar.success("🟢 Camera Status: ACTIVE")
    if st.sidebar.button("🛑 Stop Camera", use_container_width=True):
        with open(stop_signal_file, "w") as f:
            f.write("stop")
        st.sidebar.warning("Camera stopping...")
        st.rerun()


# --------------------------------------------------
# LIVE DASHBOARD
# --------------------------------------------------

@st.fragment(run_every="1s")
def live_dashboard():

    csv_file = "output/crowd_data.csv"

    # Check CSV
    if not os.path.exists(csv_file):
        st.error("crowd_data.csv not found. Please start detect.py camera script.")
        return

    # Read CSV
    df = pd.read_csv(csv_file)

    if df.empty:
        st.warning("Waiting for crowd data...")
        return

    # --------------------------------------------------
    # LATEST DATA & LSTM PREDICTION
    # --------------------------------------------------

    latest = df.iloc[-1]
    people_count = int(latest["People_Count"])
    timestamp = latest["Timestamp"]

    # Run PyTorch LSTM Inference
    lstm_model = get_lstm_model()
    history_counts = df["People_Count"].values
    predicted_count = predict_future_crowd(history_counts, model=lstm_model)

    # Crowd status
    if people_count <= 5:
        status = "Normal"
    elif people_count <= 15:
        status = "Moderate"
    else:
        status = "Congested"

    # Calculate prediction delta
    delta = predicted_count - people_count
    delta_text = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "No change")

    # --------------------------------------------------
    # LIVE CAMERA FEED & TOP METRICS
    # --------------------------------------------------

    col_vid, col_metrics = st.columns([1, 1])

    with col_vid:
        st.subheader("📷 Live Camera Feed")
        live_frame_path = "output/live_frame.jpg"
        if os.path.exists(live_frame_path):
            st.image(live_frame_path, use_container_width=True)
        else:
            st.info("Waiting for camera feed...")

    with col_metrics:
        st.subheader("📊 Real-Time Analytics")
        st.info("📷 Camera: Main Gate")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "👥 LIVE COUNT",
                f"{people_count} People"
            )

        with col2:
            st.metric(
                "🔮 LSTM FORECAST (10s)",
                f"{predicted_count} People",
                delta=f"{delta_text} in 10s"
            )

        with col3:
            st.metric(
                "🚦 CROWD STATUS",
                status
            )

    # --------------------------------------------------
    # LIVE CROWD GRAPH WITH LSTM FORECAST
    # --------------------------------------------------

    st.subheader("📈 Live Crowd Trend vs PyTorch LSTM Forecast")

    recent_data = df.tail(60).copy()
    
    # Calculate rolling LSTM predictions for the trend graph
    counts_array = recent_data["People_Count"].values
    lstm_preds = []
    
    for i in range(len(counts_array)):
        sub_seq = counts_array[: i + 1]
        pred = predict_future_crowd(sub_seq, model=lstm_model)
        lstm_preds.append(pred)
        
    recent_data["LSTM Forecast"] = lstm_preds
    recent_data = recent_data.rename(columns={"People_Count": "Actual Live Count"})
    recent_data = recent_data.set_index("Timestamp")[["Actual Live Count", "LSTM Forecast"]]

    # Display graph
    st.line_chart(
        recent_data,
        height=350
    )

    st.caption("Comparing actual live crowd count with PyTorch LSTM neural network forecast.")

    # --------------------------------------------------
    # LAST UPDATED
    # --------------------------------------------------

    st.write(
        f"🕐 Last Updated: {timestamp} | PyTorch LSTM Active"
    )


# Run dashboard
live_dashboard()