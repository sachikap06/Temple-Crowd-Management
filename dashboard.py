import streamlit as st
import pandas as pd
import os




# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Temple Crowd Management",
    page_icon="🛕",
    layout="wide"
)

st.title("🛕 Smart Temple Crowd Management")
st.caption("Staff Monitoring Dashboard")

# --------------------------------------------------
# LIVE DASHBOARD
# --------------------------------------------------

@st.fragment(run_every="1s")
def live_dashboard():

    csv_file = "output/crowd_data.csv"

    # Check CSV
    if not os.path.exists(csv_file):
        st.error("crowd_data.csv not found.")
        return

    # Read CSV
    df = pd.read_csv(csv_file)

    if df.empty:
        st.warning("Waiting for crowd data...")
        return

    # --------------------------------------------------
    # LATEST DATA
    # --------------------------------------------------

    latest = df.iloc[-1]

    people_count = int(latest["People_Count"])
    timestamp = latest["Timestamp"]

    # Crowd status
    if people_count <= 5:
        status = "Normal"
    elif people_count <= 15:
        status = "Moderate"
    else:
        status = "Congested"

    # --------------------------------------------------
    # TOP INFORMATION
    # --------------------------------------------------

    st.info("📷 Camera: Main Gate")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👥 LIVE COUNT",
            f"{people_count} People"
        )

    with col2:
        st.metric(
            "🔮 PREDICTED COUNT",
            "Coming Soon"
        )

    with col3:
        st.metric(
            "🚦 CROWD STATUS",
            status
        )

    # --------------------------------------------------
    # LIVE CROWD GRAPH
    # --------------------------------------------------

    st.subheader("📈 Live Crowd Trend")

    # Show only the latest 60 records
    recent_data = df.tail(60).copy()

    # Use only required columns
    recent_data = recent_data[
        ["Timestamp", "People_Count"]
    ]

    # Set timestamp as index
    recent_data = recent_data.set_index("Timestamp")

    # Display graph
    st.line_chart(
        recent_data,
        height=350
    )

    st.caption("Showing the latest 60 seconds of crowd activity.")

    # --------------------------------------------------
    # LAST UPDATED
    # --------------------------------------------------

    st.write(
        f"🕐 Last Updated: {timestamp}"
    )


# Run dashboard
live_dashboard()