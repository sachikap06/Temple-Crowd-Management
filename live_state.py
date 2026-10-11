import os
import json
import time
from datetime import datetime
from lstm_model import predict_future_crowd, load_trained_model

LIVE_STATE_FILE = os.path.join("output", "live_state.json")
MAX_HISTORY_LEN = 30

# ------------------------------------------------------------------
# CONGESTION CONFIRMATION & ALERT COOLDOWN SETTINGS
# Alert fires ONLY for "Congested" status, ONLY after
# CONGESTION_CONFIRMATION_COUNT consecutive congested readings,
# and will NOT re-fire during ALERT_COOLDOWN_SECONDS.
# ------------------------------------------------------------------
CONGESTION_CONFIRMATION_COUNT = 3     # consecutive congested frames before alert
ALERT_COOLDOWN_SECONDS        = 60    # minimum seconds between repeated alerts

# In-memory counters (persist across calls within the same process run)
_consecutive_congested = 0
_last_alert_epoch      = 0.0


def update_live_crowd_state(
    current_count: int,
    threshold: int,
    video_name: str = "Live Stream",
    history_counts: list = None,
):
    """
    Called by process_videos.py or detect.py whenever a new YOLO count is obtained.
    Updates output/live_state.json atomically with change detection & rolling history.

    Alert logic:
      - Only fires for "Congested" status.
      - Requires CONGESTION_CONFIRMATION_COUNT consecutive congested readings.
      - Respects ALERT_COOLDOWN_SECONDS between repeated alerts.
      - Normal / Moderate NEVER trigger an alert.
    """
    global _consecutive_congested, _last_alert_epoch

    now_str   = datetime.now().strftime("%H:%M:%S")
    now_epoch = time.time()

    # Load existing state to get previous count & history
    prev_state = {}
    if os.path.exists(LIVE_STATE_FILE):
        try:
            with open(LIVE_STATE_FILE, "r", encoding="utf-8") as f:
                prev_state = json.load(f)
        except Exception:
            pass

    previous_count = prev_state.get("current_crowd", current_count)
    history_list   = prev_state.get("history", [])

    # Restore in-memory counters from saved state when process restarts
    if _consecutive_congested == 0 and _last_alert_epoch == 0.0:
        _consecutive_congested = prev_state.get("_consecutive_congested", 0)
        _last_alert_epoch      = prev_state.get("_last_alert_epoch", 0.0)

    diff = current_count - previous_count
    if diff > 0:
        change_type = "Increased"
        change_str  = f"+{diff}"
    elif diff < 0:
        change_type = "Decreased"
        change_str  = str(diff)
    else:
        change_type = "Unchanged"
        change_str  = "0"

    # Crowd Status
    normal_limit = max(1, threshold // 3)
    if current_count <= normal_limit:
        status = "Normal"
    elif current_count <= threshold:
        status = "Moderate"
    else:
        status = "Congested"

    # ------------------------------------------------------------------
    # CONGESTION CONFIRMATION COUNTER
    # ------------------------------------------------------------------
    if status == "Congested":
        _consecutive_congested += 1
    else:
        _consecutive_congested = 0   # reset on any non-congested frame

    # ------------------------------------------------------------------
    # ALERT DECISION
    #   - Only trigger for Congested
    #   - Only after CONGESTION_CONFIRMATION_COUNT consecutive frames
    #   - Only if cooldown has elapsed since last alert
    # ------------------------------------------------------------------
    cooldown_elapsed    = (now_epoch - _last_alert_epoch) >= ALERT_COOLDOWN_SECONDS
    confirmed_congested = _consecutive_congested >= CONGESTION_CONFIRMATION_COUNT

    alert_fired           = False
    alert_message         = ""
    alert_cooldown_active = (not cooldown_elapsed) and (_last_alert_epoch > 0.0)

    if status == "Congested" and confirmed_congested and cooldown_elapsed:
        alert_fired       = True
        _last_alert_epoch = now_epoch
        alert_message     = (
            "Temple Crowd Alert: High congestion detected. "
            "Immediate crowd-management attention is required."
        )

    # LSTM prediction
    if history_counts and len(history_counts) > 0:
        counts_for_lstm = history_counts
    else:
        counts_for_lstm = [item["curr"] for item in history_list] + [current_count]

    lstm_model      = load_trained_model()
    predicted_count = predict_future_crowd(counts_for_lstm, model=lstm_model)

    # Rolling history
    new_entry = {
        "time":        now_str,
        "prev":        previous_count,
        "curr":        current_count,
        "change_str":  change_str,
        "change_type": change_type,
    }

    should_add = True
    if (
        history_list
        and history_list[-1]["curr"] == current_count
        and history_list[-1]["prev"] == previous_count
    ):
        should_add = False

    if should_add or not history_list:
        history_list.append(new_entry)
        if len(history_list) > MAX_HISTORY_LEN:
            history_list = history_list[-MAX_HISTORY_LEN:]

    state_data = {
        "current_crowd":          current_count,
        "previous_crowd":         previous_count,
        "change":                 diff,
        "change_str":             change_str,
        "change_type":            change_type,
        "status":                 status,
        "predicted_crowd":        predicted_count,
        "threshold":              threshold,
        "video_name":             video_name,
        "timestamp":              now_str,
        "last_updated_epoch":     now_epoch,
        "consecutive_congested":  _consecutive_congested,
        "alert_fired":            alert_fired,
        "alert_message":          alert_message,
        "alert_cooldown_active":  alert_cooldown_active,
        "last_alert_epoch":       _last_alert_epoch,
        # Internal counters persisted so they survive process restarts
        "_consecutive_congested": _consecutive_congested,
        "_last_alert_epoch":      _last_alert_epoch,
        "history":                history_list,
    }

    # Atomic write
    os.makedirs("output", exist_ok=True)
    temp_file = LIVE_STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    os.replace(temp_file, LIVE_STATE_FILE)

    return state_data


def read_live_crowd_state():
    """Reads the shared live crowd state from output/live_state.json."""
    if os.path.exists(LIVE_STATE_FILE):
        try:
            with open(LIVE_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None
