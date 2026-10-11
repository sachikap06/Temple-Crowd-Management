import os
import sys
import time
import json
import socket
import io
import base64
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer
except ImportError:
    ThreadingHTTPServer = HTTPServer

import pandas as pd
import numpy as np

# Ensure working directory is always project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from lstm_model import predict_future_crowd, load_trained_model

DEFAULT_PORT = 8502
DATASET_CSV = os.path.join("output", "crowd_dataset.csv")
CONFIG_JSON = os.path.join("output", "live_config.json")

# Global lazy-loaded LSTM model cache
_cached_lstm_model = None
_model_lock = threading.Lock()

def get_cached_lstm_model():
    global _cached_lstm_model
    if _cached_lstm_model is None:
        with _model_lock:
            if _cached_lstm_model is None:
                try:
                    _cached_lstm_model = load_trained_model()
                except Exception as e:
                    safe_print(f"[WARNING] Model load warning: {e}")
                    _cached_lstm_model = None
    return _cached_lstm_model

# Dataset cache to avoid reading CSV on every 1-second poll
_dataset_cache = {
    "mtime": 0,
    "current_crowd": 0,
    "history_counts": []
}

# Server state container
_server_state = {
    "running": False,
    "host": "0.0.0.0",
    "port": DEFAULT_PORT,
    "local_ip": "127.0.0.1",
    "lock": threading.Lock(),
    "last_alert_state": "NONE",  # "NONE", "EARLY", "ACTUAL"
    "last_alert_id": None,
    "last_alert_title": "ℹ️ System Status",
    "last_alert_msg": "Crowd monitoring active.",
    "last_alert_time": None
}

def safe_print(*args, **kwargs):
    """Safely print strings even if console encoding does not support unicode emojis."""
    msg = " ".join(map(str, args))
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
        print(safe_msg, **kwargs)

def get_local_ip():
    """Dynamically determine the laptop's LAN IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def is_port_in_use(port=DEFAULT_PORT):
    """Check if the specified port is already open and listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False

def get_configured_threshold():
    """Retrieve threshold from live_config.json or dataset default."""
    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, "r") as f:
                cfg = json.load(f)
                val = cfg.get("threshold", 20)
                if val is not None:
                    return int(val)
        except Exception:
            pass
    if os.path.exists(DATASET_CSV):
        try:
            df = pd.read_csv(DATASET_CSV)
            if not df.empty and "Threshold" in df.columns:
                valid_thresh = pd.to_numeric(df["Threshold"], errors='coerce').dropna()
                if not valid_thresh.empty:
                    return int(valid_thresh.iloc[-1])
        except Exception:
            pass
    return 20

def generate_qr_code_base64(url):
    """Generate QR code base64 PNG image for the given URL."""
    if not url:
        return ""
    try:
        import qrcode
        img = qrcode.make(str(url))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        safe_print(f"[WARNING] QR code generation notice: {e}")
        return ""

def get_fallback_dataset_info():
    """Cached reading of dataset CSV to avoid heavy disk I/O on every polling request."""
    global _dataset_cache
    if os.path.exists(DATASET_CSV):
        try:
            mtime = os.path.getmtime(DATASET_CSV)
            if mtime != _dataset_cache["mtime"]:
                df = pd.read_csv(DATASET_CSV)
                if not df.empty and "People_Count" in df.columns:
                    valid_counts = pd.to_numeric(df["People_Count"], errors='coerce').dropna().astype(int).tolist()
                    if valid_counts:
                        _dataset_cache["mtime"] = mtime
                        _dataset_cache["history_counts"] = valid_counts
                        _dataset_cache["current_crowd"] = valid_counts[-1]
        except Exception:
            pass
    return _dataset_cache["current_crowd"], _dataset_cache["history_counts"]

def trigger_test_alert():
    """Trigger a simulated threshold surge alert to test phone notifications."""
    with _server_state["lock"]:
        now_str = datetime.now().strftime("%H:%M:%S")
        _server_state["last_alert_id"] = int(time.time() * 1000)
        _server_state["last_alert_state"] = "ACTUAL"
        _server_state["last_alert_title"] = "🚨 TEST CROWD SURGE ALERT"
        _server_state["last_alert_msg"] = "Simulated Warning: Crowd threshold exceeded! Check mobile alert notifications."
        _server_state["last_alert_time"] = now_str
        safe_print(f"[TEST ALERT] Triggered test notification alert at {now_str}")

def evaluate_alert_status():
    """
    Evaluates current crowd count, change metrics, & LSTM prediction to determine alert status.
    Uses real-time live state generated during video processing or live camera streams.
    Implements debouncing to prevent notification spam.
    """
    try:
        from live_state import read_live_crowd_state
        live_data = read_live_crowd_state()
    except Exception:
        live_data = None

    current_time_str = datetime.now().strftime("%H:%M:%S")
    now_epoch = time.time()

    is_live = False
    if live_data and "last_updated_epoch" in live_data:
        try:
            if (now_epoch - float(live_data["last_updated_epoch"])) <= 12.0:
                is_live = True
        except (ValueError, TypeError):
            is_live = False

    if live_data:
        current_crowd = live_data.get("current_crowd", 0)
        previous_crowd = live_data.get("previous_crowd", current_crowd)
        change = live_data.get("change", 0)
        change_str = str(live_data.get("change_str", "0"))
        change_type = live_data.get("change_type", "Unchanged")
        crowd_status = live_data.get("status", "Normal")
        predicted_crowd = live_data.get("predicted_crowd", current_crowd)
        threshold = live_data.get("threshold", get_configured_threshold())
        timestamp = live_data.get("timestamp", current_time_str)
        video_name = live_data.get("video_name", "Video Stream")
        history = live_data.get("history", [])
    else:
        current_crowd, history_counts = get_fallback_dataset_info()
        threshold = get_configured_threshold()
        try:
            lstm_model = get_cached_lstm_model()
            predicted_crowd = predict_future_crowd(history_counts, model=lstm_model) if history_counts else current_crowd
        except Exception:
            predicted_crowd = current_crowd

        normal_limit = max(1, threshold // 3)
        if current_crowd <= normal_limit:
            crowd_status = "Normal"
        elif current_crowd <= threshold:
            crowd_status = "Moderate"
        else:
            crowd_status = "Congested"
        previous_crowd = current_crowd
        change = 0
        change_str = "0"
        change_type = "Unchanged"
        timestamp = current_time_str
        video_name = "Video Stream"
        history = [{"time": current_time_str, "prev": current_crowd, "curr": current_crowd, "change_str": "0", "change_type": "Unchanged"}]

    # ------------------------------------------------------------------
    # ALERT STATE (FIXED): Only ACTUAL alert, only for confirmed Congested.
    # Reads the alert_fired flag set by live_state.py after
    # CONGESTION_CONFIRMATION_COUNT consecutive congested readings
    # AND ALERT_COOLDOWN_SECONDS has elapsed. Normal/Moderate never alert.
    # ------------------------------------------------------------------
    # Read confirmed alert flags from live_state.json
    alert_fired           = live_data.get("alert_fired", False) if live_data else False
    alert_cooldown_active = live_data.get("alert_cooldown_active", False) if live_data else False
    consecutive_congested = live_data.get("consecutive_congested", 0) if live_data else 0
    live_alert_message    = live_data.get("alert_message", "") if live_data else ""

    new_alert_state = "NONE"
    alert_title     = "\u2139\ufe0f System Status"
    alert_msg       = "Crowd levels are currently within safe limits."

    if alert_fired:
        new_alert_state = "ACTUAL"
        alert_title     = "\U0001f6a8 CROWD ALERT"
        alert_msg       = (
            live_alert_message
            if live_alert_message
            else f"Temple Crowd Alert: High congestion detected. Immediate crowd-management attention is required."
        )

    with _server_state["lock"]:
        prev_state = _server_state["last_alert_state"]

        # Debouncing: Generate new alert ID only on transition INTO an alert
        if new_alert_state != prev_state and new_alert_state != "NONE":
            _server_state["last_alert_id"]    = int(time.time() * 1000)
            _server_state["last_alert_time"]  = timestamp
            _server_state["last_alert_title"] = alert_title
            _server_state["last_alert_msg"]   = alert_msg

        _server_state["last_alert_state"] = new_alert_state

        return {
            "connected":             True,
            "is_live":               is_live,
            "current_crowd":         current_crowd,
            "previous_crowd":        previous_crowd,
            "change":                change,
            "change_str":            change_str,
            "change_type":           change_type,
            "crowd_status":          crowd_status,
            "predicted_crowd":       predicted_crowd,
            "threshold":             threshold,
            "video_name":            video_name,
            "alert_type":            new_alert_state,
            "alert_title":           _server_state["last_alert_title"] if new_alert_state != "NONE" else alert_title,
            "alert_message":         _server_state["last_alert_msg"]   if new_alert_state != "NONE" else alert_msg,
            "alert_id":              _server_state["last_alert_id"]    if new_alert_state != "NONE" else None,
            "last_alert_time":       _server_state["last_alert_time"],
            "consecutive_congested": consecutive_congested,
            "alert_cooldown_active": alert_cooldown_active,
            "timestamp":             timestamp,
            "history":               history,
        }

class MobileHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence HTTP logs in console
        return

    def do_GET(self):
        try:
            if self.path == "/api/status":
                data = evaluate_alert_status()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            elif self.path in ("/", "/index.html", "/mobile") or self.path.startswith("/mobile") or self.path.startswith("/index.html"):
                html_file = os.path.join(BASE_DIR, "mobile_page.html")
                if os.path.exists(html_file):
                    with open(html_file, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_error(404, "mobile_page.html not found")
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            except Exception:
                pass

def start_mobile_server_thread(port=DEFAULT_PORT):
    local_ip = get_local_ip()
    _server_state["local_ip"] = local_ip
    _server_state["port"] = port

    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), MobileHTTPRequestHandler)
        _server_state["running"] = True
        safe_print(f"[OK] Mobile Notification Server listening on http://{local_ip}:{port}")
        httpd.serve_forever()
    except Exception as e:
        if is_port_in_use(port):
            safe_print(f"[OK] Mobile Notification Server already active on port {port}")
            _server_state["running"] = True
        else:
            safe_print(f"[WARNING] Mobile server notice on port {port}: {e}")
            _server_state["running"] = False

def ensure_mobile_server_running(port=DEFAULT_PORT):
    """Ensure mobile HTTP server runs in background daemon thread."""
    if is_port_in_use(port):
        _server_state["running"] = True
        return get_local_ip(), port

    if not _server_state["running"]:
        t = threading.Thread(target=start_mobile_server_thread, args=(port,), daemon=True)
        t.start()
        time.sleep(0.3)
    return get_local_ip(), port

if __name__ == "__main__":
    ip = get_local_ip()
    safe_print("=" * 60)
    safe_print("  Smart Temple Mobile Notification Server")
    safe_print(f"  Mobile Connection URL: http://{ip}:{DEFAULT_PORT}")
    safe_print("=" * 60)
    start_mobile_server_thread(DEFAULT_PORT)


