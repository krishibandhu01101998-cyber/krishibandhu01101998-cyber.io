# predict_all_from_tb_and_push_loop_ESC.py
# Reads actual data from ThingSpeak, runs predictions using local LSTM models,
# posts telemetry to ThingsBoard, and pushes actuals/predictions to ThingSpeak channels.
# Robust ThingSpeak push helper with retries and rate-limit guard.

import os
import json
import pickle
import warnings
import sys
import time
import platform
import select
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

warnings.filterwarnings("ignore")
print("TensorFlow:", tf.__version__)

# ========= THINGSBOARD CONFIG (write-only) =========
TB_HOST         = "https://demo.thingsboard.io"
TB_DEVICE_ID    = "e55ec190-87b8-11f0-a9b5-792e2194a5d4"
TB_DEVICE_TOKEN = "zexDc8u7iSDmmYU587xU"   # used for POSTing telemetry
TB_TELEMETRY_POST = f"{TB_HOST}/api/v1/{TB_DEVICE_TOKEN}/telemetry"

# (JWT not used for reads because reads come from ThingSpeak now)
TB_JWT = "Bearer UNUSED_WHEN_READING_FROM_THINGSPEAK"

# Telemetry tags used by the model
TB_KEY_MAP = {
    "temperature":   "temperature",
    "humidity":      "humidity",
    "rainMmHr":      "rainMmHr",
    "soilMoisture":  "soilMoisture",
}
TB_PRED_KEY_MAP = {
    "temperature":   "temperaturePred",
    "humidity":      "humidityPred",
    "rainMmHr":      "rainMmHrPred",
    "soilMoisture":  "soilMoisturePred",
}

# ========= THINGSPEAK CONFIG (READ + WRITE) =========
THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update"
THINGSPEAK_FEEDS_URL  = "https://api.thingspeak.com/channels/{channel_id}/feeds.json"

# Actual (sensor) channel (channel id 3007544)
THINGSPEAK_CHANNEL_ACTUAL_ID = 3007544
THINGSPEAK_WRITE_KEY_ACTUAL   = "DO4EA719M1UTONRV"   # write key (actual)
THINGSPEAK_READ_KEY_ACTUAL    = "M5Y79EDJWV6KETHL"   # read key (actual)

# Prediction channel (channel id 3068374)
THINGSPEAK_CHANNEL_PRED_ID    = 3068374
THINGSPEAK_WRITE_KEY_PRED     = "8BH0E3HXXDZMI330"   # write key (pred)
THINGSPEAK_READ_KEY_PRED      = "56K9Z12DA5EK1YEM"   # read key (pred)

# Field mapping for channels (adjust if your channels use different field numbers)
THINGSPEAK_FIELDS_ACTUAL = {
    "temperature":  "field1",
    "humidity":     "field2",
    "rainMmHr":     "field3",
    "soilMoisture": "field4",
}
THINGSPEAK_FIELDS_PRED = {
    "temperaturePred":  "field1",
    "humidityPred":     "field2",
    "rainMmHrPred":     "field3",
    "soilMoisturePred": "field4",
}

# ThingSpeak rate-limit guard (seconds). Free channels ≈15s min.
THINGSPEAK_MIN_INTERVAL = 15.0
_last_thingspeak_post = {"actual": 0.0, "pred": 0.0}

# ========= LOOP / HISTORY SETTINGS =========
SLEEP_SEC       =  0.001   # internal loop cadence (predictions can run fast; pushes are rate-limited)
POST_ACTUALS    = True
MAX_RETRIES     = 3

# CSV export of full history
HISTORY_PAGE_LIMIT        = 10000
HISTORY_REFRESH_MINUTES   = 60     # set 0 to export only at startup
HISTORY_START_TS_MS       = 0
EXPORT_ON_START           = True

# Fallback seeds if nothing available at boot
DEFAULT_FALLBACKS = {
    "temperature":  28.0,
    "humidity":     60.0,
    "rainMmHr":      0.0,
    "soilMoisture": 40.0,
}
GLOBAL_RANGES = {
    "temperature":  (0.0, 55.0),
    "humidity":     (0.0, 100.0),
    "rainMmHr":     (0.0, 50.0),
    "soilMoisture": (0.0, 100.0),
}

# ========= ESC detection =========
def esc_pressed():
    if platform.system().lower().startswith("win"):
        try:
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch() == b'\x1b'
        except Exception:
            return False
        return False
    else:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1) == '\x1b'
        return False

# ========= HTTP helpers =========
def http_get_with_retry(url, headers=None, params=None, timeout=60, max_retries=MAX_RETRIES):
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err

def http_post_with_retry(url, json_payload, headers=None, timeout=30, max_retries=MAX_RETRIES):
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.post(url, json=json_payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise last_err

# ========= ThingSpeak push helpers (robust) =========
def _thingspeak_get_with_retry(params, write_key, max_retries=3, timeout=20):
    """
    Internal helper: GET /update with retries. Returns response.text on success,
    or raises an exception.
    """
    p = {"api_key": write_key}
    p.update(params)
    last_err = None
    backoff = 1.0
    for attempt in range(1, max_retries+1):
        try:
            r = requests.get(THINGSPEAK_UPDATE_URL, params=p, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} ({r.reason})")
            return r.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2.0
    raise last_err

def push_actuals_to_thingspeak(actuals_payload: dict):
    """
    Push actual sensor values to actual channel using THINGSPEAK_WRITE_KEY_ACTUAL.
    Returns entry_id string (non-zero) on success, "0" on ThingSpeak rejection, or "0" on skip.
    """
    now = time.time()
    if (now - _last_thingspeak_post["actual"]) < THINGSPEAK_MIN_INTERVAL:
        print("[TS] Skipping actuals push to avoid rate-limit.")
        return "0"

    params = {}
    for tag, val in actuals_payload.items():
        field = THINGSPEAK_FIELDS_ACTUAL.get(tag)
        if field:
            params[field] = f"{float(val):.3f}"

    if not params:
        return "0"

    try:
        entry_id = _thingspeak_get_with_retry(params, THINGSPEAK_WRITE_KEY_ACTUAL)
        if entry_id == "0":
            print(f"[TS] Actuals update rejected (0) — likely rate-limited or bad write key. params={params}")
        else:
            print(f"[TS] Actuals update OK channel={THINGSPEAK_CHANNEL_ACTUAL_ID} entry_id={entry_id} params={params}")
            _last_thingspeak_post["actual"] = now
        return entry_id
    except Exception as e:
        print(f"[TS] Actuals push error: {e} params={params}")
        return "0"


def push_preds_to_thingspeak(pred_payload: dict):
    """
    Push predicted values to pred channel using THINGSPEAK_WRITE_KEY_PRED.
    pred_payload is expected to have keys like 'temperaturePred', etc.
    Returns entry_id string (non-zero) on success, "0" on failure/skip.
    """
    now = time.time()
    if (now - _last_thingspeak_post["pred"]) < THINGSPEAK_MIN_INTERVAL:
        print("[TS] Skipping preds push to avoid rate-limit.")
        return "0"

    params = {}
    # pred_payload keys are like "temperaturePred"
    for pred_tag, field in THINGSPEAK_FIELDS_PRED.items():
        val = pred_payload.get(pred_tag)
        if val is not None:
            params[field] = f"{float(val):.3f}"

    if not params:
        return "0"

    try:
        entry_id = _thingspeak_get_with_retry(params, THINGSPEAK_WRITE_KEY_PRED)
        if entry_id == "0":
            print(f"[TS] Preds update rejected (0) — likely rate-limited or bad write key. params={params}")
        else:
            print(f"[TS] Preds update OK channel={THINGSPEAK_CHANNEL_PRED_ID} entry_id={entry_id} params={params}")
            _last_thingspeak_post["pred"] = now
        return entry_id
    except Exception as e:
        print(f"[TS] Preds push error: {e} params={params}")
        return "0"

# ========= ThingSpeak READ helpers (replace TB reads) =========
def parse_thingspeak_time(ts_str):
    # ThingSpeak returns "YYYY-MM-DDTHH:MM:SSZ" typically — parse accordingly
    if not ts_str:
        return int(time.time() * 1000)
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception:
        try:
            # fallback to fromisoformat
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return int(time.time() * 1000)

def thingspeak_feeds_url(channel_id, results=100, read_key=None):
    url = THINGSPEAK_FEEDS_URL.format(channel_id=channel_id)
    params = {"results": results}
    if read_key:
        params["api_key"] = read_key
    return url, params

def ts_fetch_last_points_for_tag(tag, count):
    """
    Fetch last `count` points for a given tag from the actual ThingSpeak channel.
    Returns (ts_list, val_list) where ts are ms since epoch.
    """
    field = THINGSPEAK_FIELDS_ACTUAL.get(tag)
    if field is None:
        return [], []
    channel = THINGSPEAK_CHANNEL_ACTUAL_ID
    read_key = THINGSPEAK_READ_KEY_ACTUAL or None
    url, params = thingspeak_feeds_url(channel, results=max(count, 2*count), read_key=read_key)
    try:
        r = http_get_with_retry(url, params=params, timeout=30)
        data = r.json()
        feeds = data.get("feeds", [])
        out = []
        for f in feeds:
            raw = f.get(field)
            if raw is None or raw == "":
                continue
            try:
                val = float(raw)
            except:
                continue
            ts = parse_thingspeak_time(f.get("created_at"))
            out.append((ts, val))
        out.sort(key=lambda x: x[0])
        ts_list = [ts for ts, _ in out][-count:]
        vals    = [v for _, v in out][-count:]
        return ts_list, vals
    except Exception as e:
        print(f"[TS READ] error fetching tag={tag} channel={channel}: {e}")
        return [], []

def ts_fetch_history_for_tag(tag, start_ts_ms=HISTORY_START_TS_MS, end_ts_ms=None, page_limit=HISTORY_PAGE_LIMIT):
    """
    Fetch history (up to page_limit) for given tag from ThingSpeak actual channel.
    Returns list of (ts,val)
    """
    field = THINGSPEAK_FIELDS_ACTUAL.get(tag)
    if field is None:
        return []
    channel = THINGSPEAK_CHANNEL_ACTUAL_ID
    read_key = THINGSPEAK_READ_KEY_ACTUAL or None
    url, params = thingspeak_feeds_url(channel, results=page_limit, read_key=read_key)
    try:
        r = http_get_with_retry(url, params=params, timeout=60)
        data = r.json()
        feeds = data.get("feeds", [])
        out = []
        for f in feeds:
            raw = f.get(field)
            if raw is None or raw == "":
                continue
            try:
                val = float(raw)
            except:
                continue
            ts = parse_thingspeak_time(f.get("created_at"))
            if ts < start_ts_ms:
                continue
            if end_ts_ms and ts > end_ts_ms:
                continue
            out.append((ts, val))
        out.sort(key=lambda x: x[0])
        return out
    except Exception as e:
        print(f"[TS READ] history error for tag={tag} channel={channel}: {e}")
        return []

# Backwards-compatible interfaces (names used by the rest of your script)
def tb_fetch_last_points(host, device_id, jwt_bearer, key, count):
    # 'key' is the tag like 'temperature'
    return ts_fetch_last_points_for_tag(key, count)

def tb_fetch_key_history_all(host, device_id, jwt_bearer, key, start_ts_ms=HISTORY_START_TS_MS, end_ts_ms=None, page_limit=HISTORY_PAGE_LIMIT):
    return ts_fetch_history_for_tag(key, start_ts_ms=start_ts_ms, end_ts_ms=end_ts_ms, page_limit=page_limit)

# ========= export_history_csv (uses ThingSpeak reads) =========
def export_history_csv(host, device_id, jwt_bearer, key_map, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    key_to_rows = {}
    now_ms = int(time.time() * 1000)
    for tag, key in key_map.items():
        print(f"[history] fetching '{tag}' (tag='{key}') from ThingSpeak...")
        rows = tb_fetch_key_history_all(host, device_id, jwt_bearer, key,
                                        start_ts_ms=HISTORY_START_TS_MS,
                                        end_ts_ms=now_ms)
        key_to_rows[tag] = rows
        print(f"[history]  {tag}: {len(rows)} rows")

    dfs = []
    for tag, rows in key_to_rows.items():
        if not rows:
            continue
        ts = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(ts, unit="ms", utc=True).tz_convert(timezone.utc),
            tag: vals
        })
        dfs.append(df)

    if not dfs:
        cols = ["Timestamp"] + list(key_map.keys())
        merged = pd.DataFrame(columns=cols)
    else:
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="Timestamp", how="outer")
        merged = merged.sort_values("Timestamp").reset_index(drop=True)

    fname = f"ts_history_{int(time.time())}.csv"
    fpath = os.path.join(export_dir, fname)
    merged.to_csv(fpath, index=False)
    return fpath, len(merged)

# ========= Models & scaling =========
def load_best_lookback(meta_path: str, default_lookback: int = 4) -> int:
    if not os.path.exists(meta_path):
        print(f"[meta] {meta_path} not found — using lookback={default_lookback}")
        return default_lookback
    with open(meta_path, "r") as f:
        blob = json.load(f)
    try:
        lb = int(blob["meta"]["lookback"])
        print(f"[meta] lookback = {lb}")
        return lb
    except Exception as e:
        print("[meta] parse warning; fallback to default:", e)
        return default_lookback

def load_model_and_scaler(out_dir: str, tag: str):
    model_path  = os.path.join(out_dir, f"{tag}.h5")
    scaler_path = os.path.join(out_dir, f"{tag}_scaler.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_next(model, scaler, last_vals, tag):
    arr = np.array(last_vals, dtype=np.float32).reshape(-1,1)
    x   = scaler.transform(arr)[None, ...]
    y_scaled = model.predict(x, verbose=0)
    y = scaler.inverse_transform(y_scaled)[0,0]
    lo, hi = GLOBAL_RANGES.get(tag, (-np.inf, np.inf))
    return float(max(lo, min(hi, y)))

def build_window_from_vals(vals, lookback, fallback_value):
    if len(vals) >= lookback:
        return vals[-lookback:], "sensor"
    if len(vals) > 0:
        last = vals[-1]
        pad_needed = lookback - len(vals)
        return ([last] * pad_needed) + vals, "padded"
    return [fallback_value] * lookback, "fallback"

# ========= Model directory discovery + wait-for-upload =========
REQUIRED = [
    ("temperature.h5", "temperature_scaler.pkl"),
    ("humidity.h5", "humidity_scaler.pkl"),
    ("rainMmHr.h5", "rainMmHr_scaler.pkl"),
    ("soilMoisture.h5", "soilMoisture_scaler.pkl"),
]

def candidate_dirs():
    dirs = []
    env_dir = os.environ.get("MODELS_DIR")
    if env_dir: dirs.append(env_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    dirs += [script_dir, os.path.join(script_dir, "lstm_models_2024_5111")]
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    dirs += [downloads, os.path.join(downloads, "lstm_models_2024_5111")]
    dirs += ["/content/lstm_models_2024_5111", "/mnt/data"]
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out

def has_all_models(dirpath):
    for h5, pkl in REQUIRED:
        if not os.path.exists(os.path.join(dirpath, h5)):  return False
        if not os.path.exists(os.path.join(dirpath, pkl)): return False
    return True

def list_missing(dirpath):
    miss = []
    for h5, pkl in REQUIRED:
        if not os.path.exists(os.path.join(dirpath, h5)):  miss.append(h5)
        if not os.path.exists(os.path.join(dirpath, pkl)): miss.append(pkl)
    return miss

def resolve_models_dir():
    print("\n=== Searching for model directory ===")
    for d in candidate_dirs():
        print(f"  checking: {d}")
        if has_all_models(d):
            print(f"[models] found all required files in: {d}")
            return d
    wait_dir = os.environ.get("MODELS_DIR") or candidate_dirs()[0]
    print(f"\n[models] Not found yet. Please copy the following files into:\n  {wait_dir}\n")
    print("Required:")
    for h5, pkl in REQUIRED:
        print(f"  - {h5}\n  - {pkl}")
    print("\n(Press ESC to cancel)")
    while True:
        miss = list_missing(wait_dir)
        if not miss:
            print(f"[models] all files present in: {wait_dir}")
            return wait_dir
        print(f"[models] still missing: {', '.join(miss)}")
        for _ in range(5):
            if esc_pressed():
                print("ESC detected — exiting (models not available).")
                sys.exit(1)
            time.sleep(1)

# Locate models dir (blocks until available)
BASE_OUT_DIR = resolve_models_dir()
EXPORT_DIR   = os.path.join(BASE_OUT_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# ========= Load models & lookback =========
meta_path = os.path.join(BASE_OUT_DIR, "meta.json")
LOOKBACK  = load_best_lookback(meta_path, default_lookback=4)
targets   = ["temperature", "humidity", "rainMmHr", "soilMoisture"]

models, scalers = {}, {}
for t in targets:
    m, s = load_model_and_scaler(BASE_OUT_DIR, t)
    models[t], scalers[t] = m, s
    sz_m = os.path.getsize(os.path.join(BASE_OUT_DIR, f"{t}.h5"))
    sz_s = os.path.getsize(os.path.join(BASE_OUT_DIR, f"{t}_scaler.pkl"))
    print(f"[load] {t:<13} OK  (model {sz_m/1024:.1f} KB, scaler {sz_s/1024:.1f} KB)")

# ========= Optional export at startup =========
last_export_unix = 0
if EXPORT_ON_START:
    try:
        csv_path, nrows = export_history_csv(TB_HOST, TB_DEVICE_ID, TB_JWT, TB_KEY_MAP, EXPORT_DIR)
        push_history_link_to_tb(csv_path, nrows)
        last_export_unix = time.time()
    except Exception as e:
        print(f"[history] initial export failed: {e}")

print(f"\n→ Running prediction loop (LOOKBACK={LOOKBACK}, SLEEP_SEC={SLEEP_SEC}).")
print("→ Press ESC to stop.\n")

# ========= State across ticks =========
WINDOW_CACHE = {}   # target -> list[float] length LOOKBACK
LAST_PRED    = {}   # target -> float
LAST_TS      = {}   # target -> int (ms)

# ========= Main Loop =========
try:
    iteration = 0
    while True:
        iteration += 1
        pred_payload    = {}
        actuals_payload = {}
        src_flags       = {}
        sensor_seen     = False  # <-- CHANGED: track whether any fresh sensor data arrived this tick

        # Periodic CSV export
        if HISTORY_REFRESH_MINUTES and (time.time() - last_export_unix) >= (HISTORY_REFRESH_MINUTES * 60):
            try:
                csv_path, nrows = export_history_csv(TB_HOST, TB_DEVICE_ID, TB_JWT, TB_KEY_MAP, EXPORT_DIR)
                push_history_link_to_tb(csv_path, nrows)
                last_export_unix = time.time()
            except Exception as e:
                print(f"[history] periodic export failed: {e}")

        for t in targets:
            tb_key = TB_KEY_MAP[t]

            # 1) Read latest points from ThingSpeak actual channel
            ts_ms, vals = [], []
            try:
                ts_ms, vals = tb_fetch_last_points(TB_HOST, TB_DEVICE_ID, TB_JWT, tb_key, LOOKBACK)
            except Exception as e:
                print(f"[{iteration:04d}] [{t}] TS READ error: {e}")

            latest_ts = ts_ms[-1] if len(ts_ms) > 0 else None
            prev_ts   = LAST_TS.get(t, None)

            # 2) Build input window
            if latest_ts is not None and (prev_ts is None or latest_ts != prev_ts):
                window, src = build_window_from_vals(vals, LOOKBACK, DEFAULT_FALLBACKS[t])
                LAST_TS[t] = latest_ts
                # mark that we saw fresh sensor data for at least one tag
                sensor_seen = True  # <-- CHANGED: record that sensor data arrived this tick
            else:
                if t in WINDOW_CACHE and t in LAST_PRED:
                    window = WINDOW_CACHE[t][1:] + [LAST_PRED[t]]  # autoregressive step
                    src = "predloop"
                elif len(vals) > 0:
                    window, src = build_window_from_vals(vals, LOOKBACK, DEFAULT_FALLBACKS[t])
                else:
                    window = [DEFAULT_FALLBACKS[t]] * LOOKBACK
                    src = "fallback"

            # 3) Predict
            try:
                y_pred = predict_next(models[t], scalers[t], window, t)
            except Exception as e:
                print(f"[{iteration:04d}] [{t}] Predict error: {e}")
                y_pred = float(DEFAULT_FALLBACKS[t])

            # 4) Update state
            WINDOW_CACHE[t] = window[:]
            LAST_PRED[t]    = y_pred

            # 5) Prepare payloads
            pred_payload[TB_PRED_KEY_MAP[t]] = y_pred
            src_flags[f"{t}PredSource"] = src

            # Include ACTUALS only when window was built from fresh sensor data
            if POST_ACTUALS and src == "sensor":
                try:
                    actuals_payload[t] = float(vals[-1])
                except Exception:
                    pass

            print(f"[{iteration:04d}] {t:<13} src={src:<8} window={np.round(window,3).tolist()} → next={y_pred:.3f}")

        # 6) Push predictions + actuals to ThingsBoard (existing behavior)
        if pred_payload:
            payload = pred_payload.copy()
            # Only include actuals when they exist (we don't want to push fallback seeds)
            if POST_ACTUALS and actuals_payload:
                payload.update(actuals_payload)
            payload.update(src_flags)
            payload["clientTs"] = int(time.time() * 1000)

            try:
                http_post_with_retry(TB_TELEMETRY_POST, json_payload=payload, timeout=30)
                print(f"[{iteration:04d}] TB push OK → {payload}\n")
            except Exception as e:
                print(f"[{iteration:04d}] TB PUSH error: {e}\n")

            # ---- Immediately after TB push: push actuals and preds to ThingSpeak ----
            # Only push actuals if fresh sensor data arrived this tick and we have actuals to push.
            # If there was no fresh data, we DO NOT push actuals (so we don't push stale/fallback values).
            if POST_ACTUALS and sensor_seen and actuals_payload:
                # sensor_seen True means at least one tag had fresh sensor data this tick.
                push_actuals_to_thingspeak(actuals_payload)
            else:
                # explicit log to clarify behavior
                if POST_ACTUALS and not sensor_seen:
                    print(f"[{iteration:04d}] No fresh sensor data this tick — skipping ThingSpeak actuals push.")
                elif POST_ACTUALS and sensor_seen and not actuals_payload:
                    print(f"[{iteration:04d}] sensor_seen True but actuals_payload empty — nothing to push for actuals.")

            # Always attempt to push preds (subject to ThingSpeak rate-limit handling inside function)
            push_preds_to_thingspeak(pred_payload)

        else:
            print(f"[{iteration:04d}] No predictions this round.\n")

        if esc_pressed():
            print("ESC detected — stopping loop.")
            break

        # tiny sleep to yield CPU (internal cadence)
        time.sleep(SLEEP_SEC)

except KeyboardInterrupt:
    print("\nStopped by KeyboardInterrupt (Ctrl+C).")

print("Done.")

