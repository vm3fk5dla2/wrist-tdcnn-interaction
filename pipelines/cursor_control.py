import asyncio
import time
import torch
from bleak import BleakClient
import tkinter as tk
from collections import deque
import numpy as np

# Non-blocking CSV logging support.
# The BLE callback only enqueues rows; a background thread writes them to disk.
import os
import csv
import queue
import threading
import atexit
from PIL import Image, ImageTk


# Imports for your model, dataset, and parameters
from models.ultralight_model import UltraLightCNN1D
from preprocessing.cursor_control_preprocessing import BLEPacketDataset
from configs.cursor_control_parameters import Params


###############################################################################
# Global Config
###############################################################################
params = Params ()

# BLE info
CHARACTERISTIC_UUID = params.ble_characteristic_uuid
DEVICE_ADDRESS = params.ble_device_address


# Tkinter main window (cursor) params
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CURSOR_SIZE = 30
CURSOR_WIDTH = CURSOR_SIZE
CURSOR_HEIGHT = int (CURSOR_SIZE * 1.5)

CHROME_POPUP_MAX_FRACTION = 0.6     # fraction of control window (keeps image on-screen)

# Pinch behavior
PINCH_COOLDOWN_SECONDS = 3.0

###############################################################################
# Shared Storage
###############################################################################
# Parsed sensor tuples from the BLE callback. Task2 uses these packets to build
# sliding windows for preprocessing.
PARSED_PACKET_STORAGE = deque ()

# Preprocessed model inputs produced by Task2.
SAMPLE_STORAGE = deque ()

# Predicted classes produced by Task3.
PREDICTION_STORAGE = deque ()

MIN_MAX_DICT = None        # filled after pre-task
PRETASK_ACTIVE = False     # toggles dataset tap during pre-task
DATASET_PRE = None         # BLEPacketDataset used only during pre-task
BLE_CLIENT = None          # shared BleakClient used during calibration and live BLE notifications
REST_CLASS_INDEX = 0       # class index used when low-confidence predictions are forced to rest
REST_PROB_THRESHOLD = 0.80  # change this value as needed
PINCH_PROB_THRESHOLD = 0.80  # change this value as needed for pinch

# App shutdown coordination so the UI and CSV logger can close cleanly.
APP_SHUTDOWN = asyncio.Event()

# Task2 only starts after calibration finishes so pre-collect packets do not
# enter the inference pipeline.
PIPELINE_RX_ENABLED = threading.Event()
PIPELINE_RX_ENABLED.clear()

###############################################################################
# Selected-channel CSV logger
###############################################################################
# The BLE callback only enqueues rows. A background thread writes them to disk.
CSV_LOG_ENABLED = True

CSV_LOG_DIR = "realtime_csv_logs_0520_ws120_until_acc85_pinch_anywhere_R0.8_P0.8"
CSV_LOG_QUEUE_MAX = 50000          # protects RAM if disk stalls; callback never blocks
CSV_LOG_FLUSH_ROWS = 500           # write in batches to reduce overhead
CSV_LOG_FLUSH_SECONDS = 0.5        # max time between flushes

_CSV_LOG_QUEUE = queue.Queue (maxsize = CSV_LOG_QUEUE_MAX)
_CSV_LOG_STOP  = threading.Event ()
_CSV_LOG_THREAD = None
_CSV_LOG_FILEPATH = None
_CSV_DROPPED_ROWS = 0
_CSV_SENSOR_KEYS = []              # e.g., ["Sensor 1", "Sensor 2", ...]


def _csv_logger_worker (file_path, header):
    """Background worker that drains _CSV_LOG_QUEUE and writes to CSV in batches."""
    # Large buffer reduces syscall frequency.
    with open (file_path, "w", newline="", buffering = 1024 * 1024) as f:
        writer = csv.writer (f)
        writer.writerow (header)

        batch = []
        last_flush = time.monotonic ()

        while (not _CSV_LOG_STOP.is_set ()) or (not _CSV_LOG_QUEUE.empty ()):
            try:
                row = _CSV_LOG_QUEUE.get (timeout = 0.2)
                batch.append (row)
            except queue.Empty:
                pass

            now = time.monotonic ()
            if batch and (len (batch) >= CSV_LOG_FLUSH_ROWS or (now - last_flush) >= CSV_LOG_FLUSH_SECONDS):
                writer.writerows (batch)
                batch.clear ()
                last_flush = now

        # Final flush
        if batch:
            writer.writerows (batch)


def start_sensor_csv_logger(selected_channels):
    """Start the background thread (CSV logger). Returns the filepath or None if disabled."""
    global _CSV_LOG_THREAD, _CSV_LOG_FILEPATH, _CSV_SENSOR_KEYS

    if not CSV_LOG_ENABLED:
        return None
    
    if _CSV_LOG_THREAD is not None:
        return _CSV_LOG_FILEPATH

    selected_channels = list(selected_channels)
    _CSV_SENSOR_KEYS = [f"Sensor {i}" for i in range(1, len(selected_channels) + 1)]

    os.makedirs(CSV_LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    _CSV_LOG_FILEPATH = os.path.join(CSV_LOG_DIR, f"selected_channels_{ts}.csv")

    header = [f"Channel {ch}" for ch in selected_channels]

    _CSV_LOG_STOP.clear()
    _CSV_LOG_THREAD = threading.Thread(
        target = _csv_logger_worker,
        args = (_CSV_LOG_FILEPATH, header),
        daemon = True,
    )
    _CSV_LOG_THREAD.start()

    # Best-effort shutdown on process exit.
    atexit.register(stop_sensor_csv_logger)

    print(f"[CSV] Logging selected channels to: {_CSV_LOG_FILEPATH}")
    return _CSV_LOG_FILEPATH


def stop_sensor_csv_logger() -> None:
    """Attempt to stop the background CSV logger and flush remaining rows."""
    global _CSV_LOG_THREAD
    if _CSV_LOG_THREAD is None:
        return

    _CSV_LOG_STOP.set()
    try:
        _CSV_LOG_THREAD.join(timeout = 3.0)
    except Exception:
        pass
    _CSV_LOG_THREAD = None

    if _CSV_DROPPED_ROWS:
        print(f"[CSV] WARNING: dropped {_CSV_DROPPED_ROWS} rows (logger fell behind).")


def _enqueue_selected_channel_row(sensors: dict) -> None:
    """Non-blocking: enqueue a CSV row for the already-parsed `sensors` dict."""
    global _CSV_DROPPED_ROWS

    if (not CSV_LOG_ENABLED) or (_CSV_LOG_THREAD is None):
        return

    # Keep callback work minimal: only dict lookups and queue put.
    row = [sensors.get(key, 0) for key in _CSV_SENSOR_KEYS]
    try:
        _CSV_LOG_QUEUE.put_nowait(row)
    except queue.Full:
        # Never block the real-time pipeline.
        _CSV_DROPPED_ROWS += 1



###############################################################################
# Selected channels used throughout the pipeline
###############################################################################
_SELECTED_CHANNELS = tuple(getattr(params, "selected_channels", ()))

if not _SELECTED_CHANNELS:
    raise ValueError("params.selected_channels must not be empty.")



###############################################################################
# Global objects used across tasks (main window)
###############################################################################
model = None
device = None
root = None
canvas = None

MAIN_BG_PHOTO = None
CURSOR_PHOTO = None

cursor = None
cursor_x = None  # cursor top-left x
cursor_y = None  # cursor top-left y

chrome_popup_item = None
chrome_popup_photo = None
CHROME_POPUP_VISIBLE_UNTIL = 0.0

UI_FRAME_INTERVAL_SECONDS = 1 / 125



###############################################################################
# BLE packet callback
###############################################################################
def ble_notification_callback(_sender: int, data: bytearray):
    """
    Parse each BLE packet once, then update the pre-task dataset,
    selected-channel CSV logger, and Task2 input queue.
    """
    packet_hex = data.hex()

    try:
        sensors = BLEPacketDataset.parse_ble_packet(
            packet_hex,
            selected_channels = _SELECTED_CHANNELS,
        )
    except Exception:
        return

    _enqueue_selected_channel_row(sensors)

    # During pre-collect, tap packets into DATASET_PRE for min/max estimation.
    if PRETASK_ACTIVE and DATASET_PRE is not None:
        DATASET_PRE.add_packet(sensors)

    # After calibration, Task2 consumes parsed sensor tuples with stride 1.
    if PIPELINE_RX_ENABLED.is_set():
        num_sensors = len(_SELECTED_CHANNELS)
        sensor_vec = tuple(sensors.get(f"Sensor {i}", 0) for i in range(1, num_sensors + 1))
        PARSED_PACKET_STORAGE.append(sensor_vec)



###############################################################################
# Pre-task: data acquisition for calibration (BLE)
###############################################################################
def _draw_instruction(text):
    """
    Draw a full-canvas black overlay with white text centered.
    Returns the tag used to remove it later.
    """
    global canvas
    tag = "instruction_overlay"
    # Remove any previous overlay
    canvas.delete(tag)
    # Full black rect
    canvas.create_rectangle (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, fill = "black",
                             outline = "", tags = tag)
    # Centered white text
    canvas.create_text (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, text = text,
                        fill = "white", font = ("Helvetica", 36, "bold"), tags = tag)
    return tag


async def _show_instruction(text, seconds):
    """
    Draw a full-canvas black overlay with centered text on the main window.
    The Tk update loop keeps the main window refreshing while the overlay is shown.
    """
    tag = _draw_instruction(text)
    try:
        end_time = time.monotonic() + seconds
        while (not APP_SHUTDOWN.is_set()) and (time.monotonic() < end_time):
            await asyncio.sleep(UI_FRAME_INTERVAL_SECONDS)
    finally:
        try:
            if canvas is not None:
                canvas.delete(tag)
        except tk.TclError:
            pass

async def precollect_minmax_sequence():
    """
    Start BLE if needed, show guided screens, collect data for the selected
    channels, compute per-sensor min/max, and keep notifications running for the live inference pipeline.
    """
    global BLE_CLIENT, PRETASK_ACTIVE, DATASET_PRE, MIN_MAX_DICT
    global PIPELINE_RX_ENABLED

    # Ensure the inference pipeline does not accumulate packets during pre-collect.
    # Task2 input is enabled only right before the main tasks start.
    PIPELINE_RX_ENABLED.clear()
    PARSED_PACKET_STORAGE.clear()

    if BLE_CLIENT is None or not BLE_CLIENT.is_connected:
        BLE_CLIENT = BleakClient(DEVICE_ADDRESS)
        print("[PreCollect] Connecting to BLE ...")
        await BLE_CLIENT.connect()
        await BLE_CLIENT.start_notify(CHARACTERISTIC_UUID, ble_notification_callback)
        print("[PreCollect] Started notifications.")

    # Set up the pre-collection dataset.
    DATASET_PRE = BLEPacketDataset(params)

    # Guided sequence.
    await _show_instruction("Please wait", 45)

    PRETASK_ACTIVE = True
    await _show_instruction("Collect normalization data", 15)

    PRETASK_ACTIVE = False
    # Compute per-sensor min/max from the tapped data.
    min_max = {}
    for sensor_name, vals in (DATASET_PRE.sensors or {}).items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            min_val, max_val = 0.0, 1.0
        else:
            min_val, max_val = float(np.min(arr)), float(np.max(arr))
        min_max[sensor_name] = (min_val, max_val)
    MIN_MAX_DICT = min_max

    # Keep Task2 input disabled during pre-collect (already cleared above).
    # Clear again as a precaution.
    PIPELINE_RX_ENABLED.clear()
    PARSED_PACKET_STORAGE.clear()



###############################################################################
# Task2 preprocessing helpers
###############################################################################
# Task2 works on already-parsed sensor tuples from PARSED_PACKET_STORAGE.
# The helpers below build min/max arrays and convert each window into a model
# input with vectorized normalization and chunk-wise trimmed means.
TASK2_CHUNK_SIZE = 10

def _task2_build_minmax_arrays(min_max_dict: dict, num_sensors: int):
    """Return (mins, ranges) arrays shaped (C, 1) aligned to Sensor 1..C."""
    mins = np.zeros((num_sensors, 1), dtype = np.float32)
    ranges = np.ones((num_sensors, 1), dtype = np.float32)
    for i in range(num_sensors):
        key = f"Sensor {i+1}"
        if min_max_dict and key in min_max_dict:
            mn, mx = min_max_dict[key]
            mn_f = float(mn)
            mx_f = float(mx)
            mins[i, 0] = mn_f
            ranges[i, 0] = float(mx_f - mn_f)
        else:
            # Fallback: if min/max is missing (should not happen after precollect),
            # treat it as identity normalization.
            mins[i, 0] = 0.0
            ranges[i, 0] = 1.0
    return mins, ranges


def _task2_window_to_sample(window_buf: np.ndarray,
                            work_buf: np.ndarray,
                            mins: np.ndarray,
                            ranges: np.ndarray,
                            num_to_ignore: int,
                            chunk_size: int) -> torch.Tensor:
    """
    Convert a single (C, window_size) window into a (C, T) sample tensor,
    where T is window_size // chunk_size when divisible, otherwise ceil(window_size / chunk_size).
    """
    # work_buf = normalized (and then sorted per chunk in-place)
    work_buf[:] = window_buf

    # Per-sensor min–max normalization using calibration min/max from pre-collection.
    # If range <= 0, dataset keeps raw values (then clips), so we do the same.
    pos = (ranges[:, 0] > 0.0)
    if np.any(pos):
        work_buf[pos, :] = (work_buf[pos, :] - mins[pos, :]) / ranges[pos, :]

    np.clip(work_buf, 0.0, 1.0, out=work_buf)

    C, window_size = work_buf.shape
    if window_size % chunk_size != 0:
        # Rare fallback (params.window_size is 120, so normally divisible by 10).
        # We keep correctness over speed here.
        num_chunks = int(np.ceil(window_size / chunk_size))
        out = np.zeros((C, num_chunks), dtype = np.float32)
        for c in range(C):
            sensor_window = work_buf[c]
            col = 0
            for j in range(0, window_size, chunk_size):
                chunk = np.sort(np.array(sensor_window[j:j + chunk_size], dtype = np.float32))
                if len(chunk) == 0:
                    out[c, col] = 0.0
                elif len(chunk) <= 2 * num_to_ignore:
                    out[c, col] = float(np.mean(chunk))
                else:
                    trimmed = chunk[num_to_ignore:-num_to_ignore]
                    out[c, col] = float(np.mean(trimmed)) if len(trimmed) else 0.0
                col += 1
        return torch.from_numpy(out).float()

    num_chunks = window_size // chunk_size
    chunk_view = work_buf.reshape(C, num_chunks, chunk_size)

    # In-place sort (avoids allocating a new array)
    chunk_view.sort(axis = 2)

    if chunk_size <= 2 * num_to_ignore:
        smooth = chunk_view.mean(axis = 2)
    else:
        smooth = chunk_view[:, :, num_to_ignore:chunk_size - num_to_ignore].mean(axis = 2)

    # smooth: (C, num_chunks) float32 -> torch float32
    return torch.from_numpy(smooth).float()


def _warm_up_task2_path(min_max_dict: dict, num_sensors: int, window_size: int, num_to_ignore: int, repeat: int = 3):
    """
    Warm up the Task2 preprocessing path before main-task starts.
    This is intentionally isolated from the live pipeline so it does not affect
    packet flow.
    """
    if min_max_dict is None or num_sensors <= 0 or window_size <= 0:
        return

    mins, ranges = _task2_build_minmax_arrays(min_max_dict, num_sensors)
    dummy_window = np.zeros((num_sensors, window_size), dtype=np.float32)
    dummy_work = np.empty_like(dummy_window)

    # Use a mid-range value when min/max is available so the normalization path
    # is exercised in a representative way.
    for i in range(num_sensors):
        if ranges[i, 0] > 0.0:
            dummy_window[i, :] = mins[i, 0] + (0.5 * ranges[i, 0])

    for _ in range(max(1, int(repeat))):
        sample = _task2_window_to_sample(
            dummy_window,
            dummy_work,
            mins,
            ranges,
            num_to_ignore=num_to_ignore,
            chunk_size=TASK2_CHUNK_SIZE,
        )
        _ = sample.shape


def _warm_up_task3_path(model, device, num_sensors: int, window_size: int, repeat: int = 3):
    """
    Warm up model/device inference before main-task starts.
    This targets the first Task3 spike without touching the live pipeline.
    """
    if model is None or num_sensors <= 0 or window_size <= 0:
        return

    if window_size % TASK2_CHUNK_SIZE == 0:
        time_steps = window_size // TASK2_CHUNK_SIZE
    else:
        time_steps = int(np.ceil(window_size / TASK2_CHUNK_SIZE))

    dummy_input = torch.zeros((1, num_sensors, time_steps), dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(max(1, int(repeat))):
            logits = model(dummy_input)
            _ = torch.softmax(logits, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)


###############################################################################
# Task 2: Data Processing (Produce Single Samples)
###############################################################################
async def task2_data_processing():
    """
    Task2: PARSED_PACKET_STORAGE -> SAMPLE_STORAGE

    The task waits for one full window, keeps the latest window if backlog
    grows, builds the sample directly from PARSED_PACKET_STORAGE, and then
    advances the window by one packet.
    """
    window_size = params.window_size
    num_to_ignore = getattr(params, "num_to_ignore", 0)

    num_sensors = len(_SELECTED_CHANNELS)

    # Temporary working arrays reused for every sample.
    window_buf = np.empty((num_sensors, window_size), dtype = np.float32)
    work_buf = np.empty_like(window_buf)

    mins = None
    ranges = None
    minmax_ready = False

    while not APP_SHUTDOWN.is_set():
        if not PIPELINE_RX_ENABLED.is_set():
            await asyncio.sleep(0)
            continue

        if MIN_MAX_DICT is None:
            await asyncio.sleep(0)
            continue

        if not minmax_ready:
            mins, ranges = _task2_build_minmax_arrays(MIN_MAX_DICT, num_sensors)
            minmax_ready = True

        backlog = len(PARSED_PACKET_STORAGE)
        if backlog < window_size:
            await asyncio.sleep(0)
            continue

        # If Task2 falls behind, keep only the latest window.
        if backlog > window_size:
            excess = backlog - window_size
            for _ in range(excess):
                PARSED_PACKET_STORAGE.popleft()

        window_packets = list(PARSED_PACKET_STORAGE)

        try:
            for col, sensor_vec in enumerate(window_packets):
                if sensor_vec is None or len(sensor_vec) != num_sensors:
                    raise ValueError("Invalid sensor vector in Task2 window")

                window_buf[:, col] = sensor_vec

        except Exception:
            PARSED_PACKET_STORAGE.popleft()
            await asyncio.sleep(0)
            continue

        try:
            sample_tensor = _task2_window_to_sample(
                window_buf,
                work_buf,
                mins,
                ranges,
                num_to_ignore = num_to_ignore,
                chunk_size = TASK2_CHUNK_SIZE,
            ).unsqueeze(0)  # [1, C, T]

        except Exception:
            PARSED_PACKET_STORAGE.popleft()
            await asyncio.sleep(0)
            continue

        SAMPLE_STORAGE.append(sample_tensor)

        # Sliding window with stride 1.
        PARSED_PACKET_STORAGE.popleft()

        await asyncio.sleep(0)


###############################################################################
# Task 3: Model Inference (Consume Single Samples)
###############################################################################
async def task3_model_inference():
    """Task3: SAMPLE_STORAGE -> PREDICTION_STORAGE."""
    global model, device

    while not APP_SHUTDOWN.is_set():
        await asyncio.sleep(0)

        if not SAMPLE_STORAGE:
            continue

        sample_tensor = SAMPLE_STORAGE.popleft()

        # Shape: [1, channels, time_steps]
        sample_tensor = sample_tensor.float().to(device)

        with torch.no_grad():
            logits = model(sample_tensor)
            probs = torch.softmax(logits, dim = 1)
            max_prob, pred_idx = torch.max(probs, dim = 1)

        max_prob_val = float(max_prob.item())
        pred_class = int(pred_idx.item())

        if pred_class == 3:
            if max_prob_val < PINCH_PROB_THRESHOLD:
                pred_class = REST_CLASS_INDEX
        elif max_prob_val < REST_PROB_THRESHOLD:
            pred_class = REST_CLASS_INDEX

        PREDICTION_STORAGE.append(pred_class)


###############################################################################
# Task 4: Mouse Cursor Control
###############################################################################
async def task4_cursor_control():
    """
    Task4: PREDICTION_STORAGE -> cursor movement state

    Task4 updates only the logical cursor and popup state. tk_update_loop()
    refreshes the visible Tk items.
    """
    global cursor_x, cursor_y

    pinch_cooldown_until = 0.0

    while not APP_SHUTDOWN.is_set():
        await asyncio.sleep(0)

        if len(PREDICTION_STORAGE) == 0:
            continue

        pred_class = PREDICTION_STORAGE.popleft()

        
        # dy is reserved for future vertical gesture classes.
        dx, dy = 0, 0
        do_move = False
        do_pinch_feedback = False
        next_pinch_cooldown_until = None

        if pred_class == 1:
            dx = -3
            do_move = True

        elif pred_class == 2:
            dx = 3
            do_move = True

        elif pred_class == 3:
            now = time.monotonic()
            if now >= pinch_cooldown_until:
                do_pinch_feedback = True
                next_pinch_cooldown_until = now + PINCH_COOLDOWN_SECONDS

        if do_pinch_feedback:
            _show_chrome_popup(duration_ms = int(PINCH_COOLDOWN_SECONDS * 1000))
            pinch_cooldown_until = next_pinch_cooldown_until

        if do_move and cursor_x is not None and cursor_y is not None:
            new_x = cursor_x + dx
            new_y = cursor_y + dy
            new_x = max(0, min(new_x, WINDOW_WIDTH - 3))
            new_y = max(0, min(new_y, WINDOW_HEIGHT - 3))
            cursor_x = new_x
            cursor_y = new_y



###############################################################################
# Pinch feedback helper (chrome.png overlay)
###############################################################################
def _show_chrome_popup(duration_ms: int) -> None:
    """Request that chrome.png stays visible for the given duration."""
    global CHROME_POPUP_VISIBLE_UNTIL

    if chrome_popup_item is None:
        return

    visible_until = time.monotonic() + (max(0, int(duration_ms)) / 1000.0)
    if visible_until > CHROME_POPUP_VISIBLE_UNTIL:
        CHROME_POPUP_VISIBLE_UNTIL = visible_until


###############################################################################
# Main window UI helpers
###############################################################################
def _init_main_window_static_scene():
    """Create static main-window items once and never touch them again."""
    global canvas, MAIN_BG_PHOTO

    canvas = tk.Canvas(root, width = WINDOW_WIDTH, height = WINDOW_HEIGHT)
    canvas.pack()

    bg_image = Image.open("background.png").convert("RGBA")
    bg_image = bg_image.resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.Resampling.LANCZOS)
    MAIN_BG_PHOTO = ImageTk.PhotoImage(bg_image)
    canvas.create_image(0, 0, image = MAIN_BG_PHOTO, anchor = "nw", tags = ("main_static", "background"))


def _init_main_window_dynamic_scene():
    """Create reusable dynamic main-window items once."""
    global cursor, cursor_x, cursor_y
    global CURSOR_PHOTO, chrome_popup_item, chrome_popup_photo
    global CHROME_POPUP_VISIBLE_UNTIL

    CHROME_POPUP_VISIBLE_UNTIL = 0.0

    try:
        chrome_im = Image.open("chrome.png").convert("RGBA")
        max_w = int(WINDOW_WIDTH * CHROME_POPUP_MAX_FRACTION)
        max_h = int(WINDOW_HEIGHT * CHROME_POPUP_MAX_FRACTION)
        w, h = chrome_im.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        chrome_im = chrome_im.resize(new_size, Image.Resampling.LANCZOS)
        chrome_popup_photo = ImageTk.PhotoImage(chrome_im)
        chrome_popup_item = canvas.create_image(
            WINDOW_WIDTH // 2,
            WINDOW_HEIGHT // 2,
            image = chrome_popup_photo,
            anchor = "center",
            state = "hidden",
            tags = ("main_dynamic", "chrome_popup"),
        )
    except Exception as e:
        print(f"[UI] Could not load chrome.png: {e}")
        chrome_popup_photo = None
        chrome_popup_item = None

    cursor_image = Image.open("cursor.png").convert("RGBA")
    cursor_image = cursor_image.resize((CURSOR_WIDTH, CURSOR_HEIGHT), Image.Resampling.LANCZOS)
    CURSOR_PHOTO = ImageTk.PhotoImage(cursor_image)

    start_x = WINDOW_WIDTH // 2 - CURSOR_WIDTH // 2
    start_y = WINDOW_HEIGHT // 2 - CURSOR_HEIGHT // 2
    cursor = canvas.create_image(start_x, start_y, image = CURSOR_PHOTO, anchor = "nw", tags = ("main_dynamic", "cursor"))
    cursor_x = start_x
    cursor_y = start_y


def _apply_main_window_dynamic_updates():
    """Refresh only dynamic main-window items."""
    if canvas is None:
        return

    if cursor is not None and cursor_x is not None and cursor_y is not None:
        canvas.coords(cursor, cursor_x, cursor_y)

    if chrome_popup_item is not None:
        popup_visible = time.monotonic() < CHROME_POPUP_VISIBLE_UNTIL
        canvas.itemconfig(chrome_popup_item, state = "normal" if popup_visible else "hidden")
        if popup_visible:
            canvas.lift(chrome_popup_item)



###############################################################################
# Tkinter Update Loop (main window)
###############################################################################
async def tk_update_loop():
    """
    Refresh the main Tk window continuously.

    Static items are created once during window initialization. This loop updates
    only the dynamic items:
      - main window: cursor + chrome popup
    """
    while not APP_SHUTDOWN.is_set():
        try:
            _apply_main_window_dynamic_updates()
            root.update()
        
        except tk.TclError:
            APP_SHUTDOWN.set()
            break

        await asyncio.sleep(max(0.0, UI_FRAME_INTERVAL_SECONDS))

###############################################################################
# Main Entry Point
###############################################################################
async def main():
    global model, device
    global root

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    PIPELINE_RX_ENABLED.clear()
    PARSED_PACKET_STORAGE.clear()

    start_sensor_csv_logger(params.selected_channels)

    ui_task = None

    try:
        root = tk.Tk()
        root.title("Control Window")

        def _on_close():
            PIPELINE_RX_ENABLED.clear()
            APP_SHUTDOWN.set()
            try:
                root.destroy()
            except tk.TclError:
                pass

        root.protocol("WM_DELETE_WINDOW", _on_close)

        _init_main_window_static_scene()
        _init_main_window_dynamic_scene()

        ui_task = asyncio.create_task(tk_update_loop())

        await precollect_minmax_sequence()
        if APP_SHUTDOWN.is_set():
            return

        PIPELINE_RX_ENABLED.clear()
        PARSED_PACKET_STORAGE.clear()
        SAMPLE_STORAGE.clear()
        PREDICTION_STORAGE.clear()

        model_path = params.model_path
        model = UltraLightCNN1D().to(device)
        model.load_state_dict(torch.load(model_path, map_location = device))
        model.eval()

        num_sensors = len(_SELECTED_CHANNELS)

        _warm_up_task2_path(
            MIN_MAX_DICT,
            num_sensors = num_sensors,
            window_size = params.window_size,
            num_to_ignore = getattr(params, "num_to_ignore", 0),
        )

        _warm_up_task3_path(
            model,
            device,
            num_sensors = num_sensors,
            window_size = params.window_size,
        )

        PIPELINE_RX_ENABLED.set()

        await asyncio.gather(
            task2_data_processing(),
            task3_model_inference(),
            task4_cursor_control(),
            ui_task,
        )
    finally:
        APP_SHUTDOWN.set()
        if ui_task is not None and not ui_task.done():
            try:
                await ui_task
            except Exception:
                pass
        stop_sensor_csv_logger()


if __name__ == '__main__':
    asyncio.run(main())
