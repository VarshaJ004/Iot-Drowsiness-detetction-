import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
from contextlib import asynccontextmanager
import json
import threading
import time
import urllib.request
import winsound
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ======================= WIRELESS CONFIGURATION =======================
ESP32_HOST_URL = "http://192.168.4.1"

# Camera 0 = Built-in laptop camera (Driver Monitoring)
DRIVER_CAM_INDEX = 0

# Camera 1 = Secondary Road Hazard Camera (e.g., Iriun iPhone Webcam)
ROAD_SOURCE = 1

# ======================= MODEL ASSETS =======================
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

POTHOLE_MODEL_PATH = "pothole_best.pt"
POTHOLE_MODEL_URL = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"

# Threat Thresholds
EYE_CLOSED_THRESHOLD = 0.42
YAWN_THRESHOLD = 0.50
WARN_CLOSURE_TIME = 1.0
CRITICAL_CLOSURE_TIME = 2.0
YAWN_TRIGGER_TIME = 1.6

# Pothole Settings
POTHOLE_CONF_THRESHOLD = 0.25
YOLO_IMGSZ = 416

templates = Jinja2Templates(directory="templates")

# ======================= SYSTEM STATES =======================
system_running = True
alarm_level = 0
lock = threading.RLock()  # Reentrant lock prevents self-deadlock
latest_frame = None
pothole_model = None

yolo_lock = threading.Lock()
yolo_input_frame = None
latest_potholes = []

telemetry = {
    "face_detected": False,
    "eye_status": "OPEN",
    "yawning": "NO",
    "drowsiness_level": 5,
    "driver_condition": "SAFE",
    "road_condition": "CLEAR",
    "pothole_detected": False,
    "pothole_count": 0,
    "total_potholes_logged": 0,
    "alarm_status": "OFF",
    "seat_belt": "FASTENED",
    "yawns_detected": 0,
    "eye_blinks": 0,
    "alarms_triggered": 0,
    "fps": 30.0,
    "timestamp": ""
}

system_logs = [
    {"time": time.strftime("%I:%M:%S %p"), "type": "success", "msg": "System Initialized"},
    {"time": time.strftime("%I:%M:%S %p"), "type": "success", "msg": "Wi-Fi Vehicle Interlock Active"}
]


def add_log(msg: str, log_type: str = "success"):
    with lock:
        system_logs.insert(0, {"time": time.strftime("%I:%M:%S %p"), "type": log_type, "msg": msg})
        if len(system_logs) > 25:
            system_logs.pop()


# ======================= ASYNC WIRELESS HTTP DISPATCHER =======================
def send_esp32_wifi(cmd: str):
    """Sends HTTP commands over Wi-Fi without blocking camera frames."""
    def _worker():
        try:
            target_url = f"{ESP32_HOST_URL}/cmd?val={cmd}"
            req = urllib.request.Request(target_url, headers={'User-Agent': 'DMS-AI-Client'})
            with urllib.request.urlopen(req, timeout=0.45) as resp:
                pass
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


# ======================= NON-BLOCKING AUDIO ENGINE =======================
def audio_alert_worker():
    """Interruptible sound worker that silences immediately when eyes reopen."""
    global alarm_level, system_running
    while system_running:
        level = alarm_level
        try:
            if level == 2:
                winsound.Beep(2400, 80)
                for _ in range(3):
                    if not system_running or alarm_level != 2:
                        break
                    time.sleep(0.01)
            elif level == 1:
                winsound.Beep(1400, 100)
                for _ in range(8):
                    if not system_running or alarm_level != 1:
                        break
                    time.sleep(0.01)
            else:
                time.sleep(0.02)
        except Exception:
            time.sleep(0.04)


# ======================= ASSET VERIFICATION =======================
def download_with_headers(url: str, target_path: str):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def ensure_models_exist():
    if not os.path.exists(FACE_MODEL_PATH):
        print("[SYSTEM] Downloading Face Landmarker asset...")
        download_with_headers(FACE_MODEL_URL, FACE_MODEL_PATH)
    if not os.path.exists(POTHOLE_MODEL_PATH):
        print("[SYSTEM] Downloading Pothole YOLOv8 weights...")
        download_with_headers(POTHOLE_MODEL_URL, POTHOLE_MODEL_PATH)


# ======================= ASYNCHRONOUS YOLO WORKER =======================
def yolo_pothole_worker():
    global system_running, yolo_input_frame, latest_potholes, pothole_model

    while system_running:
        frame_to_process = None
        with yolo_lock:
            if yolo_input_frame is not None:
                frame_to_process = yolo_input_frame.copy()

        if frame_to_process is None or pothole_model is None:
            time.sleep(0.01)
            continue

        results = pothole_model(
            frame_to_process,
            conf=POTHOLE_CONF_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            verbose=False
        )

        detected_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            detected_boxes.append((x1, y1, x2, y2, conf))

        with yolo_lock:
            latest_potholes = detected_boxes

        time.sleep(0.01)


# ======================= CAMERA INITIALIZERS =======================
def initialize_driver_camera():
    cap = cv2.VideoCapture(DRIVER_CAM_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[DRIVER CAM] Laptop webcam connected on index {DRIVER_CAM_INDEX}")
            return cap
        cap.release()
    return None


def initialize_road_camera():
    if str(ROAD_SOURCE).isdigit():
        target_idx = int(ROAD_SOURCE)
        for cam_id in [target_idx, 2, 1]:
            if cam_id == DRIVER_CAM_INDEX:
                continue
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[ROAD FEED] Camera connected on index {cam_id}")
                    return cap
                cap.release()
    return None


# ======================= REAL-TIME PIPELINE =======================
def cv_pipeline():
    global system_running, alarm_level, latest_frame, telemetry
    global yolo_input_frame, latest_potholes, pothole_model

    ensure_models_exist()

    pothole_model = YOLO(POTHOLE_MODEL_PATH)
    pothole_model.model.names = {0: "pothole"}

    cap_driver = initialize_driver_camera()
    cap_road = initialize_road_camera()

    if cap_driver is None:
        print("[ERROR] Laptop webcam failed to start.")
        return

    base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    eye_closed_start_time = None
    yawn_start_time = None
    is_blinking = False
    is_yawning = False
    blink_timestamps = []
    fps_start_time = time.time()
    frame_counter = 0
    fps = 0.0
    last_pothole_log_time = 0.0
    last_emergency_stop_time = 0.0

    while system_running:
        ret_driver, frame_driver = cap_driver.read()
        if not ret_driver or frame_driver is None:
            time.sleep(0.002)
            continue

        road_frame = None
        if cap_road is not None:
            ret_road, road_frame = cap_road.read()
            if not ret_road or road_frame is None:
                cap_road.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_road, road_frame = cap_road.read()
            if road_frame is not None:
                road_frame = cv2.resize(road_frame, (640, 480))

        frame_counter += 1
        curr_time = time.time()
        if curr_time - fps_start_time >= 1.0:
            fps = frame_counter / (curr_time - fps_start_time)
            frame_counter = 0
            fps_start_time = curr_time

        # Driver Monitoring via MediaPipe
        frame_driver = cv2.flip(frame_driver, 1)
        h, w, _ = frame_driver.shape
        rgb_frame = cv2.cvtColor(frame_driver, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        face_detected = False
        eye_is_closed = False
        jaw_is_open = False
        raw_eye_score = 0.0
        raw_mouth_score = 0.0
        eye_closure_duration = 0.0
        yawn_duration = 0.0

        if detection_result.face_blendshapes and detection_result.face_landmarks:
            face_detected = True
            blendshapes = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}

            raw_left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            raw_right_blink = blendshapes.get("eyeBlinkRight", 0.0)
            raw_eye_score = max(raw_left_blink, raw_right_blink)
            raw_mouth_score = blendshapes.get("jawOpen", 0.0)

            eye_is_closed = raw_eye_score > EYE_CLOSED_THRESHOLD
            jaw_is_open = raw_mouth_score > YAWN_THRESHOLD

            landmarks = detection_result.face_landmarks[0]
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            x_min, x_max = max(0, int(min(xs) * w) - 15), min(w, int(max(xs) * w) + 15)
            y_min, y_max = max(0, int(min(ys) * h) - 25), min(h, int(max(ys) * h) + 20)

            color = (0, 0, 255) if eye_is_closed else (0, 255, 0)
            c_len = 20
            cv2.line(frame_driver, (x_min, y_min), (x_min + c_len, y_min), color, 2)
            cv2.line(frame_driver, (x_min, y_min), (x_min, y_min + c_len), color, 2)
            cv2.line(frame_driver, (x_max, y_min), (x_max - c_len, y_min), color, 2)
            cv2.line(frame_driver, (x_max, y_min), (x_max, y_min + c_len), color, 2)
            cv2.line(frame_driver, (x_min, y_max), (x_min + c_len, y_max), color, 2)
            cv2.line(frame_driver, (x_min, y_max), (x_min, y_max - c_len), color, 2)
            cv2.line(frame_driver, (x_max, y_max), (x_max - c_len, y_max), color, 2)
            cv2.line(frame_driver, (x_max, y_max), (x_max, y_max - c_len), color, 2)

            for idx in [33, 133, 362, 263, 13, 14, 1]:
                pt = landmarks[idx]
                cv2.circle(frame_driver, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 180), -1)

            if eye_is_closed:
                if not is_blinking:
                    is_blinking = True
                    blink_timestamps.append(curr_time)
                if eye_closed_start_time is None:
                    eye_closed_start_time = curr_time
                eye_closure_duration = curr_time - eye_closed_start_time
            else:
                is_blinking = False
                eye_closed_start_time = None
                eye_closure_duration = 0.0

            if jaw_is_open:
                if yawn_start_time is None:
                    yawn_start_time = curr_time
                yawn_duration = curr_time - yawn_start_time
                if yawn_duration >= YAWN_TRIGGER_TIME and not is_yawning:
                    is_yawning = True
                    with lock:
                        telemetry["yawns_detected"] += 1
                        yawns = telemetry["yawns_detected"]
                    add_log(f"Yawn Detected (#{yawns})", "warning")
            else:
                yawn_start_time = None
                is_yawning = False
        else:
            eye_closed_start_time = None
            yawn_start_time = None

        blink_timestamps = [t for t in blink_timestamps if curr_time - t <= 60]

        # Threat Assessment: Fatigue Level & Microsleep Evaluation
        status_text = "SAFE"
        alarm_lvl = 0
        drowsiness_pct = int(np.clip((raw_eye_score * 0.7 + raw_mouth_score * 0.3) * 100, 5, 99))

        is_microsleep = (eye_closure_duration >= CRITICAL_CLOSURE_TIME)
        is_high_fatigue = (drowsiness_pct >= 80)

        # --- CRITICAL HIGH FATIGUE / MICROSLEEP: STOP THE VEHICLE ---
        if is_microsleep or is_high_fatigue:
            status_text = "CRITICAL (MICROSLEEP)"
            alarm_lvl = 2
            drowsiness_pct = max(drowsiness_pct, 95)

            # Transmit STOP command over Wi-Fi repeatedly every 300ms to guarantee delivery
            if curr_time - last_emergency_stop_time > 0.3:
                send_esp32_wifi("STOP")
                last_emergency_stop_time = curr_time

            if alarm_level != 2:
                with lock:
                    telemetry["alarms_triggered"] += 1
                add_log("CRITICAL: High Fatigue / Microsleep! CAR STOPPED.", "danger")

        elif eye_closure_duration >= WARN_CLOSURE_TIME:
            status_text = "WARNING"
            alarm_lvl = 1
            drowsiness_pct = max(drowsiness_pct, 75)

        elif yawn_duration >= YAWN_TRIGGER_TIME:
            status_text = "WARNING"
            alarm_lvl = 1

        # YOLO Road Hazard Detection
        target_yolo_frame = road_frame if road_frame is not None else frame_driver
        with yolo_lock:
            yolo_input_frame = target_yolo_frame.copy()
            current_boxes = list(latest_potholes)

        potholes_in_frame = len(current_boxes)
        pothole_detected = potholes_in_frame > 0

        draw_target = road_frame if road_frame is not None else frame_driver
        for x1, y1, x2, y2, conf in current_boxes:
            cv2.rectangle(draw_target, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(draw_target, f"POTHOLE {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        if pothole_detected:
            if curr_time - last_pothole_log_time > 3.0:
                add_log(f"Road Hazard: {potholes_in_frame} Pothole(s) Ahead!", "warning")
                last_pothole_log_time = curr_time
                with lock:
                    telemetry["total_potholes_logged"] += potholes_in_frame
            if alarm_lvl == 0:
                alarm_lvl = 1

        alarm_level = alarm_lvl

        with lock:
            telemetry.update({
                "face_detected": face_detected,
                "eye_status": "CLOSED" if eye_is_closed else "OPEN",
                "yawning": "YES" if jaw_is_open else "NO",
                "drowsiness_level": drowsiness_pct,
                "driver_condition": status_text,
                "road_condition": "HAZARD DETECTED" if pothole_detected else "CLEAR",
                "pothole_detected": pothole_detected,
                "pothole_count": potholes_in_frame,
                "alarm_status": "ON" if alarm_level > 0 else "OFF",
                "eye_blinks": len(blink_timestamps),
                "fps": round(fps, 1),
                "timestamp": time.strftime("%I:%M:%S %p")
            })

        cv2.putText(frame_driver, "DRIVER MONITOR", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if road_frame is not None:
            cv2.putText(road_frame, "ROAD HAZARD FEED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            display_frame = np.hstack((frame_driver, road_frame))
        else:
            display_frame = frame_driver

        ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ret:
            with lock:
                latest_frame = buffer.tobytes()

    cap_driver.release()
    if cap_road is not None:
        cap_road.release()


# ======================= LIFESPAN MANAGER =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global system_running
    system_running = True
    t_audio = threading.Thread(target=audio_alert_worker, daemon=True)
    t_cv = threading.Thread(target=cv_pipeline, daemon=True)
    t_yolo = threading.Thread(target=yolo_pothole_worker, daemon=True)

    t_audio.start()
    t_cv.start()
    t_yolo.start()
    yield
    system_running = False
    time.sleep(0.15)


app = FastAPI(title="AI Driver Assistant DMS", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


async def generate_mjpeg():
    global latest_frame, system_running
    try:
        while system_running:
            with lock:
                frame_data = latest_frame
            if frame_data is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
            await asyncio.sleep(0.033)
    except (GeneratorExit, asyncio.CancelledError):
        pass


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while system_running:
            with lock:
                payload = {
                    "telemetry": telemetry,
                    "logs": system_logs[:6]
                }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.06)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print(" [✓] AI Driver Assistant DMS Online")
    print(" [✓] Target Vehicle: http://192.168.4.1 (ESP32-Safety-Car)")
    print(" [✓] Dashboard: http://127.0.0.1:8000")
    print("="*55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")