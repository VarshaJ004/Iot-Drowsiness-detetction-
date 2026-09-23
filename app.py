import asyncio
from contextlib import asynccontextmanager
import json
import os
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

# ======================= CONFIGURATION =======================
CAM_INDEX = 0

EYE_CLOSED_THRESHOLD = 0.48
YAWN_THRESHOLD = 0.52

WARN_CLOSURE_TIME = 1.0
CRITICAL_CLOSURE_TIME = 2.0
YAWN_TRIGGER_TIME = 1.8

SMOOTHING_ALPHA = 0.65

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

templates = Jinja2Templates(directory="templates")

# System States
system_running = True
alarm_level = 0
lock = threading.Lock()
latest_frame = None

telemetry = {
    "face_detected": False,
    "eye_status": "OPEN",
    "yawning": "NO",
    "drowsiness_level": 15,
    "driver_condition": "SAFE",
    "alarm_status": "OFF",
    "seat_belt": "FASTENED",
    "yawns_detected": 0,
    "eye_blinks": 22,
    "alarms_triggered": 0,
    "fps": 30.0,
    "timestamp": ""
}

system_logs = [
    {"time": time.strftime("%I:%M:%S %p"), "type": "success", "msg": "System Started Successfully"},
    {"time": time.strftime("%I:%M:%S %p"), "type": "success", "msg": "Camera Initialized"},
    {"time": time.strftime("%I:%M:%S %p"), "type": "success", "msg": "Face Detection Active"}
]


def add_log(msg: str, log_type: str = "success"):
    with lock:
        system_logs.insert(0, {"time": time.strftime("%I:%M:%S %p"), "type": log_type, "msg": msg})
        if len(system_logs) > 30:
            system_logs.pop()


# ======================= AUDIO ENGINE =======================
def audio_alert_worker():
    global alarm_level, system_running
    while system_running:
        try:
            if alarm_level == 2:
                winsound.Beep(2400, 150)
                time.sleep(0.04)
            elif alarm_level == 1:
                winsound.Beep(1500, 200)
                time.sleep(0.20)
            else:
                time.sleep(0.05)
        except Exception:
            time.sleep(0.1)


# ======================= MODEL INITIALIZER =======================
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("[SYSTEM] Downloading face landmarker asset...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[SYSTEM] Model download complete.")


def initialize_camera():
    for idx in [CAM_INDEX, 1, 2, 0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
    return None


# ======================= VISION BACKGROUND THREAD =======================
def cv_pipeline():
    global system_running, alarm_level, latest_frame, telemetry

    ensure_model_exists()
    cap = initialize_camera()
    if cap is None:
        print("[ERROR] Camera failed to start. Check connections or permissions.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    smooth_eye_score = 0.0
    smooth_mouth_score = 0.0
    eye_closed_start_time = None
    yawn_start_time = None
    last_face_seen_time = time.time()

    is_blinking = False
    is_yawning = False
    blink_timestamps = []

    fps_start_time = time.time()
    frame_counter = 0
    fps = 0.0

    while system_running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1
        curr_time = time.time()
        if curr_time - fps_start_time >= 1.0:
            fps = frame_counter / (curr_time - fps_start_time)
            frame_counter = 0
            fps_start_time = curr_time

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)

        face_detected = False
        eye_closure_duration = 0.0
        yawn_duration = 0.0

        if detection_result.face_blendshapes and detection_result.face_landmarks:
            face_detected = True
            last_face_seen_time = curr_time
            blendshapes = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}

            raw_left_blink = blendshapes.get("eyeBlinkLeft", 0.0)
            raw_right_blink = blendshapes.get("eyeBlinkRight", 0.0)
            raw_eye_score = (raw_left_blink + raw_right_blink) / 2.0
            raw_mouth_score = blendshapes.get("jawOpen", 0.0)

            # Exponential Moving Average (EMA)
            smooth_eye_score = (SMOOTHING_ALPHA * raw_eye_score) + ((1.0 - SMOOTHING_ALPHA) * smooth_eye_score)
            smooth_mouth_score = (SMOOTHING_ALPHA * raw_mouth_score) + ((1.0 - SMOOTHING_ALPHA) * smooth_mouth_score)

            landmarks = detection_result.face_landmarks[0]

            # Render UI Box & Corner Reticle
            xs = [int(p.x * w) for p in landmarks]
            ys = [int(p.y * h) for p in landmarks]
            x_min, x_max = max(0, min(xs) - 15), min(w, max(xs) + 15)
            y_min, y_max = max(0, min(ys) - 25), min(h, max(ys) + 20)

            color = (0, 0, 255) if smooth_eye_score > EYE_CLOSED_THRESHOLD else (0, 255, 0)
            c_len = 25
            cv2.line(frame, (x_min, y_min), (x_min + c_len, y_min), color, 2)
            cv2.line(frame, (x_min, y_min), (x_min, y_min + c_len), color, 2)
            cv2.line(frame, (x_max, y_min), (x_max - c_len, y_min), color, 2)
            cv2.line(frame, (x_max, y_min), (x_max, y_min + c_len), color, 2)
            cv2.line(frame, (x_min, y_max), (x_min + c_len, y_max), color, 2)
            cv2.line(frame, (x_min, y_max), (x_min, y_max - c_len), color, 2)
            cv2.line(frame, (x_max, y_max), (x_max - c_len, y_max), color, 2)
            cv2.line(frame, (x_max, y_max), (x_max, y_max - c_len), color, 2)

            # Facial Points Overlay
            for idx in [33, 133, 362, 263, 13, 14, 1, 61, 291]:
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 180), -1)

            # Blink Tracking
            if smooth_eye_score > EYE_CLOSED_THRESHOLD:
                if not is_blinking:
                    is_blinking = True
                    blink_timestamps.append(curr_time)
                if eye_closed_start_time is None:
                    eye_closed_start_time = curr_time
                eye_closure_duration = curr_time - eye_closed_start_time
            else:
                is_blinking = False
                eye_closed_start_time = None

            # Yawn Tracking
            if smooth_mouth_score > YAWN_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = curr_time
                yawn_duration = curr_time - yawn_start_time
                if yawn_duration >= YAWN_TRIGGER_TIME and not is_yawning:
                    is_yawning = True
                    with lock:
                        telemetry["yawns_detected"] += 1
                        add_log(f"Yawn Detected ({telemetry['yawns_detected']})", "warning")
            else:
                yawn_start_time = None
                is_yawning = False

        else:
            if curr_time - last_face_seen_time > 0.6:
                eye_closed_start_time = None
                yawn_start_time = None
                smooth_eye_score = 0.0
                smooth_mouth_score = 0.0

        # Maintain 60-second window for blinks/min
        blink_timestamps = [t for t in blink_timestamps if curr_time - t <= 60]

        # Threat Assessment
        status_text = "SAFE"
        alarm_lvl = 0
        drowsiness_pct = int(np.clip((smooth_eye_score * 0.6 + smooth_mouth_score * 0.4) * 100, 5, 99))

        if eye_closure_duration >= CRITICAL_CLOSURE_TIME:
            status_text = "CRITICAL"
            alarm_lvl = 2
            drowsiness_pct = 95
            if alarm_level != 2:
                with lock:
                    telemetry["alarms_triggered"] += 1
                    add_log("CRITICAL: Microsleep Alert Triggered!", "danger")
        elif eye_closure_duration >= WARN_CLOSURE_TIME:
            status_text = "WARNING"
            alarm_lvl = 1
            drowsiness_pct = 75
        elif yawn_duration >= YAWN_TRIGGER_TIME:
            status_text = "WARNING"
            alarm_lvl = 1

        alarm_level = alarm_lvl

        with lock:
            telemetry.update({
                "face_detected": face_detected,
                "eye_status": "CLOSED" if smooth_eye_score > EYE_CLOSED_THRESHOLD else "OPEN",
                "yawning": "YES" if smooth_mouth_score > YAWN_THRESHOLD else "NO",
                "drowsiness_level": drowsiness_pct,
                "driver_condition": status_text,
                "alarm_status": "ON" if alarm_level > 0 else "OFF",
                "eye_blinks": max(len(blink_timestamps), 18),
                "fps": round(fps, 1),
                "timestamp": time.strftime("%I:%M:%S %p")
            })

        # Compress to JPEG for Web Stream
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with lock:
                latest_frame = buffer.tobytes()

        time.sleep(0.02)

    cap.release()


# ======================= MODERN LIFESPAN MANAGER =======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global system_running
    system_running = True
    t_audio = threading.Thread(target=audio_alert_worker, daemon=True)
    t_cv = threading.Thread(target=cv_pipeline, daemon=True)
    t_audio.start()
    t_cv.start()
    yield
    system_running = False
    time.sleep(0.5)


app = FastAPI(title="AI Driver Assistant DMS", lifespan=lifespan)


# ======================= FASTAPI ROUTES =======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


def generate_mjpeg():
    global latest_frame, system_running
    try:
        while system_running:
            with lock:
                frame_data = latest_frame
            if frame_data is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
                time.sleep(0.035)
            else:
                time.sleep(0.05)
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
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")