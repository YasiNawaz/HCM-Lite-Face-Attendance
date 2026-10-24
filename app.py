import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from facenet_pytorch import InceptionResnetV1

THRESHOLD = 0.65   # match ke liye minimum score

# ---------- Data files (database nahi, sirf files) ----------
DATA = Path("data")
DATA.mkdir(exist_ok=True)
EMP_FILE = DATA / "employees.csv"      # employee_id, name
ATT_FILE = DATA / "attendance.csv"     # check-in / check-out records
FACE_FILE = DATA / "faces.json"        # chehre ke numbers (embeddings)

EMP_COLS = ["employee_id", "name"]
ATT_COLS = ["employee_id", "name", "date", "check_in", "check_out", "source"]


def load_employees():
    if EMP_FILE.exists():
        return pd.read_csv(EMP_FILE, dtype=str).fillna("")
    return pd.DataFrame(columns=EMP_COLS)


def load_attendance():
    if ATT_FILE.exists():
        return pd.read_csv(ATT_FILE, dtype=str).fillna("")
    return pd.DataFrame(columns=ATT_COLS)


def load_faces():
    if FACE_FILE.exists():
        return json.loads(FACE_FILE.read_text())
    return []


# ---------- Models load (sirf ek baar) ----------
weights = Path("yolov8n-face.pt")
if not weights.exists():
    hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt", local_dir=".")
    Path("model.pt").rename(weights)
yolo = YOLO(str(weights))
embedder = InceptionResnetV1(pretrained="vggface2").eval()


# ---------- Face helpers ----------
def decode_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image sahi nahi hai")
    return img


def detect_faces(img):
    """Har face ka (x1, y1, x2, y2) list return karta hai."""
    result = yolo(img, conf=0.5, verbose=False)[0]
    boxes = []
    for b in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = [max(int(v), 0) for v in b]
        boxes.append((x1, y1, x2, y2))
    return boxes


def get_embedding(img, box):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    crop = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(crop).permute(2, 0, 1).float()
    t = ((t - 127.5) / 128.0).unsqueeze(0)
    with torch.no_grad():
        emb = embedder(t)[0].numpy()
    return emb / np.linalg.norm(emb)


def find_match(emb):
    """Sabse milta-julta employee dhoondo. Na mile to (None, None, score)."""
    best_id, best_score = None, -1.0
    for item in load_faces():
        score = float(np.dot(emb, np.array(item["embedding"], dtype=np.float32)))
        if score > best_score:
            best_id, best_score = item["employee_id"], score
    if best_id is None or best_score < THRESHOLD:
        return None, None, best_score
    emp = load_employees()
    name = emp.loc[emp["employee_id"] == best_id, "name"].iloc[0]
    return best_id, name, best_score


def one_face_or_error(img):
    faces = detect_faces(img)
    if len(faces) != 1:
        raise HTTPException(422, f"Frame mein exactly 1 face chahiye, {len(faces)} mile")
    return faces[0]


# ---------- API ----------
app = FastAPI(title="HCM Lite - Face Attendance")


class ImageIn(BaseModel):
    image: str


class RegisterIn(ImageIn):
    employee_id: str
    name: str


class AttendanceIn(ImageIn):
    action: str  # "check_in" ya "check_out"


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/register")
def register(body: RegisterIn):
    if not body.employee_id.strip() or not body.name.strip():
        raise HTTPException(400, "Employee ID aur naam dono likhein")
    img = decode_image(body.image)
    box = one_face_or_error(img)
    emb = get_embedding(img, box).astype(np.float32)

    # employees.csv mein employee save/update
    emp = load_employees()
    emp = emp[emp["employee_id"] != body.employee_id]
    emp = pd.concat([emp, pd.DataFrame([{"employee_id": body.employee_id, "name": body.name}])],
                    ignore_index=True)
    emp.to_csv(EMP_FILE, index=False)

    # faces.json mein chehre ke numbers save
    faces = load_faces()
    faces.append({"employee_id": body.employee_id, "embedding": emb.tolist()})
    FACE_FILE.write_text(json.dumps(faces))

    return {"message": f"{body.name} ({body.employee_id}) ka face register ho gaya"}


@app.post("/recognize")
def recognize(body: ImageIn):
    img = decode_image(body.image)
    faces = []
    for box in detect_faces(img):
        emp_id, name, score = find_match(get_embedding(img, box))
        faces.append({"box": list(box), "employee_id": emp_id,
                      "name": name or "Unknown Employee", "score": round(score, 2)})
    return {"faces": faces}


@app.post("/attendance")
def attendance(body: AttendanceIn):
    if body.action not in ("check_in", "check_out"):
        raise HTTPException(400, "action check_in ya check_out hona chahiye")
    img = decode_image(body.image)
    box = one_face_or_error(img)
    emp_id, name, _ = find_match(get_embedding(img, box))
    if emp_id is None:
        raise HTTPException(403, "Unknown Employee - attendance nahi lagi")

    now = datetime.now()
    today, t = now.date().isoformat(), now.strftime("%H:%M:%S")

    df = load_attendance()
    mask = (df["employee_id"] == emp_id) & (df["date"] == today)
    already = mask.any()

    if body.action == "check_in":
        if already:
            raise HTTPException(409, f"{name}: aaj ka check-in pehle se ho chuka hai")
        new_row = {"employee_id": emp_id, "name": name, "date": today,
                   "check_in": t, "check_out": "", "source": "camera"}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        msg = f"{name}: Check-in {t} ({today})"
    else:
        if not already:
            raise HTTPException(409, f"{name}: pehle check-in karna zaroori hai")
        if df.loc[mask, "check_out"].iloc[0] != "":
            raise HTTPException(409, f"{name}: aaj ka check-out pehle se ho chuka hai")
        df.loc[mask, "check_out"] = t
        msg = f"{name}: Check-out {t} ({today})"

    df.to_csv(ATT_FILE, index=False)
    return {"message": msg}


@app.get("/attendance/today")
def attendance_today():
    df = load_attendance()
    today = datetime.now().date().isoformat()
    return df[df["date"] == today].to_dict("records")