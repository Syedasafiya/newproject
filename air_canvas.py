import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

# ─── Download model if not present ──────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand landmark model (~8 MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded!")

# ─── MediaPipe Tasks Setup ───────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions        = mp_python.BaseOptions
HandLandmarker     = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode  = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = HandLandmarker.create_from_options(options)

# ─── Color Palette ──────────────────────────────────────────────────────────
COLORS = {
    "Purple": (180,  30, 220),
    "Blue":   ( 30, 120, 255),
    "Green":  ( 30, 210,  80),
    "Yellow": ( 30, 220, 220),
    "Red":    ( 30,  30, 240),
    "White":  (240, 240, 240),
    "Eraser": (  0,   0,   0),
}
COLOR_NAMES = list(COLORS.keys())

HEADER_H    = 80
SWATCH_W    = 110
BRUSH_SIZES = [4, 8, 14, 20]
BRUSH_LABELS = ["S", "M", "L", "XL"]
BRUSH_X0    = 10
BRUSH_Y0    = HEADER_H + 10

# ─── State ───────────────────────────────────────────────────────────────────
selected_color = "Green"
brush_idx      = 1
canvas         = None
xp, yp         = 0, 0
mode           = "IDLE"
stroke_count   = 0
snap_idx       = 0
drawing        = False

# ─── Helpers ─────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, -1)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, -1)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, -1)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, -1)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)

def get_touched_color(x, y):
    if not (10 <= y <= HEADER_H - 10):
        return None
    for i, name in enumerate(COLOR_NAMES):
        x0 = 240 + i * (SWATCH_W + 6)
        if x0 <= x <= x0 + SWATCH_W:
            return name
    return None

def get_touched_brush(x, y):
    if not (BRUSH_X0 <= x <= BRUSH_X0 + 54):
        return None
    for j in range(len(BRUSH_SIZES)):
        by = BRUSH_Y0 + 16 + j * 54 + 20
        if abs(y - by) < 22:
            return j
    return None

def fingers_up(lm_list):
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    return [lm_list[t][1] < lm_list[b][1] for t, b in zip(tips, bases)]

def draw_hand_skeleton(frame, landmarks, w, h):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, (255, 255, 255), -1)

def draw_ui(frame, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.putText(frame, "AI Air Canvas", (14, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)

    for i, name in enumerate(COLOR_NAMES):
        bgr = COLORS[name]
        x0  = 240 + i * (SWATCH_W + 6)
        y0, y1 = 10, HEADER_H - 10
        if name == selected_color:
            draw_rounded_rect(frame, x0-3, y0-3, x0+SWATCH_W+3, y1+3, 8, (255,255,255), -1)
        draw_rounded_rect(frame, x0, y0, x0+SWATCH_W, y1, 8, bgr, -1)
        lbl_color = (30,30,30) if name != "Eraser" else (200,200,200)
        cv2.putText(frame, name, (x0+6, y0+38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, lbl_color, 1, cv2.LINE_AA)

    panel_h = len(BRUSH_SIZES) * 54 + 16
    draw_rounded_rect(frame, BRUSH_X0, BRUSH_Y0,
                      BRUSH_X0+54, BRUSH_Y0+panel_h, 10, (28,28,40), -1)
    for j, (bsz, lbl) in enumerate(zip(BRUSH_SIZES, BRUSH_LABELS)):
        bx = BRUSH_X0 + 27
        by = BRUSH_Y0 + 16 + j*54 + 20
        if j == brush_idx:
            cv2.circle(frame, (bx, by), 20, (255,255,255), -1)
            cv2.putText(frame, lbl, (bx-8, by+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20,20,30), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (bx, by), 20, (80,80,100), -1)
            cv2.putText(frame, lbl, (bx-8, by+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200,200,200), 1, cv2.LINE_AA)

    badge_color = (30,200,100) if mode == "DRAW" else (50,150,255) if mode == "SELECT" else (80,80,100)
    bx2, by2 = w - 160, HEADER_H + 14
    draw_rounded_rect(frame, bx2, by2, bx2+145, by2+38, 8, badge_color, -1)
    cv2.putText(frame, f"Mode: {mode}", (bx2+10, by2+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:2d}", (w-160, HEADER_H+75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Strokes: {stroke_count}", (w-160, HEADER_H+100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Clear   [S] Screenshot   [ESC] Quit",
                (14, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130,130,150), 1, cv2.LINE_AA)
    return frame

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    global canvas, xp, yp, mode, selected_color, brush_idx, stroke_count, snap_idx, drawing

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    prev_time = time.time()
    ts_ms = 0

    print("✅ AI Air Canvas started!  C=clear  S=screenshot  ESC=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        now      = time.time()
        fps_val  = int(1 / max(now - prev_time, 1e-5))
        prev_time = now

        # ── Run detection ────────────────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms    += int(1000 / 30)
        detection = detector.detect_for_video(mp_image, ts_ms)

        if detection.hand_landmarks:
            landmarks = detection.hand_landmarks[0]
            lm_list   = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            x1, y1 = lm_list[8]   # index tip
            up      = fingers_up(lm_list)

            draw_hand_skeleton(frame, landmarks, w, h)

            if up[0] and up[1]:                        # ── SELECT mode
                mode    = "SELECT"
                drawing = False
                xp, yp  = 0, 0

                tc = get_touched_color(x1, y1)
                if tc:
                    selected_color = tc

                tb = get_touched_brush(x1, y1)
                if tb is not None:
                    brush_idx = tb

                cv2.circle(frame, (x1, y1), 12, (255,255,255), 2)

            elif up[0] and not up[1]:                  # ── DRAW mode
                mode      = "DRAW"
                color_bgr = COLORS[selected_color]
                thickness = BRUSH_SIZES[brush_idx]

                if y1 > HEADER_H:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if not drawing:
                        drawing = True
                        stroke_count += 1

                    if selected_color == "Eraser":
                        cv2.line(canvas, (xp, yp), (x1, y1), (0,0,0), thickness * 6)
                    else:
                        cv2.line(canvas, (xp, yp), (x1, y1), color_bgr, thickness * 2)

                    cv2.circle(frame, (x1, y1), thickness, color_bgr, cv2.FILLED)
                xp, yp = x1, y1

            else:                                      # ── IDLE
                mode    = "IDLE"
                xp, yp  = 0, 0
                drawing = False
        else:
            mode    = "IDLE"
            xp, yp  = 0, 0
            drawing = False

        # ── Merge canvas ─────────────────────────────────────────────────────
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask3   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame   = np.where(mask3 > 0, canvas, frame).astype(np.uint8)

        frame = draw_ui(frame, fps_val)
        cv2.imshow("AI Air Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('c'), ord('C')):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            stroke_count = 0
            print("🗑️  Canvas cleared")
        elif key in (ord('s'), ord('S')):
            fname = f"canvas_snap_{snap_idx:03d}.png"
            cv2.imwrite(fname, frame)
            snap_idx += 1
            print(f"📸 Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("👋 Bye!")

if __name__ == "__main__":
    main()