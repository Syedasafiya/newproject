# ✋ AI Air Canvas — Professional Hand Gesture Drawing App

Draw in the air using your webcam and hand gestures, powered by MediaPipe.

---

## 🚀 Setup

```bash
pip install -r requirements.txt
python air_canvas.py
```

---

## 🎨 Features

| Feature | Details |
|---|---|
| **7 Colors** | Purple, Blue, Green, Yellow, Red, White, Eraser |
| **4 Brush sizes** | S / M / L / XL |
| **Live FPS counter** | Top-right corner |
| **Stroke counter** | Tracks drawing strokes |
| **Mode indicator** | DRAW / SELECT / IDLE badge |
| **Screenshot** | Press `S` to save canvas |
| **Clear** | Press `C` to wipe canvas |

---

## 🤚 Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only up | **DRAW** mode — move to draw |
| ✌️ Index + Middle up | **SELECT** mode — hover over color/brush to switch |
| ✊ Fist / other | **IDLE** — lifts pen |

### Selecting Colors / Brush Size
- Raise **index + middle finger** together
- Move your index fingertip over any **color swatch** in the top bar to select it
- Move over the **left panel circles** (S / M / L / XL) to change brush size

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Clear canvas |
| `S` | Save screenshot (`canvas_snap_000.png`, etc.) |
| `ESC` | Quit |

---

## 📁 Files

```
ai_air_canvas/
├── air_canvas.py      ← Main application
├── requirements.txt   ← Dependencies
└── README.md          ← This file
```
