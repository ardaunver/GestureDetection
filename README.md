# Hand Gesture Detection

Realtime hand gesture detection using your laptop webcam, built with OpenCV
and MediaPipe's `GestureRecognizer` task.

Recognizes: Closed Fist, Open Palm, Pointing Up, Thumbs Up, Thumbs Down,
Victory, and I Love You.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
python3 gesture_app.py
```

Press `q` to quit.

On macOS, the first run will prompt you to grant camera access to your
terminal app (System Settings > Privacy & Security > Camera). Approve it and
re-run the script.

## Notes

- `models/gesture_recognizer.task` is Google's pretrained MediaPipe gesture
  model (~8MB), committed directly so no extra download step is needed.
- Change `CAMERA_INDEX` in `gesture_app.py` if you have multiple cameras.
