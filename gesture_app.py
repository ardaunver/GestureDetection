"""Realtime hand gesture detection using the laptop webcam.

Uses MediaPipe's GestureRecognizer task for hand landmarks plus its
pretrained classifier (Fist, Open Palm, Pointing Up, Thumbs Up/Down,
Victory, I Love You). When the pretrained classifier doesn't recognize the
pose ("None"), a rule-based classifier built on the same hand landmarks
kicks in to cover extra gestures (OK Sign, Rock On, Call Me, Three, Four)
plus a generic finger-count fallback for anything else.
"""
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

MODEL_PATH = "models/gesture_recognizer.task"
CAMERA_INDEX = 0
MIN_GESTURE_CONFIDENCE = 0.5

GESTURE_LABELS = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Up": "Thumbs Up",
    "Thumb_Down": "Thumbs Down",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
}

HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MIDDLE_MCP, PINKY_MCP = 9, 17


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_finger_states(lm):
    """Returns {finger_name: is_extended} for thumb + 4 fingers."""
    wrist = lm[WRIST]
    states = {
        name: dist(wrist, lm[FINGER_TIPS[name]]) > dist(wrist, lm[FINGER_PIPS[name]])
        for name in FINGER_TIPS
    }
    pinky_mcp = lm[PINKY_MCP]
    states["thumb"] = dist(lm[THUMB_TIP], pinky_mcp) > dist(lm[THUMB_MCP], pinky_mcp) * 1.1
    return states


def classify_custom_gesture(lm, states):
    """Rule-based fallback classifier for gestures the pretrained model misses."""
    palm_size = dist(lm[WRIST], lm[MIDDLE_MCP])
    thumb_index_dist = dist(lm[THUMB_TIP], lm[FINGER_TIPS["index"]])

    if states["index"] and states["pinky"] and not states["middle"] and not states["ring"]:
        return "Rock On"
    if (states["thumb"] and states["pinky"] and not states["index"]
            and not states["middle"] and not states["ring"]):
        return "Call Me"
    if thumb_index_dist < palm_size * 0.35 and states["middle"] and states["ring"] and states["pinky"]:
        return "OK Sign"
    if (states["index"] and states["middle"] and states["ring"]
            and not states["pinky"] and not states["thumb"]):
        return "Three"
    if states["index"] and states["middle"] and states["ring"] and states["pinky"] and not states["thumb"]:
        return "Four"

    count = sum(states.values())
    return f"{count} Finger{'s' if count != 1 else ''}"


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in HAND_CONNECTIONS:
        start, end = connection.start, connection.end
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 128, 255), -1)


def label_for_hand(hand_landmarks, gestures):
    """Prefer the pretrained classifier's result; fall back to custom rules."""
    top = gestures[0] if gestures else None
    if top and top.category_name in GESTURE_LABELS and top.score >= MIN_GESTURE_CONFIDENCE:
        return GESTURE_LABELS[top.category_name], top.score

    states = get_finger_states(hand_landmarks)
    return classify_custom_gesture(hand_landmarks, states), (top.score if top else 0.0)


def main():
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check camera permissions/index.")

    start_time = time.time()
    prev_frame_time = start_time

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            for hand_landmarks, gestures in zip(result.hand_landmarks, result.gestures):
                draw_landmarks(frame, hand_landmarks)
                label, confidence = label_for_hand(hand_landmarks, gestures)

                wrist = hand_landmarks[0]
                h, w, _ = frame.shape
                x, y = int(wrist.x * w), int(wrist.y * h)
                text = f"{label} ({confidence:.0%})" if confidence else label
                cv2.putText(frame, text, (x - 20, y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(frame, text, (x - 20, y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1)

            now = time.time()
            fps = 1 / (now - prev_frame_time) if now != prev_frame_time else 0
            prev_frame_time = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
