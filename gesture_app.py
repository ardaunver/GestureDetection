"""Realtime hand gesture detection using the laptop webcam.

Uses MediaPipe's GestureRecognizer task (hand landmarks + a pretrained
gesture classifier) on frames pulled from OpenCV's VideoCapture.
"""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

MODEL_PATH = "models/gesture_recognizer.task"
CAMERA_INDEX = 0

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


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in HAND_CONNECTIONS:
        start, end = connection.start, connection.end
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 128, 255), -1)


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

                if gestures:
                    top = gestures[0]
                    label = GESTURE_LABELS.get(top.category_name, top.category_name or "Unknown")
                    confidence = top.score

                    wrist = hand_landmarks[0]
                    h, w, _ = frame.shape
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    text = f"{label} ({confidence:.0%})"
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
