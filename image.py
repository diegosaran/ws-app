import cv2
import mediapipe as mp
import numpy as np
import math

def hand_tracking():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    mask_layer = 255 * np.ones((720, 1280, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, image = cap.read()

            if not success:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )

                    thumb_tip = None
                    index_tip = None
                    for id, lm in enumerate(hand_landmarks.landmark):
                        if id == 4:
                            thumb_tip = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                        elif id == 8:
                            index_tip = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))

                    if thumb_tip and index_tip:
                        distance = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
                        if distance < 30:
                            cv2.circle(mask_layer, index_tip, radius=10, color=(0, 0, 255), thickness=-1)
                        elif distance > 170:
                            mask_layer = 255 * np.ones((720, 1280, 3), dtype=np.uint8)

            image_masked = cv2.bitwise_and(image, mask_layer)
            ret, frame = cv2.imencode('.jpg', image_masked)
            if not ret:
                break
            data = frame.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')