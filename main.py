import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
from hand_landmarks import HandLandmarksOnImage
from face_landmarks import FaceLandmarksOnImage
from constants import HAND_MODEL_PATH, FACE_MODEL_PATH, PADDING
import numpy as np
from handle_chewing import handle_chewing, stop_alarm


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


def main():
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    hand_detector = HandLandmarker.create_from_options(hand_options)
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    webcam_cap = cv2.VideoCapture(0)

    face = FaceLandmarksOnImage()
    hands = HandLandmarksOnImage()

    while True:
        _, frame = webcam_cap.read()

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_detection = hand_detector.detect(mp_image)
        face_detection = face_detector.detect(mp_image)

        drawn = hands.draw_hand_landmarks_on_image(frame, hand_detection)
        drawn = face.draw_face_landmarks_on_image(drawn, face_detection)

        mouth_box = face.locate_mouth(face_detection, frame.shape)

        if mouth_box:
            x_min, y_min, x_max, y_max = mouth_box
            cv2.rectangle(
                drawn,
                (x_min - PADDING, y_min - PADDING),
                (x_max + PADDING, y_max + PADDING),
                (0, 255, 0),
                2,
            )

            # Check if fingers are inside the mouth
            is_chewing = hands.check_if_chewing(
                mouth_box, hand_detection.hand_landmarks, frame.shape
            )

            if is_chewing:
                handle_chewing()

                # Make frame red
                overlay = drawn.copy()
                red_tint = np.full(overlay.shape, (0, 0, 255), dtype=np.uint8)
                cv2.addWeighted(red_tint, 0.3, overlay, 0.7, 0, overlay)
                drawn = overlay
            else:
                stop_alarm()

        cv2.imshow("cam", drawn)

        # (Q) to quit
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    webcam_cap.release()
    cv2.destroyAllWindows()
    stop_alarm()


if __name__ == "__main__":
    main()
