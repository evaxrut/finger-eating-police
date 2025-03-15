import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
from hand_landmarks import draw_hand_landmarks_on_image
from face_landmarks import draw_face_landmarks_on_image
from constants import HAND_MODEL_PATH, FACE_MODEL_PATH

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

    while True:
        _, frame = webcam_cap.read()

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_detection = hand_detector.detect(mp_image)
        face_detection = face_detector.detect(mp_image)

        drawn = draw_hand_landmarks_on_image(frame, hand_detection)
        drawn = draw_face_landmarks_on_image(drawn, face_detection)

        cv2.imshow("cam", drawn)

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    webcam_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
