import mediapipe as mp
from mediapipe import solutions
import numpy as np
from mediapipe.framework.formats import landmark_pb2


class FaceLandmarksOnImage:
    def draw_face_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image

    def locate_mouth(self, detection_result, image_shape):
        face_landmarks_list = detection_result.face_landmarks
        if not face_landmarks_list:
            return None  # No face detected

        h, w = image_shape[:2]  # Get image dimensions

        mouth_indices = [
            61, 185, 40, 39, 37, 267, 269, 270, 409, 291,  # Outer lips
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324   # Inner lips
        ]
        for face_landmarks in face_landmarks_list:
            mouth_points = [
                (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in mouth_indices
            ]

            x_min = min(p[0] for p in mouth_points)
            y_min = min(p[1] for p in mouth_points)
            x_max = max(p[0] for p in mouth_points)
            y_max = max(p[1] for p in mouth_points)

            return (x_min, y_min, x_max, y_max)  # Bounding box around mouth

        return None  # No face detected