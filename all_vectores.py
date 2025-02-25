import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face_mesh = mp.solutions.face_mesh
# Setup Face Mesh with iris tracking
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start video capture
cap = cv2.VideoCapture(3)

while True:
    temp = []
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    results_m = face_mesh.process(rgb_frame)

    if results.pose_landmarks:
                        
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get shoulder landmarks
            landmarks = results.pose_landmarks.landmark
            landmarks = landmarks[:9]

            i =0

            temp = []
            for l in landmarks:
                temp += [l.x, l.y, l.z]

            
    if results_m.multi_face_landmarks:
        #break
        for face_landmarks in results_m.multi_face_landmarks:

            left = face_landmarks.landmark[468]
            print([left.x, left.y, left.z])
            right = face_landmarks.landmark[473]
            temp += [left.x, left.y, left.z, right.x, right.y, right.z]



            

            # Draw iris landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            
    print(len(temp))
    # Show the frame
    cv2.imshow('Body Tracking', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
