
import socket           
import sys
import joblib
from gaze_logger import init_logger
import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')



def main():
    logger = init_logger(str(sys.argv[1]), f"gaze_{str(sys.argv[1])}.log") 
    socket = False
    if sys.argv[1] == "--socket":
        socket = True
        try:
            logger.log_info("Connecting to DecisionMaker...")
            sGaze = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
            sGaze.connect(('127.0.0.1', 50002))
            logger.log_info("Connected to DecisionMaker.")
        except ConnectionRefusedError:
            logger.log_error("Connectionto DecisionMaker Refused.")
        except Exception as e:
                logger.log_exception("An unexpected error occurred while connecting to DecisonMaker", e)
                raise

        msgaze = "Client " + str(sys.argv[1]) + " gaze"
        sGaze.send(msgaze.encode())
        logger.log_message("Identify participant gauze", msgaze)
    
    try:
        logger.log_info("Initialize MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Initialize drawing utility
        mp_drawing = mp.solutions.drawing_utils

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

        logger.log_info("Initialize MediaPipe Pose.")
    except Exception as e:
            logger.log_exception("An unexpected error occurred while Initialize MediaPipe Pose", e)
            raise
    
    try:
        logger.log_info("Start video capture...")
        cap = cv2.VideoCapture(2)
        logger.log_info("Start video capture.")
    except Exception as e:
        logger.log_error("An unexpected error occurred while Start video capture", e)
        raise

    try:
        logger.log_info("Loading SVM model...")
        poly = joblib.load('poly.pkl')
        logger.log_info("SVM model loaded successfully.")
    except FileNotFoundError:
        logger.log_error("SVM model file not found.")
        raise
    except Exception as e:
        logger.log_error("An unexpected error occurred while loading the SVM model", e)
        raise

    while True:
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
            landmarks = landmarks[:13]

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
                
            
        poly_pred = poly.predict([temp])
        print(poly_pred[0].encode())
        logger.log_message("Current gaze target", poly_pred[0])
        logger.log_message("vector",str(temp) )
        if socket:
            sGaze.send(poly_pred[0].encode())
        
        cv2.imshow('Body Tracking', frame)
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

