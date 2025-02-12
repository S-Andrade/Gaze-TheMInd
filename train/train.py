from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import csv
import numpy as np
import time
import glob, os
import joblib
from sklearn import model_selection
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

def getData():
    pose = ""
    X = []
    y = []
    #l = ["./data/player0"]
    l = [".\\data\\player0"]
    w = 0
    i=0
    for ls in l:
        os.chdir(ls)    
        for file in glob.glob("*.tsv"):    
            with open(file, encoding="utf-8") as file:       
                tsv_file = csv.reader(file, delimiter="\t")
                for line in tsv_file:
                    #print(len(line))
                    
                    if len(line) == 1:
                        pose = line[0]
                    elif len(line) == 0:
                        pass
                    else:
                        #if len(line) == 39:
                        X.append(line)
                        y.append(pose)
        os.chdir("../..")
    X = np.array(X)
    y = np.array(y)
    return X,y

def training(X,y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

    return poly, rbf

def trainingAll(X,y):
    
    rbf = svm.SVC(kernel='rbf_player0', gamma=0.5, C=0.1).fit(X, y)
    poly = svm.SVC(kernel='poly_player0', degree=3, C=1).fit(X, y)

    return poly, rbf

def run(poly,rbf):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_mp = mp_pose.Pose()

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
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose_mp.process(rgb_frame)
        results_m = face_mesh.process(rgb_frame)
        
        temp = []

            
        if results_m.multi_face_landmarks:
            #break
            for face_landmarks in results_m.multi_face_landmarks:

                left = face_landmarks.landmark[468]
                #print([left.x, left.y, left.z])
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
                rbf_pred = rbf.predict([temp])
                print(poly_pred[0]+" "+ rbf_pred[0])
                cv2.putText(frame, "poly: {}".format(poly_pred[0]), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, "rbf: {}".format(rbf_pred[0]), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # Show the frame
        cv2.imshow('Body Tracking', frame)
        
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
                            



X, y = getData()
print("start train")
poly, rbf = training(X,y)
print("end train")
joblib.dump(poly, 'poly_player0.pkl')
joblib.dump(rbf, 'rbf_player0.pkl')
#os.chdir(".\\data")
poly = joblib.load('poly_player0.pkl')
rbf = joblib.load('rbf_player0.pkl')
run(poly,rbf)