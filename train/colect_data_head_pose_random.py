import time
import csv
from playsound import playsound
import sys
import random
import numpy as np
import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

def main(player, filename):    
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

      poses = {"free_Shrek": 0, "free_Robot": 0, "free_Center": 0, "free_Tablet":0, "glance_Shrek": 0, "glance_Robot": 0, "glance_Center": 0, "glance_Tablet":0 }
      data = {"Shrek":[], "Robot":[], "Center":[], "Tablet":[]}



      previous = "glance_"

      for iteration in range(32):
            pos = [label for label, value in poses.items() if value < 4 and previous not in label]
            if pos == [] :
                  pos = [label for label, value in poses.items() if value < 4]
            pose = random.choice(pos)
            poses[pose] += 1
            if "free_" in pose:
                  previous = pose[5:]
            if "glance_" in pose:
                  previous = pose[7:]
            print(pose)
            f = pose+".mp3"
            playsound(f)
            print(poses)

            i = 0
            data_pose = []
            while i < 100:
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
                                    
                              data_pose.append(temp)
                              print(temp)
      
                                    
                  cv2.imshow('Body Tracking', frame)
                  i += 1
                  print(i)                
                                    
                              
            if "free_" in pose:
                  pose = pose[5:]
            if "glance_" in pose:
                  pose = pose[7:]
            print(pose)
            

            data[pose] += data_pose
            
    
      playsound("bip.mp3")
      filename = "data/" + player + "/"+ filename + ".tsv"
      with open(filename, 'w', encoding='utf8', newline='') as tsv_file:
                  tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                  for pose in data.keys():
                        tsv_writer.writerow([pose])
                        for line in data[pose]:
                              tsv_writer.writerow(line)
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])