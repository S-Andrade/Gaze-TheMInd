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
      cap = cv2.VideoCapture(3)
      frame_width = int(cap.get(3))
      frame_height = int(cap.get(4))

      poses = {"free_Shrek": 0, "free_Robot": 0, "free_Center": 0, "free_Tablet":0, "glance_Shrek": 0, "glance_Robot": 0, "glance_Center": 0, "glance_Tablet":0 }
      data = {"Shrek":[], "Robot":[], "Center":[], "Tablet":[]}



      previous = "glance_"
      
      filen = 0

      for iteration in range(8):
            pos = [label for label, value in poses.items() if value < 1 and previous not in label]
            if pos == [] :
                  pos = [label for label, value in poses.items() if value < 1]
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

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(f"videos\\{player}\\{filename}_{pose}_{filen}.avi", fourcc, 20.0, (frame_width, frame_height))
            filen += 1
            input()

            i = 0
            
            while i < 100:
                  ret, frame = cap.read()
                  if not ret:
                        break
                  #cv2.imshow('Video', frame)
                  out.write(frame)              
                  i += 1                 
            playsound("bip.mp3")

            if "free_" in pose:
                  pose = pose[5:]
            if "glance_" in pose:
                  pose = pose[7:]
            print(pose)
            

           
            
    
      playsound("bip.mp3")
 
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])