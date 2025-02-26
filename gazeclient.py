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
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import cv2
import time
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
sys.path.append('..\\')
from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS
from gaze_logger import init_logger



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

    if str(sys.argv[1]) == "0":
        poly = joblib.load('train\\poly_player0.pkl')
    if str(sys.argv[1]) == "1":
        poly = joblib.load('train\\poly_player1.pkl')
    
    cudnn.enabled = True
    arch="ResNet50"
    batch_size = 1
    gpu = select_device('0', batch_size=batch_size)
    snapshot_path = "..\\models\\L2CSNet_gaze360.pkl"
   
    

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

      
    cap = cv2.VideoCapture(2)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()    
            start_fps = time.time()  
           
            faces = detector(frame)
            if faces is not None: 
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                    poly_pred = poly.predict([[pitch_predicted, yaw_predicted]])
                    print(poly_pred[0].encode())
                    if socket:
                        sGaze.send(poly_pred[0].encode())
                    
            if cv2.waitKey(1) & 0xFF == 27:
                break

if __name__ == "__main__":
    main()

