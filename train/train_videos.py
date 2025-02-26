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
sys.path.append('..\\..\\')
from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

def getData():
    cudnn.enabled = True
    arch="ResNet50"
    batch_size = 1
    gpu = select_device('0', batch_size=batch_size)
    snapshot_path = "..\\..\\models\\L2CSNet_gaze360.pkl"
   
    

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

    targets = ["Shrek","Robot","Tablet","Center"]

    pose = ""
    X = []
    y = []
    #l = ["./data/player0"]
    l = [".\\videos\\player1"]
    for ls in l:
        os.chdir(ls)    
        for file in glob.glob("*.avi"): 
            pose = [t for t in targets if t in file][0]   
            cap = cv2.VideoCapture(file)
            with torch.no_grad():
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break   
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
                            X += [[pitch_predicted, yaw_predicted]]
                            y += [pose]
                            
                            #draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                    #myFPS = 1.0 / (time.time() - start_fps)
                    #cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    #cv2.imshow(file,frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            cap.release()
            cv2.destroyAllWindows()
            print("Playback finished.")
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
    cudnn.enabled = True
    arch="ResNet50"
    batch_size = 1
    gpu = select_device('0', batch_size=batch_size)
    snapshot_path = "..\\..\\models\\L2CSNet_gaze360.pkl"
   
    

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
                    rbf_pred = rbf.predict([[pitch_predicted, yaw_predicted]])
                    print(poly_pred[0]+" "+ rbf_pred[0])
                    cv2.putText(frame, "poly: {}".format(poly_pred[0]), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, "rbf: {}".format(rbf_pred[0]), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                            



X, y = getData()
print(len(X))
print(len(y))
print("start train")
poly, rbf = training(X,y)
print("end train")
joblib.dump(poly, 'poly_player1.pkl')
joblib.dump(rbf, 'rbf_player1.pkl')

"""poly = joblib.load('poly_player0.pkl')
rbf = joblib.load('rbf_player0.pkl')
run(poly,rbf)"""