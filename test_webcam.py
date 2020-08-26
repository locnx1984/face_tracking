#!/usr/bin/python
#
# 3D Matching Node for Point Cloud
# Authors: Loc Nguyen 
# Solomon Technology Corp.
# Copyright - 2020 
# 
# The software contains proprietary information of Solomon Technology Corp.  
# It is provided under a license agreement containing restrictions on use and disclosure 
# and is also protected by copyright law. Reverse engineering of the software is prohibited. 
# 
# No part of this publication may be reproduced, stored in a retrieval system, 
# or transmitted in any form or by any means, electronic, mechanical, photocopying, recording or otherwise 
# without the prior written permission of Solomon Technology Corp. 
# 
# for testing: connect a webcam
 
from facenet_pytorch import MTCNN
import numpy as np
import torch
import cv2

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu

#FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)


#FaceMesh
from facemesh import FaceMesh
net = FaceMesh().to(gpu)
net.load_weights("facemesh.pth")  
  
def show_webcam():
    cam = cv2.VideoCapture(0)
    while True: 
        # Read the frame
        _, img = cam.read() 
        
        #Detect by FaceNet 
        faces, _ = mtcnn.detect(img) 

        if (faces is not None):
            # Draw the rectangle around each face 
            for (X1, Y1, X2, Y2) in faces:  
                x=int(X1)          
                y=int(Y1)          
                w=int(X2-X1)          
                h=int(Y2-Y1)          

                off_w=int(w/6)
                off_h=int(h/6)
                x1=max(0,int(x)-off_w)
                y1=max(0,int(y)-off_h)
                w1=min(w+2*(x-x1),img.shape[1]-1)
                h1=min(h+2*(y-y1),img.shape[0]-1) 

                # cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
                roi= img[y1:y1 + h1, x1:x1 + w1]

                roi2 = cv2.resize(roi, (192, 192))

                #FaceMesh
                detections = net.predict_on_image(roi2).cpu().numpy()
                detections.shape

                x, y  = [detections[:, 0], detections[:, 1]]
                Points=zip(x, y)
                for i,point in enumerate(Points):
                    x_=int(point[0]*w1/192)+x1
                    y_=int(point[1]*h1/192)+y1 
                    cv2.circle(img, (x_,y_), 1,(0,0,255))  
                    #Face Mesh
                    #226 left eye  446 right eye
                    #1 and 4: nose
                    #168 between eyes
                    if (i==1 or i==4 or i==168 or i==226 or i==446):
                        cv2.circle(img, (x_,y_), 3,(0,255,255))  

        # Display
        cv2.imshow('face', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    cv2.destroyAllWindows()
def main(): 
    show_webcam()
if __name__ == "__main__": 
    main()
    