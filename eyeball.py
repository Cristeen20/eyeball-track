import os
import cv2
import numpy as np
from flask import Flask
from flask_restful import Api, Resource, reqparse
import dlib
from PIL import Image
from io import BytesIO
import base64
pwd=os.getcwd()
import json
from flask import jsonify,make_response
from flask_cors import CORS,cross_origin
import urllib.request
from flask import Flask, json, render_template, request, redirect, jsonify

app = Flask(__name__, static_url_path='/', static_folder='')
CORS(app)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(shape,mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cx=0
    cy=0
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), 2)
        #print(cx,cy)
        
    except:
        pass
    return cx,cy
def nothing(x):
    pass



@app.route('/<string:p>', methods=['POST'])
def get(p):
        print(p)
        im = Image.open(BytesIO(base64.b64decode(p)))
        test_img=cv2.imread(im)
        flag=1
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(pwd+'\\shape_predictor_68_face_landmarks.dat')
        left = [36, 37, 38, 39, 40, 41]
        right = [42, 43, 44, 45, 46, 47]
        kernel = np.ones((9, 9), np.uint8)
        #cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
        sec1=2
        frame_count=0
        flag_log=-1
        log_count=0
        posflag=0

        if test_img is None:
                print('\n **********End of Video**********')
        else:
                        
            gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            rects, scores, idx = detector.run(gray, 1, 0.25)
            if(len(rects)==0):
                flag=0   
            else:         
                        #print(rects)
                        
                    #print(face)
                        #RECOGNITION...................................................................
                        

                        #ATTENTION.........................................................................................
                        shape = predictor(gray, rects[0])
                        shape = shape_to_np(shape)
                        mask = np.zeros(test_img.shape[:2], dtype=np.uint8)
                        mask = eye_on_mask(shape,mask, left)
                        mask = eye_on_mask(shape,mask, right)
                        mask = cv2.dilate(mask, kernel, 5)
                        eyes = cv2.bitwise_and(test_img, test_img, mask=mask)
                        mask = (eyes == [0, 0, 0]).all(axis=2)
                        eyes[mask] = [255, 255, 255]
                        mid = (shape[42][0] + shape[39][0]) // 2
                        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                        #threshold = cv2.getTrackbarPos('threshold', 'image')
                        _, thresh = cv2.threshold(eyes_gray, 90, 255, cv2.THRESH_BINARY)
                        thresh = cv2.erode(thresh, None, iterations=2) #1
                        thresh = cv2.dilate(thresh, None, iterations=4) #2
                        thresh = cv2.medianBlur(thresh, 3) #3
                        thresh = cv2.bitwise_not(thresh)
                        cx1,cy1=contouring(thresh[:, 0:mid], mid, test_img)
                        #print("left",cx1,cy1)
                        cx,cy=contouring(thresh[:, mid:], mid, test_img, True)
                        #print("right",cx,cy)
                        height, width = test_img.shape[:2]
                        for (xs, ys) in shape[36:48]:
                            #print("shape : ",xs,ys)
                            cv2.circle(test_img, (xs, ys), 2, (255, 0, 0), -1)  
                        att=" "
                        (xa,ya)=shape[42]
                        x0=xa
                        y0=ya
                        
                        (xa,ya)=shape[43]
                        x1=xa
                        y1=ya
                        
                        (xa,ya)=shape[44]
                        x2=xa
                        y2=ya
                        #print(y1,y2,cy)
                        (xa,ya)=shape[46]
                        x11=xa
                        y11=ya
                        (xa,ya)=shape[47]
                        x12=xa
                        y12=ya
                        (xa,ya)=shape[37]#left eye
                        x3=xa
                        y3=ya
                        (xa,ya)=shape[38]
                        x4=xa
                        y4=ya
                        (xa,ya)=shape[40]
                        x13=xa
                        y13=ya
                        (xa,ya)=shape[41]
                        x14=xa
                        y14=xa
                        #eyeclose
                        
                        if((abs(y2-y11)<1)and(abs(y1-y12)<1))or((abs(y3-y14)<1)and(abs(y4-y13)<1)):   #eye close  
                            #attention.append("DISTRACTED")#close
                            att="DISTRACTED"
                            #nam=facerec.recognise(gray,xr,yr,wr,hr)
                            #face_set.append(nam)
                        
                        if((cx<x1)and(cx<x2))or((cx<=x12)and(cx<x11)):
                            #attention.append("DISTRACTED")#right
                            att="DISTRACTED"
                            #nam=facerec.recognise(gray,xr,yr,wr,hr)
                            #face_set.append(nam)
                    #   very attentive.............
                        elif((cx>(x1))and(cx<(x2)))and((cx1>(x3))and(cx1<(x4))):
                            #attention.append("ATTENTIVE")
                            att="ATTENTIVE"
                            #nam=facerec.recognise(gray,xr,yr,wr,hr)
                            #face_set.append(nam)
                            
                        #left looking......right eye........
                        elif((cx1>x3)and(cx1>=(x4-1)))and((cx1>x14)and(cx1>=x13)):
                            #attention.append("DISTRACTED")#left
                            att="DISTRACTED"
                            #nam=facerec.recognise(gray,xr,yr,wr,hr)
                            #face_set.append(nam)
                            #print("lookleft",cx,x1,x2)
                        #right look...........lefteye......
        xr = rects[0].left()
        yr = rects[0].top()
        wr = rects[0].right()
        hr = rects[0].bottom()
        org=(xr,yr-9)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale=0.7
        color = (255,255, 0)
        test_img = cv2.putText(test_img, att, (xr,yr-3), font, fontScale,color, 2, cv2.LINE_AA, False)
        cv2.imshow('attention measurement ',test_img)
        if cv2.waitKey(1000) == ord('q'):#wait until 'q' key is pressed
            print("*******Exit*******")
            
        #return(att,cx1,cy1,cx,cy,flag)
        return{
            'Attention': att,
        }


img=cv2.imread("frame112.jpg")

@app.route('/')
def test():
        return 'Welcome'


    
if __name__ == '__main__':
    app.run(debug=True)

#print(atten(img))
                
