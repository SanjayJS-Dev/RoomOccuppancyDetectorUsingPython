import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model=YOLO('yolov8s.pt')

area1=[(238,390),(218,402),(440,402),(432,390)]
area2=[(212,406),(188,420),(462,420),(445,406)]     

cv2.namedWindow('RGB')

camera=cv2.VideoCapture(2)

obj_file = open("obj_list.txt", "r")
data = obj_file.read()
class_list = data.split("\n")

entry = {}
people_in = set()
exit = {}
people_out = set()

tracker=Tracker()
while True:

    readed,frame = camera.read()
    if readed==False:
        break

    frame=cv2.resize(frame,(750,500))
    results=model.predict(frame)
    res=results[0].boxes.data
    px=pd.DataFrame(res).astype("float")
    list=[]
             
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    
    pid = tracker.update(list)
    for person in pid:
        xn1,yn1,xn2,yn2,id=person
        if(cv2.pointPolygonTest(np.array(area1,np.int32),(xn2,yn2),False) >= 0):
            entry[id] = (xn2,yn2)
            cv2.rectangle(frame,(xn1,yn1),(xn2,yn2),(0,0,255),2)
        if(id in entry) and (cv2.pointPolygonTest(np.array(area2,np.int32),(xn2,yn2),False) >= 0):
            cv2.rectangle(frame,(xn1,yn1),(xn2,yn2),(255,0,255),2)
            cv2.circle(frame,(xn2,yn2),5,(255,0,255),-1)
            cv2.putText(frame,str(id),(xn1,yn1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
            people_in.add(id)


        if(cv2.pointPolygonTest(np.array(area2,np.int32),(xn2,yn2),False) >= 0):
            exit[id] = (xn2,yn2)
            cv2.rectangle(frame,(xn1,yn1),(xn2,yn2),(255,0,255),2)
        if(id in exit) and (cv2.pointPolygonTest(np.array(area1,np.int32),(xn2,yn2),False) >= 0):
            cv2.rectangle(frame,(xn1,yn1),(xn2,yn2),(0,0,255),2)
            cv2.circle(frame,(xn2,yn2),5,(255,0,255),-1)
            cv2.putText(frame,str(id),(xn1,yn1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
            people_out.add(id)
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(50,50,50),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    print(len(people_in)-len(people_out))
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

camera.release()
cv2.destroyAllWindows()

