import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model=YOLO('yolov8s.pt')

area1=[(313,398),(304,412),(700,415),(690,402)]
area2=[(299,420),(283,440),(730,449),(710,426)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)       

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture(0)

my_file = open("obj_list.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:    
    ret,frame = cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    res=results[0].boxes.data
    px=pd.DataFrame(res).astype("float")
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            
            if(cv2.pointPolygonTest(np.array(area2,np.int32),(x2,y2),False) >= 0):
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(x2,y2),5,(255,0,255),-1)
                cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
        
      
            
            
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(50,50,50),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

