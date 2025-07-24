import cv2
import numpy as np

cap = cv2.VideoCapture("C:/Users/HP/Downloads/video.mp4")
print("cap", cap)

min_width_rect = 80
min_height_rect= 80

count_line_position = 550



#Initialise algo 
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx= x+x1
    cy= y+y1
    return cx,cy

detect =[]
offset = 6

counter = 0



while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    
    # Applying on each frame
    img_subs = algo.apply(blur)
    dilat = cv2.dilate(img_subs,np.ones((5,5)))    
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilaeada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernal)
    dilaeada = cv2.morphologyEx(dilaeada,cv2.MORPH_CLOSE,kernal)
    counter,h= cv2.findContours(dilaeada, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #v2.imshow("Detected",dilaeada)
    cv2.line(frame,( 25, count_line_position),(1200, count_line_position),(255,127,0),3)
    
    for (i,c) in enumerate(counter):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
        
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame,center,4 ,(0,0,255),-1)
        
        for (x,y) in detect:
            if (y<count_line_position+offset) and (y>count_line_position-offset):
                counter+=1
            cv2.line(frame,( 25, count_line_position),(1200, count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("Vehicle counter"+str(counter))
            
    cv2.putText(frame,"Vehicler"+str(counter),(450,70), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
            
            
    
    
    frame = cv2.resize(frame, (700,450))
    cv2.imshow("frame", frame)
    k=cv2.waitKey(25)   # 0 means static - 25 means in n25 sec how many frames are getting desplayed
    if k== ord("q") & 0xFF:
        break
cap.release()
cv2.destroyAllWindows()