import numpy as np
import imutils
import cv2
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bounding-box", required=True,
	help="comma separated list of top, right, bottom, left coordinates of hand ROI")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


if not args.get("video", False):
	camera = cv2.VideoCapture(1)

else:
	camera = cv2.VideoCapture(args["video"])
(top, right, bot, left) = np.int32(args["bounding_box"].split(","))

a=0
while True:
	# grab the current frame
    (grabbed, frame) = camera.read()
    if args.get('video') and not grabbed:
        break
	# if we are viewing a video and we did not grab a frame, then we have reached the
	# end of the video    
	# resize the frame and flip it so the frame is no longer a mirror view
    frame = imutils.resize(frame, width=600)
    frame = cv2.flip(frame, 1)
    
    
    
    (frameH, frameW) = frame.shape[:2]
    roi = frame[top:bot, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)
    gY = cv2.convertScaleAbs(gY)
    dilated=cv2.dilate(gY.copy(),None,iterations=5)
    eroded=cv2.erode(dilated.copy(),None,iterations=5)
    #cv2.imshow("Sobel Y", gY)

    eroded=cv2.GaussianBlur(eroded,(7,7),0)
    #cv2.imshow('Eroded',eroded)

    (T,thresh)=cv2.threshold(eroded,50,255,cv2.THRESH_BINARY)
    #cv2.imshow('thresh',thresh)
    masked=cv2.bitwise_and(roi,roi,mask=thresh)
    cv2.imshow('Masked',masked)

    clone1=roi.copy()
    (cnts,_)=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(0,115),(frameW,115),(0,255,0),2)
    overlay=frame.copy()
    output=frame.copy()
    for c in cnts:
    	(x,y,w,h)=cv2.boundingRect(c)
        if w>45 and h <20:
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            
            

	    if y+(h/2) < 95:
	        
		cv2.rectangle(overlay,(280,80),(320,200),(0,255,0),-1)
		cv2.addWeighted(overlay,0.3,output,1-0.3,0,frame)
		cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame,'Accept',(280,70),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)
            if y+(h/2) > 95:

		cv2.rectangle(overlay,(280,80),(320,200),(0,0,255),-1)
		cv2.addWeighted(overlay,0.3,output,1-0.3,0,frame)
		cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
	        cv2.putText(frame,'Reject',(280,70),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)

                    




    cv2.imshow('frame',frame)
    cv2.imshow('counter',roi)

    key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



	
