import numpy as np
import imutils
import cv2
import os
print os.getcwd()
import argparse
from sklearn.metrics.pairwise import euclidean_distances

a=0
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bounding-box", required=True,
	help="comma separated list of top, right, bottom, left coordinates of hand ROI")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
cyy=[]
c1=[]
cent=[]
c2=[]
b1=0
b2=0

if not args.get("video", False):
	camera = cv2.VideoCapture(1)

else:
	camera = cv2.VideoCapture(args["video"])
(top, right, bot, left) = np.int32(args["bounding_box"].split(","))

back_sub=cv2.BackgroundSubtractorMOG()

while True:
   
    #print b
    
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
    bagmask=back_sub.apply(gray)
   # kernelo = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    
    kernelc = cv2.getStructuringElement(cv2.MORPH_RECT, (23,23))
   # mask = cv2.morphologyEx(bagmask, cv2.MORPH_OPEN, kernelo)
    mask = cv2.morphologyEx(bagmask , cv2.MORPH_CLOSE, kernelc)
#    cv2.imshow("bagmask", bagmask)
    cv2.imshow("mask", mask)

    (cnts,_)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(0,115),(frameW,115),(255,0,0),2)
    cv2.putText(roi,'IN',(10,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)
    cv2.putText(roi,'OUT',(200,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)
    cent=[]
   # print len(cnts)
    b=len(cnts)
    for c in cnts:
	(x,y,w,h)=cv2.boundingRect(c)


	if w*h>10000:
	    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
	    M = cv2.moments(c)
	    (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(roi,(cX,cY),3,(0,0,255),-1)
	    cent.append([cX,cY])
	    

	    	    


	    if len(c2)>0:
		    #print 'hi'
		a=a+1


	    if cY>113 and cY<117:
		c1.append([cX,cY])
		if len(c1)>0:
		    c2=c1
		    
		c1=[]
		
		

		
		    
	    if a>5:
   		print cent
		print c2
		
		dist=euclidean_distances(c2,cent)
		    
		    #print dist
		i=np.argmin(dist)
		    #print i
		        
		if cent[i][1]-c2[0][1] > 0:
	            print 'in'
		    b1=b1+2
		    print b1
		    
		if cent[i][1]-c2[0][1] < 0:
	            print 'out'
		    b2=b2+1
		    #cv2.putText(roi,str(b2),(210,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)
		a=0
		c2=[]
		    
		    #continue
		
		
		#cv2.line(frame,(0,115),(frameW,115),(0,255,255),2)


	    if cY>109 and cY<120:
		cv2.line(frame,(0,115),(frameW,115),(0,255,255),2)


	if w*h>2000 and w*h<10000:
	    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
	    M = cv2.moments(c)
	    (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(roi,(cX,cY),3,(0,0,255),-1)
	    cent.append([cX,cY])
	    

	    	    


	    if len(c2)>0:
		    #print 'hi'
		a=a+1


	    if cY>113 and cY<117:
		c1.append([cX,cY])
		if len(c1)>0:
		    c2=c1
		    
		c1=[]
		
		

		
		    
	    if a>5:
   		print cent
		print c2
		
		dist=euclidean_distances(c2,cent)
		    
		    #print dist
		i=np.argmin(dist)
		    #print i
		        
		if cent[i][1]-c2[0][1] > 0:
	            print 'in'
		    b1=b1+1
		    print b1
		    
		if cent[i][1]-c2[0][1] < 0:
	            print 'out'
		    b2=b2+1
		    #cv2.putText(roi,str(b2),(210,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),4)
		a=0
		c2=[]
		    
		    #continue
		
		
		#cv2.line(frame,(0,115),(frameW,115),(0,255,255),2)


	    if cY>109 and cY<120:
		cv2.line(frame,(0,115),(frameW,115),(0,255,255),2)




	cv2.putText(roi,str(b1),(50,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),4)
	cv2.putText(roi,str(b2),(270,400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),4)
	cv2.imshow('counter',roi)
       

    key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

