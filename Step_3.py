#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import math
import Step_1
import numpy as np


# In[2]:


#calling of variables from step_1
mTime1      = Step_1.mTime1
steps = Step_1.steps
vidString = Step_1.v_string
videoObject = cv2.VideoCapture(vidString)
frameRate = Step_1.frameRate
nframes = Step_1.nframes
heights = Step_1.heights
hwidth = Step_1.hwidth
xend = Step_1.xend
rescale = Step_1.rescale_frame


# In[ ]:





# In[3]:



for i in range(1,2):
    ffolder = 'videos'+str(i)
    print('======== '+ffolder+' ==START!=====')
    areas=[]
    centriod =[]
    gfolder = 'C:\\Users\\JAH\\Desktop\\PROJECT\\WorkDone\\Vid\\' 
    pfolder = gfolder +'vidLR\\'
    savepath = pfolder
    savedata = 1
    pathv = pfolder + 'video4.avi'
    areas_Orig = [(2*hwidth*heights[t]) for t in range(0,len(heights))]
    f_frame = Step_1.f_frame
    emptyI=0*f_frame
     
    videoObject1 = cv2.VideoCapture(pathv)
    frameRate = videoObject1.get(cv2.CAP_PROP_FPS)
    nframes = videoObject1.get(cv2.CAP_PROP_FRAME_COUNT)
    dataM = []
    f_dataM =[]
    
    jj = 0
    ii = 0
    while ii<nframes:
        
        videoObject1.set(1, ii)
    
        ret, frame = videoObject1.read()
        if ret == True:
            #converting the colour of the frame to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #blurring the image for further proceessing
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            binaryImage = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
            
           # Copy the thresholded image.
            binaryImage_copy = binaryImage.copy()

            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = binaryImage_copy.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

            # Floodfill from point (0, 0)
            cv2.floodFill(binaryImage_copy, mask, (0,0), 255)
            
            binaryImage_copy = cv2.bitwise_not(binaryImage_copy)
            
            #equivalent way of getting the labels in the frame
            #Not needed in python tho...
            num_labels, labeledImage = cv2.connectedComponents(binaryImage_copy)
            
            #getting the blobs in the  frame using findContours
            contours , hie =cv2.findContours(binaryImage,cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_NONE)
            
            #number of blobs 
            num_blobs = len(contours)
            
            #drawig rectangles around  the blobs using drawContours
            draw = cv2.drawContours(frame,contours,-1,(0,0,255),2)
            
            
                
             #showing the cars in black and white frame        
            cv2.imshow("indentified blobs",draw)
            
            #
            
            if num_blobs ==0:
                ones =  np.ones(5)
                nan = ones*np.nan
                dataM.append([nan])
                areas.append(np.ones(1)*np.nan)
                centriod.append(np.ones(2)*np.nan)
                jj+=1
                ii+=1
            else:

                for i in range(0,num_blobs):
                    areas.append(cv2.contourArea(contours[i]))
                    M = cv2.moments(contours[i])
                    centriod.append([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
                ndata = [ [areas[t]]+ centriod[t] for t in range(0,len(areas)) ]
                dataM.append([[num_blobs]+ ndata[i+jj]+[mTime1[i+jj]] for i in range(0,num_blobs)])
                jj+=num_blobs
                ii+=1
                    
            
            
            if cv2.waitKey(1) == ord('b'):
                if num_blobs ==0:
                    ones =  np.ones(5)
                    nan = ones*np.nan
                    dataM.append([nan])
                    jj+=1
                
                for i in range(0,num_blobs):
                    areas.append(cv2.contourArea(contours[i]))
                    M = cv2.moments(contours[i])
                    centriod.append([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
                ndata = [ [areas[t]]+ centriod[t] for t in range(0,len(areas)) ]
                dataM.append([[num_blobs]+ ndata[i+jj]+[mTime1[i+jj]] for i in range(0,num_blobs)])
                break
            
            
for i in range(0,len(dataM)):
    for t  in range(0,len(dataM[i])):
        f_dataM.append(dataM[i][t])
f_dataM = np.matrix(f_dataM) 
   
    


cv2.destroyAllWindows()


# In[14]:


'''
PURPOSE: 
   After getting the area and centroid of each car 
   at each time, this code is used to arrange all 
   cars of a particular area from entry to exit in first 
   few (say 5) rows, the car that follows to be in the next 
   few (say 6-10) rows, etc.
 INPUT
 data -> data obtained from the script:
    TrackCars_manual_differentAreas_Analysis.m
 OUTPUT
 datt -> with vehicles arranged in order
 AUTHOR: Dr. Joseph K. Ansong
 DATE: 3 Dec. 2019.

'''
import numpy as np
def function_arrange_cars(data, halfwidth):
    hwidth = halfwidth;
    dataMM = data
  
  
    
    size =len(dataMM)
    i = 0
    jx=0
    
    #getting the first index of non_nan element.
    while i <= size:
        if str(dataMM[i,0]) != 'nan':
        #if dataMM[i,0]!=0:
            jx +=i
            break
        else:
            i+=1
    #deletiing the nans above the non_nan element.
    if jx>0:
        r = np.arange(jx)
        dataMM=np.delete(dataMM,r,0)
    
    
    #the rows and columns of the data
    (sx, ny) = dataMM.shape
    
    #% get the value and x-location of first non-nan
    fv=dataMM[0,1]
    
    #populate the very first non_data into the new data called datt
    datt = dataMM[0,0:ny]
    kj = 0
    ii=1
    
    #global index for a particular cars(or the indecies where a car was locatted)
    #this is used to delete the recoed of a particular car after storing them into the new data datt
    idxx =[0]
    
    #a loop to populate the datt based on the entry of a car, i.e daty contains records of cars after cars based 
    #on when they entered, this is done by using their area and thier x_centriod
    
    while ii < sx:
        
        if (dataMM[ii,1] == fv) &  (dataMM[ii,2] >= datt[-1,2]):
            kj+=1
            datt= np.append(datt,dataMM[ii,:],axis=0)
            idxx.append(ii)
          
            ii+=1
            
         #In case of a jam, it is possible to mistakenly click a location
         #which shows that the vehicle moved backwards. The ff is to use the
         #same position as the previous location. The last condition ensures 
         #that a car with the same dimensions will not be used since it also 
         #satisfies the second condition.
        
        elif (dataMM[ii,1]== fv) & (dataMM[ii,2] < datt[-1,2]) & (abs(dataMM[ii,2]-datt[-1,2]) <= hwidth ):
            
            
            kj+=1
            datt= np.append(datt,dataMM[ii-1,:],axis=0)
            idxx.append(ii)
            
            ii+=1
            
         #RETURN to beginning if code encounters vehicle with same dimension 
         #but it is a different vehicle   
            
        elif (dataMM[ii,1]==fv) & (dataMM[ii,2] <datt[-1,2]) & (abs(dataMM[ii,2]-datt[-1,2])> 5*hwidth):
           
            #wrong code, need to be corrected
            dataMM = np.delete(dataMM,idxx,0)
            #====index of first Non-NaN value
            
            size =len(dataMM)
            i = 0
            jx=0
            while i <= size:
                if str(dataMM[i,0]) != 'nan':
                #if dataMM[i,0]!=0:
                    jx +=i
                    break
                else:
                    i+=1
            if jx>0:
                r = np.arange(jx)
                dataMM=np.delete(dataMM,r,0)
                
            (sx, ny) = dataMM.shape
            

            fv=dataMM[0,1]

            datt= np.append(datt,dataMM[ii,:],axis=0)
            
            del idxx, kj
            kj = 0
            idxx =[0]
           
            ii=1
            
            
        
            #====index of first Non-NaN value
        
        elif ii==sx-1:
           
            dataMM = np.delete(dataMM,idxx,0)
            print(dataMM)

            #====index of first Non-NaN value
            size =len(dataMM)
            i = 0
            jx=0
            while i <= size:
                if str(dataMM[i,0]) != 'nan':
                #if dataMM[i,0]!=0:
                    jx +=i
                    break
                else:

                    i+=1
            if jx>0:
                r = np.arange(jx)
                dataMM=np.delete(dataMM,r,0)
            

            (sx, ny) = dataMM.shape
            fv=dataMM[0,1]
            
            print('new area taken: '+str(fv))
            
            datt = np.append(datt,dataMM[0,:],axis=0)
            
            del idxx, kj
            kj = 0
            idxx =[0]
            
            ii=1
            
            

        elif (len(dataMM)==0):
            #size(dataMM)
            
            break
        else:
            
            ii+=1
   
   
    return datt
            
            


# In[6]:


f_dataM.shape


# In[15]:


datt=function_arrange_cars(f_dataM,hwidth)
datt


# In[21]:


import cv2
img = cv2.imread('picture.jpg')
gr =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gr, (7, 7), 0)
threshold = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
con , hie =cv2.findContours(threshold,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
#number of blobs 
blobs = len(con)
dra = cv2.drawContours(img,con,-1,(0,0,255),2)
#cv2.imshow("Filtered ", dra)
cv2.waitKey(0)


# In[21]:


f_dataM.shape


# In[20]:


datt.shape


# In[ ]:




