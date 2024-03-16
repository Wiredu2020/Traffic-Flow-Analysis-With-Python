#!/usr/bin/env python
# coding: utf-8

# PhysicalCover
# PixCover
# metersPerPix
# TotDistance

# In[1]:


'''
Warning: To avoid errors, before you run any other codes,
make sure you run the first_frame() function
'''

import numpy as np
import cv2
import math
from datetime import datetime as dt
from datetime import timedelta

#creating the path to the video
v_string = 'C:\\Users\\JAH\\Desktop\\PROJECT\\WorkDone\\Vid\\VID_20200304_062316.mp4'



vidString = 'VID_20200304_062316.mp4'

#getting the dtails video(date and time the vidoe was taking) 
YMD= vidString[4:12]
HMS = vidString[13:19] 
Year = YMD[0:4]
Month=YMD[4:6]
Day=YMD[6:8]
Hour = HMS[0:2]
Minute=HMS[2:4] 
Second=HMS[4:6]
Y= int(Year)
Mn = int(Month)
D = int(Day)
H = int(Hour)
M = int(Minute)
S = int(Second)


#creating the video object using cv2
vid_stream = cv2.VideoCapture(v_string)


if (vid_stream.isOpened()==True):
    print("Your Video Is Loaded Successfully")
    print('=================================================')
else:
    print("Error In Loading Video")
    print('=================================================')

#getting details of frame
frameRate = vid_stream.get(cv2.CAP_PROP_FPS)
nframes = vid_stream.get(cv2.CAP_PROP_FRAME_COUNT)
print('=================================================')
print('ENTER THE START AND STOP TIMES OF VIDEO IF AVAILABLE')
print('=================================================')

if math.ceil(frameRate)>=40:
    steps = 20
else:
    steps = 9


#Borrowed function to convert the python datetime object to equivalent MATLAB date and time object
#source:
def datenum(d):
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)


    
    
#coverting time details
stop = dt(Y,Mn,D,H,M,S)
vidStop = datenum(stop)
dTime = stop

vidEndTimeSec =  nframes/frameRate
startDate = dTime -timedelta(seconds = vidEndTimeSec)
vidStart = datenum(startDate)


mTime = np.linspace(vidStart,vidStop,int(nframes))
mTime1 = np.arange(startDate, stop, timedelta(seconds =0.03373 )).astype(dt)
#DateString.append(dt.fromtimestamp(mTime))
#print(DateString)
#DateString = datestr(mTime)


#capturing the video frames and getting each frame from the captured video
vid= cv2.VideoCapture(v_string)
rat, f_frame = vid.read()


#setting a global points from the clicked area in the first frame
posList = []
points = []
Locations = []
xmin=0
ymin=0
xmax=0
ymax=0
xend=0
xcorner=0
ycorner=0
#function to get the points of the clicked area in the frame
def onMouse(event, x, y, flags, param):
    global posList,xmax,ymax
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        
    return posList

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        
    return posList

def last_Onmouse(event, x, y, flags, param):
    global Locations
    if event == cv2.EVENT_LBUTTONDOWN:
        Locations.append((x, y))
        
    return Locations

#resizing image if the image is too large
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#getting the first frame of the video, and other deatails like
#xmax,ymax,and others
def first_frame():
    global rescale_frame,vid,rat,f_frame,xmin,ymin,xmax,ymax,ycorner,xcorner,xend
    
    #rescalling the default frame size to diffrent size
    f_frame = rescale_frame(f_frame, percent=75)
    
    i=0
    while i<2:
        if rat == True :
            #converting the image to gray image
            gray = cv2.cvtColor(f_frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("F_frame",gray)
            #get the action action of the mouse
            cv2.setMouseCallback("F_frame", onMouse)
            #posNp = np.array(posList)
            if cv2.waitKey(25) & 0xFF ==ord('q'):
                break
        else:
            i+=1
    #finding the maximum position clicked in the x and y location
    for i in posList:
            if i[0] > xmax:
                xmax=i[0]
            if i[1] >ymax:
                ymax =i[1]
        
    xend= math.floor(xmax-(10/100)*xmax)            
    xcorner= math.floor((20/100)*xmax)
    ycorner= math.floor((20/100)*ymax)




first_frame()

# drawing the line on the frame, the eixt line for cars
#before you will run this codee make sure you have already run the previous codee
def set_barrier():
    global xmin,ymin,xmax,ymax,rat, f_frame
    i=0
    while i<2:
        if rat == True :
            gray = cv2.cvtColor(f_frame, cv2.COLOR_BGR2GRAY)
            
            #drawing an exit line for the cars
            barrier = cv2.line(gray,(xend,ymin),(xend,ymax),(0,0,0),2)
            
            #drawing tiled rectangle at the upper left
            #click on this rectangle if no car is in the current frame.
            Rectangle_fiiled = cv2.rectangle(barrier, (1, 1), (xcorner,ycorner), (0, 0, 255), -1)
            cv2.imshow("exit_point",Rectangle_fiiled)
            #posNp = np.array(posList)
            if cv2.waitKey(25) & 0xFF ==ord('q'):
                break
        else:
            i+=1
            
set_barrier()            

print('=================================================')
print('ENTER AN ESTIMATE OF THE NUMBER OF CARS THAT CAN FIT TOTAL DISTANCE')
print('=================================================')

#Estimated number of cars within a frame is 5
EstNumCarsInLength = 5
Lx = EstNumCarsInLength
hstat = math.ceil((2/100)*xmax)
hend = 3*EstNumCarsInLength+hstat-3
heights = np.linspace(hstat,hend,EstNumCarsInLength)
heights = [math.ceil(i) for i in heights]
e_list =[math.exp(n / Lx) for n in list(range(1, Lx+1))]
heights = [math.floor(e_list[t]*heights[t]) for t in range(0,len(heights))]


print('=================================================')
print('ENTER A KNOWN PHYSICAL DISTANCE FOR: PhysicalCover')
print('=================================================')


#Function to calculate the distance betwween two locations clicked.
#We use two big trees along the road size as point of reference to check the accuracy of this function.
# The length between those trees is aproximately 11.89m
def distance_covered_from_two_points():
    PhysicalCover = 11.89
    global vid,rat,f_frame,postList
    width = vid. get(cv2. CAP_PROP_FRAME_WIDTH )
    height = vid. get(cv2. CAP_PROP_FRAME_HEIGHT )
    i=0
    while i<2:
        if rat == True :
            gray = cv2.cvtColor(f_frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Total distance covered",gray)
            cv2.setMouseCallback("Total distance covered", on_mouse)
            
            if cv2.waitKey(25) & 0xFF ==ord('q'):
                break
        else:
            i+=1
   
    
    try:
        PixCover = points[1][0]-points[0][0]
        metersPerPix= PhysicalCover/PixCover
        TotDistance = (xmax/PixCover)*PhysicalCover
        TotDistanceKM = TotDistance/1000
        print('Total distance covered in Kilometers is:',TotDistanceKM)
    except:
          print("Something went wrong")
    return [TotDistanceKM,TotDistance,metersPerPix,PixCover]
            
distances = distance_covered_from_two_points()
TotDistanceKM = distances[0]
TotDistance = distances[1]
metersPerPix =distances[2]
PixCover=distances[3]

print('=========================================================')
print('CLICK A FEW LOCATIONS TO CHECK RECTANGLES...Right to Left')
print('=========================================================')

#crating a parameter to be use to create the width of cars in rectangular shapes.
hwidth = math.ceil((4/100)*xmax)

#Converting the blobs of the vehicles into a rectangular shape on a new video
#with a black background for tracking of them for futher analysis.


def rectangle():
    global hwidth,out
    empty = 0*f_frame
    #taking the time of a required frame in our video.
    t_msec = 1000*(7*60+29)
    vid.set(cv2.CAP_PROP_POS_MSEC, t_msec)
    rat, re_frame =vid.read()
    re_frame = rescale_frame(re_frame, percent=75)
    i=0
    while i<3:
        if rat == True :
            gray = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Click for Locations For Your Rectangles",gray)
            cv2.setMouseCallback("Click for Locations For Your Rectangles", last_Onmouse)
            if cv2.waitKey(25) & 0xFF ==ord('q'):

                break
        else:
            i+=1
    
    #getting the x and y cordinates of each car clicked.
    #We put them together as one array
    xco = []
    yco = []  
    for i in range(0,len(Locations)):
        xco.append(Locations[i][0]-hwidth)
        yco.append(Locations[i][1])
    Xn=len(xco)
    xdist = (2*hwidth)*np.ones(Xn)
    height = heights[0:Xn]
 
    #Creating the new video of the clicked blobs of car..
    for i in range(0,Xn):
        rect = cv2.rectangle(empty,(xco[i],yco[i]),(int(xco[i]+xdist[i]),yco[i]+height[i]),(255,255,255),-1)
    i=0
    while i<2:
        if rat == True :
            cv2.imshow("rect",empty)

            if cv2.waitKey(25) & 0xFF ==ord('q'):
                break
        else:
            i+=1

rectangle()  
    
vid_stream.release()
cv2.destroyAllWindows()


# In[ ]:





# In[3]:





# In[ ]:




