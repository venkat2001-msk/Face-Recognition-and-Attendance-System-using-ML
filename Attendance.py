import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path='imagesAttendance'  #Image Database Where Our Images are kept
images=[]                #Creating an images array to store our images
classNames=[]            #The names of the images are loaded here
myList= os.listdir(path)
print((myList))

for cl in myList:                                 #PART 1
    curImg=cv2.imread(f'{path}/{cl}')             #Here the images are loaded and appended to images array
    images.append(curImg)                         #Then the image names that contains the person's name are loaded
    classNames.append(os.path.splitext(cl)[0])    #without the .jpg file
print(classNames)

def findEncodings(images):                                   #Part 2
    encodeList=[]                                 #Here the person's images loaded into the database are encoded
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)     #First color is changed to rgb
        encode = face_recognition.face_encodings(img)[0]     #then the image's encodings are found and loaded into
                                                            #the encodeList array
        encodeList.append(encode)
    return encodeList

def markAttendance(name):                                          #Final Part
    with open('Attendance.csv','r+') as f:                # We use a csv file where we append the persons name with
        myDataList=f.readlines()                          #date and time it can be seen in excel or comma seperated
        nameList=[]                                       #format
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString=now.strftime("%m/%d/%Y, %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')
encodeListKnown=findEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)                                                           #Part 3
while True:                                                               #Here the system's webcam are turned on
                                                                          #we read the the images in the cam
    success, img=cap.read()                                               #we change it to a smaller version
                                                                          #then we change the color to rgb format
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)                             #There may be multiple people in the camera
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)                          #so we first encode the found faces
    facesCurFrame = face_recognition.face_locations(imgS)                 #and we compare it with the encodings of images
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) #in the person's database
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)  #the image captured at the nearest location
        #print(faceDis)                                                     #will be taken into account
        matchIndex=np.argmin(faceDis)


        if matches[matchIndex]:                                                         #Part4
            name=classNames[matchIndex].upper()                                  #if the images match we tone down the
            #print(name)                                                         #images are marked down using a
                                                                                 #rectangular box
            y1,x2,y2,x1=faceLoc                                                  #it works as a notification of
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4                                     #person to know whether face is detected
                                                                                 #or not
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
