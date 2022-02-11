import cv2
import face_recognition
import numpy as np
import os
from collections import Counter
from datetime import datetime

frameWidth, frameHeight = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# set your own Images folder
# insert facial images to the folder so the algorithm can detect the face
# otherwise the program will not work!!!
# also make sure the images only have one face each (only one face in one image)
path = "Images"

def getKnownNames(Names):
    file_names = os.listdir(path)
    for i in range(len(file_names)):
        Names[i] = os.path.splitext(file_names[i])[0]

    return Names

def getKnownEncodings(Encodings):
    for file in os.listdir(path):
        img = cv2.imread(f"{path}/{file}")
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Encodings.append(face_recognition.face_encodings(img)[0])

    return Encodings

def getCurEncodings_and_Points(img, CurEncodings, CurPoints):
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    Encodings = face_recognition.face_encodings(imgS)
    FaceLoc = face_recognition.face_locations(imgS)

    for encoding, faceLoc in zip(Encodings, FaceLoc):
        x1, y1, x2, y2 = faceLoc[3]*4, faceLoc[0]*4, faceLoc[1]*4, faceLoc[2]*4
        cv2.rectangle(Result, (x1,y1), (x2,y2), (0,0,0), 3)

        CurPoints.append(((x1+x2)//2-100, y2+25))
        CurEncodings.append(encoding)

    return CurEncodings, CurPoints

def markAttendance(Name):
     curNames = set()
     with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()

        for data in myDataList:
            curNames.add(data.split(sep=",")[0])

        if Name not in curNames:
            time = datetime.now()
            tString = datetime.strftime(time, "%H:%M:%S")
            f.writelines(f"\n{Name},{tString}")


KnownNames = getKnownNames({})
KnownEncodings = getKnownEncodings([])

while True:
    success, img = cap.read()
    Result = img.copy()

    CurEncodings, CurPoints = getCurEncodings_and_Points(img, [], [])

    for encoding, point in zip(CurEncodings, CurPoints):
        Matches = face_recognition.compare_faces(KnownEncodings, encoding)
        faceDis = face_recognition.face_distance(KnownEncodings, encoding)

        matchIndex = np.argmin(faceDis)

        if Matches[matchIndex]:
            cv2.putText(Result, KnownNames[matchIndex], point, cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            markAttendance(KnownNames[matchIndex])

        if Counter(Matches).most_common(1)[0][0] == False and Counter(Matches).most_common(1)[0][1] == len(Matches):
            cv2.putText(Result, "Unknown", point, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Webcam", Result)
    cv2.waitKey(1)
