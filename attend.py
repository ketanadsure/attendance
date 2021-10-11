import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'image'
image = []
person_name = []
mylist = os.listdir(path)
for ci in mylist:
    current_image = cv2.imread(f'{path}/{ci}')
    image.append(current_image)
    person_name.append(os.path.splitext((ci)[0]))
def faceencodings(iamge):
    encode_list = []
    for img in image:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings((img)[0])
        encode_list.append(encode)
    return encode_list
encodelistknown = faceencodings(image)
def attendance(name):
    with open("attendace.csv",'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = Line.Split(' , ')
            namelist.append(entry[0])
        if name not in namelist:
            time_now = datetime.now()
            tstring = time_now.strftime('%H:%M:%S')
            dstring = time_now.strftime('%d%m%y')
            f.writelines(f'{name},{tstring},{dstring}')

cap = cv2.VideoCapture(0)
while True:
    ret , frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25,0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    face_current_frame = face_recognition.faceLocations(faces)
    encode_current_frame = face_recognition.face_encodings(faces, face_current_frame)
    for encodeface, faceloc in zip(encode_current_frame,face_current_frame):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        matchIndex = np.argmin(facedis)
        if matches[matchIndex]:
            name = person_name[matchIndex].upper()
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(frame, attendance(name),(x1+6),(y2-6),cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),3)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
         break
cap.release()
cv2.destroyAllWindows()



