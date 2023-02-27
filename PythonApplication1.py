import face_recognition as fr
import numpy as np 
import cv2
import os 

#faces_path = "C:\\Users\\Alyon\\source\\repos\\PythonApplication1\\PythonApplication1\\faces\\known"
#unknown_path = "C:\\Users\\Alyon\\source\\repos\\PythonApplication1\\PythonApplication1\\faces\\unknown"
#test_path = "C:\\Users\\Alyon\\source\\repos\\PythonApplication1\\PythonApplication1\\faces\\test"

faces_path = ".\\faces\\known"
unknown_path = ".\\faces\\unknown"
test_path = ".\\faces\\test"

# Function to get face names, as well as face encodings
def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}\\{name}")
        face_encodings.append(fr.face_encodings(face)[0])

        face_names[i] = name.split(".")[0] # To remove ".jpg" or any other image extension
    
    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

#video = cv2.VideoCapture(0)

scl = 1

# while True:
  #  success, image = video.read()


unknown_face_names = os.listdir(unknown_path)
unknown_face_encodings = []

for i, name in enumerate(unknown_face_names):
   
    image = fr.load_image_file(f"{unknown_path}\\{name}", mode = 'RGB')
    #image = cv2.imread('f"{unknown_path}\\{name}"', cv2.IMREAD_COLOR)
    
    resized_image = cv2.resize(image, (int(image.shape[1]/scl), int(image.shape[0]/scl)))

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
     
    #cv2.imshow("frame", rgb_image)       
    #cv2.waitKey(0)
    
    face_locations = fr.face_locations(rgb_image)

    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    countUnknown = 0
    
    for face_encoding, face_location in zip(unknown_encodings, face_locations):
   
        result = fr.compare_faces(face_encodings, face_encoding, 0.4)

        flag = 0
          
        if True in result:
            
            names = face_names[result.index(True)] + ".jpg"
            
            top, right, bottom, left = face_location

            cv2.rectangle(rgb_image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
    
            cv2.putText(rgb_image, name, (left*scl, bottom*scl + 20), font, 0.8, (255, 255, 255), 1)
            
            cv2.imwrite(''f"{test_path}\\{names}"'', rgb_image)
       
            flag = 1 
    
    if flag == 0:
            
        names = "Unknown" + str(name) + ".jpg";

        top, right, bottom, left = face_location

        cv2.rectangle(rgb_image, (left*scl, top*scl), (right*scl, bottom*scl), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
    
        cv2.putText(rgb_image, name, (left*scl, bottom*scl + 20), font, 0.8, (255, 255, 255), 1)
            
        cv2.imwrite(''f"{test_path}\\{name}"'', rgb_image)
    
cv2.waitKey(1)