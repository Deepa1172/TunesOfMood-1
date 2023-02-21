import mediapipe as mp
import numpy as np
import cv2

cap = cv2.VideoCapture(0) #capturing video through webcam

name = input("Enter the name of the emotion: ")

holistic = mp.solutions.holistic #takes in the frame and return all the landmarks (keypoints) of the body
hands = mp.solutions.hands # for the visuals
holis = holistic.Holistic() # holistic class with holis obj
drawing = mp.solutions.drawing_utils #for the visuals 

X = []
dataSize = 0

while True:
    lst = []
    _, frm = cap.read() #reading the frame captured
    frm = cv2.flip(frm, 1) #flipping the frame; 1 -> left to right flip
    res =holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) #processing the frame and converting BGR to RGB, since cv2 reads in BGR format

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x) #storing all the face landmarks wrt to the nose, therefore no need for offset
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x) #storing all the points wrt index finger tip
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42): #if no hand, lst = 0 for all 42 points ( 21 pts for each axis)
                lst.append(0.0)
                
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x) #storing all the points wrt index finger tip
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42): #if no hand, lst = 0 for all 42 points ( 21 pts for each axis)
                lst.append(0.0)

        X.append(lst)
        dataSize = dataSize+1


    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS) #drawing the landmarks for face
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS) #drawing the landmarks for left hand
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS) #drawing the landmarks for right hand

    cv2.putText(frm, str(dataSize), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2) #text on the webcam window to show the no.of frames taken

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or dataSize>399: #exit window if waitkey pressed or frames > 400
        cv2.destroyAllWindows() #exiting the window
        cap.release() #release camera resource
        break


np.save(f"{name}.npy", np.array(X)) #f -> format
print(np.array(X).shape)
