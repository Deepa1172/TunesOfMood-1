import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import pandas as pd

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion=""

if not(emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

            np.save("emotion.npy", np.array([pred]))

            
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


moodNow = st.text("Want to listen to songs that resonate with your mood? Look no further! Tell us how you feel through your webcam! ")
    
webrtc_streamer(key="key", desired_playing_state=True,
            video_processor_factory=EmotionProcessor)

btn = st.button("Recommend songs")

if btn:
    if not(emotion):
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        dataset = pd.read_csv("spotify_songs.csv")
        columnNames = list(dataset.columns)
        dataset = dataset.drop(columns = ['track_id', 'playlist_genre', 'key', 'track_popularity', 'track_album_id', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_subgenre', 'duration_ms', 'language'])
        
        textDict = {'energetic' : 1, 'happy' : 2, 'calm' : 3, 'sad' : 4}
        if emotion not in textDict:
            st.text("Sorry! I haven't felt this mood yet :( ")

        elif(textDict[emotion] == 1):
            #making a dataset of energetic songs
            st.text_area("Energetic")
            energetic = dataset[(dataset['energy'] >= 0.5) & (dataset['energy'] <= 1.0)]
            st.dataframe(energetic.sample(frac = 0.5).reset_index())

        elif(textDict[emotion] == 2):
            #making  a dataset of happy songs
            st.text_area("\nHappy")
            happy = dataset[dataset['valence'] >= 0.5]
            st.dataframe(happy.sample(frac = 0.5).reset_index())

        elif(textDict[emotion] == 3):
            #making a dataset of calm songs
            st.text_area("\nCalm")
            calm = dataset[(dataset['energy'] < 0.5) & ((dataset['valence'] >= 0.33) & (dataset['valence'] <= 0.7)) & (dataset['tempo'] <= 95)]
            st.dataframe(calm.sample(frac = 0.5).reset_index())

        elif(textDict[emotion] == 4):
            #making a dataset of sad song
            st.text_area("\nSad")
            sad = dataset[(dataset['valence'] < 0.33)]
            st.dataframe(sad.sample(frac = 0.5).reset_index())

        else:
            print("Sorry! I haven't felt this mood yet :( ")