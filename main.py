from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import imutils


# Initialize Flask app
app = Flask(__name__)
drowsiness_alert = False

# Initialize pygame mixer for alarm
mixer.init()
mixer.music.load("siren-alert-96052.mp3")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Thresholds for drowsiness detection
thresh = 0.25
frame_check = 20
flag = 0
alarm_on = False  # To keep track of whether the alarm is playing

# Define eye landmark indices for mediapipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Simplified version
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Simplified version

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Video feed generator function
def gen_frames():
    global flag, alarm_on
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=450)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    leftEye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE]
                    rightEye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                 int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE]
                    
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    # Draw the eye contours
                    cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 0), 1)

                    # Check for drowsiness (EAR below the threshold)
                    if ear < thresh:
                        flag += 1
                        if flag >= frame_check and not alarm_on:  # Trigger alarm only if not already playing
                            alarm_on = True
                            mixer.music.play(-1)  # Play alarm in a loop
                            cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        flag = 0
                        if alarm_on:  # Stop the alarm if eyes are open
                            mixer.music.stop()
                            alarm_on = False

            # Encode the frame as JPEG and return it as part of the HTTP response
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('')
def index():
    # Render HTML page
    return render_template('index.html')

@app.route('video_feed')
def video_feed():
    # Return the response generated from gen_frames
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('get_alert_status')
def get_alert_status():
    global drowsiness_alert
    return jsonify({"alert": drowsiness_alert})

if __name__ == '__main__':
    app.run(debug=True)
