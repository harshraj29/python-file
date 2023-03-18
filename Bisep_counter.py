import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template
app = Flask(__name__)

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
def calculate_angle(a,b,c):
    a=np.array(a) #First
    b=np.array(b) #Mid
    c=np.array(c) #End
    
    radian=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radian*180.0/np.pi)
    
    if angle>180.0:
        angle=360-angle
        
    return angle
@app.route('/')
def index():
    return render_template('index.html')

    
def gen_frames():
    angle=0
    cap=cv2.VideoCapture(0)
    counter=0
    stage=None
    ##setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret,frame=cap.read()
            
            #coluring image to RGB
            #image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=frame
            image.flags.writeable=False
            
            
            #make detection
            results=pose.process(image)
            
            #colouring image to BGR
            image.flags.writeable=True
            #image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            #Extract Landmark
            try:
                landmarks = results.pose_landmarks.landmark
                #get coordinate
                shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                #calculate angle
                angle=calculate_angle(shoulder,elbow,wrist)
                #Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                #curl counter
                if angle>160:
                    stage="Down"
                if angle<3 and stage=="Down":
                    stage="Up"
                    counter+=1
                    print(counter)
                
                
            except:
                pass
            #render curl counter
            #setup status box
            cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
            #rep data
            if(angle<3):
                cv2.putText(image,'move down',(15,12),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,'Reps',(15,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            
            cv2.putText(image,str(counter),
                        (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            
            #Render detection
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # cv2.imshow('Mediapipe Feed',image)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
            
    cap.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)