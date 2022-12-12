from flask import Flask, render_template, Response
import cv2
import time
import pickle
import numpy as np
import subprocess
from pydub import AudioSegment
import IPython.display as ipd
from gtts import gTTS 

app = Flask(__name__)

AudioSegment.converter = "project"

# load the COCO class labels our YOLO model was trained on
LABELS = open("final_model/coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("final_model/yolov3.cfg", "final_model/yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize

def gen_frames():  # generate frame by frame from camera
    #function about gen_frame
    cap = cv2.VideoCapture(-1)
    frame_count = 0
    start = time.time()
    first = True
    frames = []
    while True:
        frame_count += 0
        # Capture frame-by-frameq
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        frames.append(frame)

        if frame_count == 300:
            break
            
        if ret:
            key = cv2.waitKey(1)
            if frame_count % 60 == 0:
                end = time.time()
                # grab the frame dimensions and convert it to a blob
                (H, W) = frame.shape[:2]
                # construct a blob from the input image and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes and
                # associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
                net.setInput(blob)
                layerOutputs = net.forward(ln)

                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []
                centers = []
                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > 0.5:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                            centers.append((centerX, centerY))
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
                colors= np.random.uniform(0,255, size=(len(LABELS),3))
                for i in idxs:
                    i=i
                    x,y,w,h = boxes[i]
                    classID=classIDs[i]
                    color=colors[classID]   
                    cv2.rectangle(frame, (round(x),round(y)), (round(x+w), round(y+h)),color,2)
                    label="%s: %.2f" %(LABELS[classID], confidences[i])
                    cv2.putText(frame, label, (x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1, color,2)
                cv2.waitKey(50)
                texts = []

                if len(idxs) > 0:
                    for i in idxs.flatten():
                        centerX, centerY = centers[i][0], centers[i][1]
                        if centerX <= W/3:
                            W_pos = "kiri "
                        elif centerX <= (W/3 * 2):
                            W_pos = "tengah "
                        else:
                            W_pos = "kanan "
                        
                        if centerY <= H/3:
                            H_pos = "atas "
                        elif centerY <= (H/3 * 2):
                            H_pos = "tengah "
                        else:
                            H_pos = "bawah "
                        texts.append(H_pos + W_pos + LABELS[classIDs[i]])
                print(texts)
                
                if texts:
                    description = ', '.join(texts)
                    tts = gTTS(description, lang='id')
                    tts.save('tts.mp3')
                    ipd.Audio('tts.mp3')
                    subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])

                



        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eksplore')
def eksplore():
    return render_template('explore.html')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/step')
def step():
    """Video streaming home page."""
    return render_template('step.html')



if __name__ == '__main__':
    app.run(debug=True)