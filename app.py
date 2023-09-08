import cv2
import torch
from flask import Flask, request, render_template, Response
# from IPython.display import Image, display
import os
import math 

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load fine-tuned custom model
model = torch.hub.load('yolov7', 'custom', 'best.pt', force_reload=True, source='local', trust_repo=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         return "No file part"

    #     file = request.files['file']

    #     if file.filename == '':
    #         return "No selected file"

    #     if file:
    #         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #         file.save(filename)

    #         img = cv2.imread(filename)
    #         results = model(img)  
    #         df = results.pandas().xyxy[0]
            
    #         for index, row in df.iterrows():
    #             if df['confidence'][index] > 0.7:
    #                 confidence = df['confidence'][index]
    #                 label = f"Confidence: {confidence:.2f}"
    #                 cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 155, 0), 2)
    #                 cv2.putText(img, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 0), 2)
            
    #         annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + file.filename)
            
    #         cv2.imwrite(annotated_path,img)
    #         # display_image = Image(filename=annotated_path)
            
    #         return render_template('index.html', display_image=annotated_path)

    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    camera.set(15, 640)
    camera.set(18, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
    
        result = model(frame)
        df = result.pandas().xyxy[0]
        for index, row in df.iterrows():
            name = row['name']
            label = f"{name}"
            if row['confidence'] > 0.3:
                  cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0,0,255), 2)
                  cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 0), 2)
         
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace: boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
    
    
    
 
