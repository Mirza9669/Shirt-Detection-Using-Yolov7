from flask import Flask, render_template, Response
import cv2
import torch

# Load fine-tuned custom model
model = torch.hub.load('yolov7', 'custom', 'best.pt', force_reload=True, source='local', trust_repo=True)


app = Flask(__name__)

confidence_threshold = 0.01

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(12, 640)
    cap.set(12, 480)
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        results = model(frame)
        df = results.pandas().xyxy[0]
        for index, row in df.iterrows():
            name = row['name']
            confidence = row['confidence']
            label = f"{name} {confidence:.2f}"
            if row['confidence'] > confidence_threshold:
                cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 155, 0), 2)
                cv2.putText(frame, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)