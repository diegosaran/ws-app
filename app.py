from flask import Flask, render_template, Response
import image  # Import the image.py file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', video_feed="/video_feed")

@app.route('/video_feed')
def video_feed():
    return Response(image.hand_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)