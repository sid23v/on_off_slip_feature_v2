from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading

CAMERA_INDEX = 1
MOTION_THRESHOLD = 45.0
OVERLAY_ALPHA = 0.25

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

feature_params = dict(
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

cap = cv2.VideoCapture(CAMERA_INDEX)

tracking_active = False
ref_gray = None
ref_pts = None
displacement_vec = np.zeros(2)

lock = threading.Lock()

app = Flask(__name__)

def generate_frames():
    global tracking_active, ref_gray, ref_pts, displacement_vec

    while True:
        with lock:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            movement_mag = 0.0

            if tracking_active and ref_pts is not None:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    ref_gray, gray, ref_pts, None, **lk_params
                )

                if next_pts is not None:
                    good_new = next_pts[status == 1]
                    good_ref = ref_pts[status == 1]

                    displacement_vec = np.mean(good_new - good_ref, axis=0)
                    movement_mag = np.linalg.norm(displacement_vec)

        
            if movement_mag > MOTION_THRESHOLD:
                overlay = frame.copy()
                overlay[:] = (0, 0, 255)
                frame = cv2.addWeighted(
                    overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0
                )

                cv2.putText(
                    frame, "MOTION LIMIT EXCEEDED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2
                )

            center = np.array([w // 2, h // 2])
            live_pos = center + displacement_vec.astype(int)

            cv2.circle(frame, tuple(center),
                       int(MOTION_THRESHOLD), (0, 255, 0), 1)

            cv2.drawMarker(frame, tuple(center),
                           (0, 255, 0),
                           cv2.MARKER_CROSS, 24, 2)

            reticle_radius = int(6 + min(movement_mag, 25))
            cv2.circle(frame, tuple(live_pos),
                       reticle_radius, (255, 0, 0), 2)

            if movement_mag < MOTION_THRESHOLD * 0.6:
                arrow_color = (0, 255, 0)
            elif movement_mag < MOTION_THRESHOLD:
                arrow_color = (0, 255, 255)
            else:
                arrow_color = (0, 0, 255)

            cv2.arrowedLine(frame,
                            tuple(center),
                            tuple(live_pos),
                            arrow_color, 3, tipLength=0.2)

            bar_x, bar_y = 20, h - 35
            bar_w, bar_h = 320, 14

            fill_ratio = min(movement_mag / MOTION_THRESHOLD, 1.0)
            fill_w = int(bar_w * fill_ratio)

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (255, 255, 255), 2)

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h),
                          arrow_color, -1)

            cv2.putText(frame,
                        f"Displacement: {movement_mag:.2f}px",
                        (bar_x, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Endoscopy Motion Monitor</title>
        <style>
            body {
                background-color: #111;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
            }
            #video-container {
                width: 720px;
                height: 540px;
                margin: auto;
                border: 2px solid #444;
            }
            img {
                width: 720px;
                height: 540px;
                object-fit: contain;
            }
            button {
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <h2>Endoscopy Motion Monitor</h2>
        <div id="video-container">
            <img src="/video_feed">
        </div>
        <br>
        <button onclick="fetch('/start')">Set Reference</button>
        <button onclick="fetch('/reset')">Reset</button>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_tracking():
    global tracking_active, ref_gray, ref_pts, displacement_vec
    with lock:
        ret, frame = cap.read()
        if ret:
            ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ref_pts = cv2.goodFeaturesToTrack(
                ref_gray, mask=None, **feature_params
            )
            displacement_vec[:] = 0
            tracking_active = True
    return "Reference set"

@app.route('/reset')
def reset_tracking():
    global tracking_active, ref_pts, displacement_vec
    with lock:
        tracking_active = False
        ref_pts = None
        displacement_vec[:] = 0
    return "Tracking reset"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
