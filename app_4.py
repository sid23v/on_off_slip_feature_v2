from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
MARKER_SIZE = 12
DISTANCE_THRESHOLD = 150

overlay_counter = 0
OVERLAY_FRAMES = 20

current_command = None
lock = threading.Lock()

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(flann_params, search_params)

reference_frame = None
ref_kp = None
ref_desc = None
live_pt = None
ref_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

latest_frame = None

def camera_reader():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        latest_frame = frame.copy()

threading.Thread(target=camera_reader, daemon=True).start()


def compute_keypoints_descriptors(gray):
    return sift.detectAndCompute(gray, None)


def reset_tracking():
    global reference_frame, ref_kp, ref_desc, live_pt, overlay_counter
    reference_frame = None
    ref_kp = None
    ref_desc = None
    live_pt = None
    overlay_counter = OVERLAY_FRAMES


def get_distance_color(distance):
    ratio = distance / DISTANCE_THRESHOLD
    if ratio < 0.5:
        return (0, 255, 0)
    elif ratio < 0.8:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def draw_scale_bar(display, distance):
    bar_height = 300
    bar_width = 30
    x1 = FRAME_WIDTH - 60
    y1 = 100
    x2 = x1 + bar_width
    y2 = y1 + bar_height

    cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 2)

    d = min(distance, DISTANCE_THRESHOLD)
    fill_h = int((d / DISTANCE_THRESHOLD) * bar_height)
    fill_top = y2 - fill_h

    color = get_distance_color(distance)

    cv2.rectangle(display, (x1 + 2, fill_top), (x2 - 2, y2 - 2), color, -1)

    cv2.putText(display, f"{distance}px", (x1 - 20, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)


def generate_frames():
    global reference_frame, ref_kp, ref_desc, live_pt, overlay_counter, current_command

    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # FAST encoding

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = latest_frame.copy()
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.circle(display, ref_center, MARKER_SIZE, (255, 0, 0), -1)

        distance = 0

        with lock:
            cmd = current_command
            current_command = None

        if cmd == "s":
            reference_frame = gray.copy()
            ref_kp, ref_desc = compute_keypoints_descriptors(reference_frame)
            live_pt = ref_center

        elif cmd == "r":
            reset_tracking()

        elif cmd == "q":
            break

        if reference_frame is not None:
            kp, desc = compute_keypoints_descriptors(gray)

            if desc is not None and len(kp) > 10:
                matches = flann.knnMatch(ref_desc, desc, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]

                if len(good) > 8:
                    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H is not None:
                        live_h = H @ np.array([[ref_center[0]], [ref_center[1]], [1.0]])
                        live_x = int(live_h[0] / live_h[2])
                        live_y = int(live_h[1] / live_h[2])
                        live_pt = (live_x, live_y)

                        dx = live_x - ref_center[0]
                        dy = live_y - ref_center[1]
                        distance = int(np.sqrt(dx * dx + dy * dy))

                        if distance > DISTANCE_THRESHOLD:
                            reset_tracking()

        if live_pt is not None:
            cv2.circle(display, live_pt, MARKER_SIZE, (0, 255, 0), -1)
            arrow_color = get_distance_color(distance)
            cv2.arrowedLine(display, ref_center, live_pt, arrow_color, 3, tipLength=0.3)

        draw_scale_bar(display, distance)

        if overlay_counter > 0:
            overlay = display.copy()
            red_layer = np.zeros_like(display)
            red_layer[:] = (0, 0, 255)
            cv2.addWeighted(red_layer, 0.4, overlay, 0.6, 0, overlay)
            overlay_counter -= 1
            display = overlay

        _, buffer = cv2.imencode('.jpg', display, jpeg_params)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-store\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

@app.route('/')
def index():
    return render_template_string("""
<html>
<head>
<title>Endoscopy Tracking</title>

<style>
body {
    background-color: black;
    text-align: center;
    color: white;
    font-family: Arial, sans-serif;
}

button {
    padding: 12px 20px;
    font-size: 18px;
    margin: 10px;
    background-color: #333;
    color: white;
    border: 2px solid white;
    border-radius: 8px;
    cursor: pointer;
}
button:hover {
    background-color: #555;
}

.container {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.button-row {
    display: flex;
    flex-direction: row;
    justify-content: center;
}
</style>

<script>
document.addEventListener("keydown", function(event) {
    if (event.key === "s") sendCmd("s");
    if (event.key === "r") sendCmd("r");
    if (event.key === "q") sendCmd("q");
});
function sendCmd(c) { fetch("/command/" + c); }
</script>

</head>
<body>

<h2>On/Off/Slip Feature - Test Application</h2>

<div class="container">
    <img src="/video_feed" width="800" height="600">
    
    <div class="button-row">
        <button onclick="sendCmd('s')">Set Reference (S)</button>
        <button onclick="sendCmd('r')">Reset (R)</button>
        <button onclick="sendCmd('q')">Quit (Q)</button>
    </div>
</div>

</body>
</html>
""")


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route('/command/<cmd>')
def command(cmd):
    global current_command
    with lock:
        current_command = cmd
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
