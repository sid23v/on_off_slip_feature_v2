from flask import Flask, Response, render_template_string
import cv2
import numpy as np

app = Flask(__name__)

camera_index = 1
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Endoscopy Circle Detection</title>
    <style>
        body {
            margin: 0;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        #container {
            width: 960px;
            height: 720px;
        }
        img {
            width: 960px;
            height: 720px;
            display: block;
        }
    </style>
</head>
<body>
    <div id="container">
        <img src="{{ url_for('video_feed') }}">
    </div>
</body>
</html>
"""

def generate_frames():
    while True:
        success, img_color = cap.read()
        if not success:
            break

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        try:
            img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
            img_norm = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX)

            _, bin_img = cv2.threshold(
                img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if np.sum(bin_img == 255) < np.sum(bin_img == 0):
                bin_img = cv2.bitwise_not(bin_img)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
            if num_labels < 2:
                raise RuntimeError()

            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

            edges = cv2.Canny(mask, 50, 150)
            pts = np.column_stack(np.where(edges > 0))
            pts = np.flip(pts, axis=1)

            if len(pts) < 100:
                raise RuntimeError()

            x = pts[:, 0]
            y = pts[:, 1]

            A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
            B = x ** 2 + y ** 2
            C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

            cx, cy = C[0], C[1]
            radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).mean()

            cv2.circle(img_color, (int(cx), int(cy)), int(radius), (0, 255, 0), 6)
            cv2.circle(img_color, (int(cx), int(cy)), 8, (0, 0, 255), -1)

            cv2.putText(
                img_color,
                f"Radius: {int(radius)} px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception:
            pass

        ret, buffer = cv2.imencode(".jpg", img_color)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

