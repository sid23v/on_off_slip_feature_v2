import cv2
import numpy as np

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

MARKER_SIZE = 12
DISTANCE_THRESHOLD = 150

overlay_counter = 0
OVERLAY_FRAMES = 20  

PHASE_DS_WIDTH = 320
PHASE_DS_HEIGHT = 240

PHASE_RESPONSE_MIN = 0.05

PATCH_SIZE = 220
TEMPLATE_SCALES = [0.7, 0.85, 1.0, 1.15]
TEMPLATE_MATCH_THRESH = 0.55
SEARCH_RADIUS = 350 

MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
INLIER_RATIO_MIN = 0.20
RANSAC_REPROJ_THRESH = 8.0

sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(flann_params, search_params)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

reference_frame = None
ref_kp = None
ref_desc = None
ref_patch = None
ref_patch_center = None

ref_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

live_pt = None
prev_live_pt = None


def compute_keypoints_descriptors(gray):
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def reset_tracking():
    global reference_frame, ref_kp, ref_desc, ref_patch, ref_patch_center, live_pt, prev_live_pt, overlay_counter
    reference_frame = None
    ref_kp = None
    ref_desc = None
    ref_patch = None
    ref_patch_center = None
    live_pt = None
    prev_live_pt = None
    overlay_counter = OVERLAY_FRAMES
    print("Auto-reset triggered (distance threshold crossed).")

def get_distance_color(distance):

    ratio = distance / DISTANCE_THRESHOLD

    if ratio < 0.5:
        return (0, 255, 0)        # Green
    elif ratio < 0.8:
        return (0, 255, 255)      # Yellow
    else:
        return (0, 0, 255)        # Red


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
    fill_bottom = y2
    fill_top = y2 - fill_h

    color = get_distance_color(distance)

    cv2.rectangle(display, (x1 + 2, fill_top), (x2 - 2, fill_bottom - 2), color, -1)
    
    cv2.putText(display, f"{distance}px", (x1 - 20, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    cv2.putText(display, f"Max {DISTANCE_THRESHOLD}", (x1 - 50, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)


def extract_ref_patch(ref_gray):
    """Extract square patch centered at ref_center from the reference frame."""
    half = PATCH_SIZE // 2
    cx, cy = ref_center
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, ref_gray.shape[1] - 1)
    y1 = min(cy + half, ref_gray.shape[0] - 1)

    patch = ref_gray[y0:y1, x0:x1].copy()
    center_in_patch = (min(half, cx - x0), min(half, cy - y0))
    return patch, center_in_patch


def template_match_fallback(curr_gray, search_center=None):
    """
    Multi-scale template matching fallback using the fixed reference patch.
    If search_center provided, restrict search to square around it with SEARCH_RADIUS.
    Returns matched center coordinate in current frame (x,y) or None.
    """
    global ref_patch, ref_patch_center
    if ref_patch is None:
        return None

    best_val = -1.0
    best_pt = None
    h_ref0, w_ref0 = ref_patch.shape[:2]

    for scale in TEMPLATE_SCALES:
        w_ref = max(3, int(w_ref0 * scale))
        h_ref = max(3, int(h_ref0 * scale))
        scaled_patch = cv2.resize(ref_patch, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

        if search_center is not None:
            sx = int(max(0, search_center[0] - SEARCH_RADIUS))
            sy = int(max(0, search_center[1] - SEARCH_RADIUS))
            ex = int(min(curr_gray.shape[1], search_center[0] + SEARCH_RADIUS))
            ey = int(min(curr_gray.shape[0], search_center[1] + SEARCH_RADIUS))
            if ex - sx < w_ref or ey - sy < h_ref:
                continue
            search_img = curr_gray[sy:ey, sx:ex]
            top_left_offset = (sx, sy)
        else:
            search_img = curr_gray
            top_left_offset = (0, 0)

        try:
            res = cv2.matchTemplate(search_img, scaled_patch, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = (max_loc[0] + top_left_offset[0], max_loc[1] + top_left_offset[1])
            center_x = int(best_loc[0] + w_ref / 2)
            center_y = int(best_loc[1] + h_ref / 2)
            best_pt = (center_x, center_y)

    if best_val >= TEMPLATE_MATCH_THRESH:
        return best_pt
    else:
        return None


def phase_correlation_fallback(curr_gray):
    """
    Estimate global translation between reference_frame and current gray using phase correlation.
    Works on downscaled images for speed/robustness.
    Returns (shift_x_full, shift_y_full, response) where shift is how much current frame
    """
    global reference_frame
    if reference_frame is None:
        return None, None, 0.0

    ref_ds = cv2.resize(reference_frame, (PHASE_DS_WIDTH, PHASE_DS_HEIGHT), interpolation=cv2.INTER_AREA)
    cur_ds = cv2.resize(curr_gray, (PHASE_DS_WIDTH, PHASE_DS_HEIGHT), interpolation=cv2.INTER_AREA)

    ref_f = np.float32(ref_ds)
    cur_f = np.float32(cur_ds)
    ref_f -= np.mean(ref_f)
    cur_f -= np.mean(cur_f)

    try:
        window = cv2.createHanningWindow((PHASE_DS_WIDTH, PHASE_DS_HEIGHT), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(ref_f, cur_f, window)
    except Exception:
        try:
            shift, response = cv2.phaseCorrelate(ref_f, cur_f)
        except Exception:
            return None, None, 0.0

    dx_ds, dy_ds = shift 
    scale_x = FRAME_WIDTH / PHASE_DS_WIDTH
    scale_y = FRAME_HEIGHT / PHASE_DS_HEIGHT

    shift_x_full = dx_ds * scale_x
    shift_y_full = dy_ds * scale_y

    return shift_x_full, shift_y_full, response


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Camera not found!")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    display = frame.copy()
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = clahe.apply(gray_raw)

    cv2.circle(display, ref_center, MARKER_SIZE, (255, 0, 0), -1)

    distance = 0

    if reference_frame is not None:
        kp, desc = compute_keypoints_descriptors(gray)

        used_method = None
        matched_live_pt = None

        try:
            if desc is not None and len(kp) > 10 and ref_desc is not None and len(ref_kp) > 5:
                matches_fwd = flann.knnMatch(ref_desc, desc, k=2)
                good_fwd = []
                for m_n in matches_fwd:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_fwd.append(m)

                matches_bwd = flann.knnMatch(desc, ref_desc, k=2)
                best_bwd = {}
                for m_n in matches_bwd:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        if m.queryIdx not in best_bwd or m.distance < best_bwd[m.queryIdx][0]:
                            best_bwd[m.queryIdx] = (m.distance, m.trainIdx)

                mutual = []
                for m in good_fwd:
                    ref_idx = m.queryIdx
                    cur_idx = m.trainIdx
                    if cur_idx in best_bwd and best_bwd[cur_idx][1] == ref_idx:
                        mutual.append(m)

                if len(mutual) >= MIN_GOOD_MATCHES:
                    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)

                    if H is not None and mask is not None:
                        mask = mask.ravel()
                        inliers = int(np.sum(mask == 1))
                        inlier_ratio = inliers / max(1, len(mutual))

                        if inliers >= MIN_INLIERS and inlier_ratio >= INLIER_RATIO_MIN:
                            ref_center_h = np.array([[ref_center[0]], [ref_center[1]], [1.0]])
                            live_h = H @ ref_center_h
                            if abs(live_h[2, 0]) > 1e-6:
                                live_x = int(live_h[0, 0] / live_h[2, 0])
                                live_y = int(live_h[1, 0] / live_h[2, 0])
                                matched_live_pt = (live_x, live_y)
                                used_method = f"SIFT(H) inliers={inliers} ratio={inlier_ratio:.2f}"
        except Exception:
            matched_live_pt = None

        if matched_live_pt is None:
            search_center = prev_live_pt if prev_live_pt is not None else None
            template_pt = template_match_fallback(gray, search_center=search_center)
            if template_pt is not None:
                matched_live_pt = template_pt
                used_method = "Template"

        if matched_live_pt is None:
            sx, sy, resp = phase_correlation_fallback(gray)
            if sx is not None and resp is not None:
                
                if resp >= PHASE_RESPONSE_MIN and (not np.isnan(sx)) and (not np.isnan(sy)):
                    est_x = int(ref_center[0] + sx)
                    est_y = int(ref_center[1] + sy)
                    est_dist = int(np.sqrt(sx*sx + sy*sy))
                    if est_dist > DISTANCE_THRESHOLD:
                        reset_tracking()
                    else:
                        matched_live_pt = (est_x, est_y)
                        used_method = f"PhaseCorr(resp={resp:.3f})"

        if matched_live_pt is not None:
            live_pt = matched_live_pt
            prev_live_pt = live_pt

            dx = live_pt[0] - ref_center[0]
            dy = live_pt[1] - ref_center[1]
            distance = int(np.sqrt(dx*dx + dy*dy))

            if distance > DISTANCE_THRESHOLD:
                reset_tracking()
        else:
            pass

    if live_pt is not None:
        cv2.circle(display, live_pt, MARKER_SIZE, (0, 255, 0), -1)

        arrow_color = get_distance_color(distance)

        cv2.arrowedLine(display, live_pt, ref_center,
                        arrow_color, 3, tipLength=0.3)

    draw_scale_bar(display, distance)

    if overlay_counter > 0:
        overlay = display.copy()
        red_layer = np.zeros_like(display, dtype=np.uint8)
        red_layer[:] = (0, 0, 255)
        alpha = 0.4
        cv2.addWeighted(red_layer, alpha, overlay, 1 - alpha, 0, overlay)
        overlay_counter -= 1
        display = overlay

    cv2.putText(display, 'S: Set reference | R: Reset | Q: Quit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    cv2.imshow("Endoscopy SIFT Tracking", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        reference_frame = gray.copy()
        ref_kp, ref_desc = compute_keypoints_descriptors(reference_frame)

        ref_patch, ref_patch_center = extract_ref_patch(reference_frame)

        if ref_desc is None or len(ref_kp) < 5:
            print("Reference frame has insufficient features!")
            reset_tracking()
            continue

        live_pt = ref_center
        prev_live_pt = ref_center
        print("Reference set. Tracking enabled.")

    elif key == ord('r'):
        reset_tracking()

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
