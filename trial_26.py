import cv2
import numpy as np

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

MARKER_SIZE = 12
DISTANCE_THRESHOLD = 100

overlay_counter = 0
OVERLAY_FRAMES = 20  

PHASE_DS_WIDTH = 320
PHASE_DS_HEIGHT = 240

PHASE_RESPONSE_MIN = 0.05

PATCH_SIZE = 220
TEMPLATE_SCALES = [0.7, 0.85, 1.0, 1.15]
TEMPLATE_MATCH_THRESH = 0.55
TEMPLATE_STRONG_BONUS = 0.15
SEARCH_RADIUS = 350 

MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
INLIER_RATIO_MIN = 0.20
RANSAC_REPROJ_THRESH = 8.0

MIN_KP_CURRENT = 30
FEATURE_LOSS_FRAMES = 4
feature_loss_counter = 0

ENTROPY_MIN = 3.5
LOW_ENTROPY_FRAMES = 4
low_entropy_counter = 0

# Vertical lift
SCALE_MIN = 0.95
SCALE_MAX = 1.75

last_scale_est = 1.0   # AXIAL DISPLAY

# Consensus / confidence gating
CONFIDENCE_MIN = 0.55
CONFIDENCE_STRONG = 0.75
CONSENSUS_DIST = 40
CONSENSUS_MIN_METHODS = 2
LOST_FRAMES_MAX = 3

# Optical flow quality threshold
FLOW_ERR_MAX = 12.0

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
prev_gray = None

lost_counter = 0

# ======= feature-rich gating variables & thresholds =======
was_feature_rich = False                        # True only when a valid (feature-rich) reference was set
FEATURE_RICH_KP_MIN = 45                        # min keypoints to be considered feature-rich
FEATURE_RICH_ENTROPY_MIN = 4.0                  # min entropy to be considered feature-rich
# ==========================================================


def clamp01(value):
    return max(0.0, min(1.0, value))


def pt_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def in_frame(pt):
    return 0 <= pt[0] < FRAME_WIDTH and 0 <= pt[1] < FRAME_HEIGHT


def compute_keypoints_descriptors(gray):
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def reset_tracking():
    global reference_frame, ref_kp, ref_desc, ref_patch, ref_patch_center
    global live_pt, prev_live_pt, overlay_counter
    global feature_loss_counter, low_entropy_counter, last_scale_est
    global was_feature_rich, prev_gray, lost_counter

    reference_frame = None
    ref_kp = None
    ref_desc = None
    ref_patch = None
    ref_patch_center = None
    live_pt = None
    prev_live_pt = None
    overlay_counter = OVERLAY_FRAMES

    feature_loss_counter = 0
    low_entropy_counter = 0
    last_scale_est = 1.0
    was_feature_rich = False
    prev_gray = None
    lost_counter = 0

    print("Auto-reset triggered.")


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

    fill_bottom = y2
    fill_top = y2 - fill_h

    color = get_distance_color(distance)

    cv2.rectangle(display, (x1 + 2, fill_top), (x2 - 2, fill_bottom - 2), color, -1)
    
    cv2.putText(display, f"{distance}px", (x1 - 20, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    cv2.putText(display, f"Radial", (x1 - 10, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)


# AXIAL BAR 
def draw_axial_bar(display, scale_est):
    bar_height = 300
    bar_width = 30

    x1 = FRAME_WIDTH - 110
    y1 = 100
    x2 = x1 + bar_width
    y2 = y1 + bar_height

    cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 2)

    # Normalize scale deviation to bar (1.0 is center)
    max_dev = max(abs(SCALE_MAX - 1.0), abs(1.0 - SCALE_MIN))
    dev = abs(scale_est - 1.0)
    dev = min(dev, max_dev)

    fill_h = int((dev / max_dev) * bar_height)

    fill_bottom = y2
    fill_top = y2 - fill_h

    ratio = dev / max_dev
    if ratio < 0.5:
        color = (0, 255, 0)
    elif ratio < 0.8:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(display, (x1 + 2, fill_top), (x2 - 2, fill_bottom - 2), color, -1)

    cv2.putText(display, f"Z {scale_est:.2f}x", (x1 - 25, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)

    cv2.putText(display, f"Axial", (x1 - 5, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)


def extract_ref_patch(ref_gray):
    half = PATCH_SIZE // 2
    cx, cy = ref_center
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, ref_gray.shape[1] - 1)
    y1 = min(cy + half, ref_gray.shape[0] - 1)

    patch = ref_gray[y0:y1, x0:x1].copy()
    center_in_patch = (min(half, cx - x0), min(half, cy - y0))
    return patch, center_in_patch


def compute_entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[64],[0,256])
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


# ===== semantic test for "feature-rich" region =====
def is_feature_rich(gray, kp):
    """
    Return True when the given frame is clearly feature-rich:
      - enough SIFT keypoints (spatial abundance)
      - minimum texture entropy
    These thresholds are chosen to separate the printed circle area from white fabric.
    """
    if kp is None:
        return False
    if len(kp) < FEATURE_RICH_KP_MIN:
        return False
    ent = compute_entropy(gray)
    if ent < FEATURE_RICH_ENTROPY_MIN:
        return False
    return True
# ==================================================


def template_match_fallback(curr_gray, search_center=None):
    global ref_patch

    if ref_patch is None:
        return None, None

    best_val = -1.0
    best_pt = None
    h_ref0, w_ref0 = ref_patch.shape[:2]

    for scale in TEMPLATE_SCALES:
        w_ref = max(3, int(w_ref0 * scale))
        h_ref = max(3, int(h_ref0 * scale))
        scaled_patch = cv2.resize(ref_patch, (w_ref, h_ref))

        if search_center is not None:
            sx = int(max(0, search_center[0] - SEARCH_RADIUS))
            sy = int(max(0, search_center[1] - SEARCH_RADIUS))
            ex = int(min(curr_gray.shape[1], search_center[0] + SEARCH_RADIUS))
            ey = int(min(curr_gray.shape[0], search_center[1] + SEARCH_RADIUS))
            if ex - sx < w_ref or ey - sy < h_ref:
                continue
            search_img = curr_gray[sy:ey, sx:ex]
            offset = (sx, sy)
        else:
            search_img = curr_gray
            offset = (0, 0)

        res = cv2.matchTemplate(search_img, scaled_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            cx = int(max_loc[0] + w_ref / 2 + offset[0])
            cy = int(max_loc[1] + h_ref / 2 + offset[1])
            best_pt = (cx, cy)

    return best_pt, best_val


def phase_correlation_fallback(curr_gray):
    global reference_frame
    if reference_frame is None:
        return None, None, 0.0

    ref_ds = cv2.resize(reference_frame, (PHASE_DS_WIDTH, PHASE_DS_HEIGHT))
    cur_ds = cv2.resize(curr_gray, (PHASE_DS_WIDTH, PHASE_DS_HEIGHT))

    ref_f = np.float32(ref_ds - np.mean(ref_ds))
    cur_f = np.float32(cur_ds - np.mean(cur_ds))

    shift, response = cv2.phaseCorrelate(ref_f, cur_f)

    scale_x = FRAME_WIDTH / PHASE_DS_WIDTH
    scale_y = FRAME_HEIGHT / PHASE_DS_HEIGHT

    return shift[0]*scale_x, shift[1]*scale_y, response


def optical_flow_fallback(prev_gray_frame, curr_gray_frame, prev_point):
    if prev_gray_frame is None or prev_point is None:
        return None, 0.0

    p0 = np.array([[prev_point]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray_frame,
        curr_gray_frame,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if st is None or st[0][0] != 1:
        return None, 0.0

    e = float(err[0][0]) if err is not None else 0.0
    conf = clamp01(1.0 - (e / FLOW_ERR_MAX))

    pt = (int(p1[0][0][0]), int(p1[0][0][1]))
    return pt, conf


def select_consensus(candidates):
    if not candidates:
        return None, 0.0, []

    viable = [c for c in candidates if c["conf"] >= CONFIDENCE_MIN]

    if not viable:
        strongest = max(candidates, key=lambda c: c["conf"])
        if strongest["conf"] >= CONFIDENCE_STRONG:
            return strongest["pt"], strongest["conf"], [strongest["method"]]
        return None, 0.0, []

    if len(viable) == 1:
        if viable[0]["conf"] >= CONFIDENCE_STRONG:
            return viable[0]["pt"], viable[0]["conf"], [viable[0]["method"]]
        return None, 0.0, []

    best_group = []
    best_score = -1.0

    for i, ci in enumerate(viable):
        group = [ci]
        for j, cj in enumerate(viable):
            if i == j:
                continue
            if pt_distance(ci["pt"], cj["pt"]) <= CONSENSUS_DIST:
                group.append(cj)
        score = sum(g["conf"] for g in group)
        if len(group) > len(best_group) or (len(group) == len(best_group) and score > best_score):
            best_group = group
            best_score = score

    if len(best_group) < CONSENSUS_MIN_METHODS:
        strongest = max(viable, key=lambda c: c["conf"])
        if strongest["conf"] >= CONFIDENCE_STRONG:
            return strongest["pt"], strongest["conf"], [strongest["method"]]
        return None, 0.0, []

    sum_conf = sum(g["conf"] for g in best_group)
    if sum_conf <= 0:
        return None, 0.0, []

    x = sum(g["pt"][0] * g["conf"] for g in best_group) / sum_conf
    y = sum(g["pt"][1] * g["conf"] for g in best_group) / sum_conf
    avg_conf = sum_conf / len(best_group)
    methods = [g["method"] for g in best_group]

    return (int(x), int(y)), avg_conf, methods


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

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

    # ----- TRACKING PATH (core idea preserved) -----
    if reference_frame is not None:
        kp, desc = compute_keypoints_descriptors(gray)

        # If reference was explicitly set from a feature-rich frame, enforce instant reset on leaving that region.
        try:
            curr_feature_rich = is_feature_rich(gray, kp)
            if was_feature_rich and not curr_feature_rich:
                print("Auto-reset (left feature-rich region - abrupt move).")
                reset_tracking()
                continue
        except Exception:
            pass

        if kp is not None and len(kp) < MIN_KP_CURRENT:
            feature_loss_counter += 1
        else:
            feature_loss_counter = 0

        if feature_loss_counter >= FEATURE_LOSS_FRAMES:
            print("Auto-reset (low feature density / axial motion)")
            reset_tracking()
            continue

        entropy_val = compute_entropy(gray)
        if entropy_val < ENTROPY_MIN:
            low_entropy_counter += 1
        else:
            low_entropy_counter = 0

        if low_entropy_counter >= LOW_ENTROPY_FRAMES:
            print("Auto-reset (low texture entropy / axial motion)")
            reset_tracking()
            continue

        candidates = []

        # --- Method 1: SIFT + homography ---
        try:
            if desc is not None and ref_desc is not None:
                matches = flann.knnMatch(ref_desc, desc, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]

                if len(good) >= MIN_GOOD_MATCHES:
                    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)

                    if H is not None:
                        sx = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
                        sy = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
                        scale_est = (sx + sy) / 2.0
                        last_scale_est = scale_est

                        if scale_est < SCALE_MIN or scale_est > SCALE_MAX:
                            print(f"Auto-reset (axial Z scale): {scale_est:.2f}")
                            reset_tracking()
                            continue

                        inliers = int(mask.sum()) if mask is not None else 0
                        inlier_ratio = inliers / max(len(good), 1)

                        if inliers >= MIN_INLIERS and inlier_ratio >= INLIER_RATIO_MIN:
                            ref_center_h = np.array([[ref_center[0]], [ref_center[1]], [1.0]])
                            live_h = H @ ref_center_h
                            pt = (int(live_h[0] / live_h[2]), int(live_h[1] / live_h[2]))

                            if in_frame(pt):
                                conf_inliers = min(1.0, inliers / max(MIN_INLIERS, 1))
                                conf_ratio = min(1.0, inlier_ratio / max(INLIER_RATIO_MIN, 1e-6))
                                conf_matches = min(1.0, len(good) / (MIN_GOOD_MATCHES * 1.5))
                                conf = clamp01((0.6 * conf_inliers + 0.4 * conf_ratio) * conf_matches)
                                candidates.append({"pt": pt, "conf": conf, "method": "homography"})
        except Exception:
            pass

        # --- Method 2: Template matching ---
        tmpl_pt, tmpl_score = template_match_fallback(gray, prev_live_pt)
        strong_thresh = TEMPLATE_MATCH_THRESH + TEMPLATE_STRONG_BONUS
        if tmpl_pt is not None and tmpl_score is not None and tmpl_score >= strong_thresh:
            if in_frame(tmpl_pt):
                conf = clamp01((tmpl_score - strong_thresh) / max(1.0 - strong_thresh, 1e-6))
                candidates.append({"pt": tmpl_pt, "conf": conf, "method": "template"})

        # --- Method 3: Phase correlation ---
        sx, sy, resp = phase_correlation_fallback(gray)
        if resp >= PHASE_RESPONSE_MIN:
            pt = (int(ref_center[0] + sx), int(ref_center[1] + sy))
            if in_frame(pt):
                conf = clamp01((resp - PHASE_RESPONSE_MIN) / max(1.0 - PHASE_RESPONSE_MIN, 1e-6))
                candidates.append({"pt": pt, "conf": conf, "method": "phase"})

        # --- Method 4: Optical flow (prev -> current) ---
        flow_pt, flow_conf = optical_flow_fallback(prev_gray, gray, prev_live_pt)
        if flow_pt is not None and flow_conf > 0.0:
            if in_frame(flow_pt):
                candidates.append({"pt": flow_pt, "conf": flow_conf, "method": "flow"})

        # Consensus selection
        consensus_pt, consensus_conf, consensus_methods = select_consensus(candidates)

        if consensus_pt is None:
            lost_counter += 1
            if lost_counter >= LOST_FRAMES_MAX:
                print("Auto-reset (lost consensus)")
                reset_tracking()
                continue
        else:
            lost_counter = 0
            live_pt = consensus_pt
            prev_live_pt = live_pt
            prev_gray = gray.copy()

            dx = live_pt[0] - ref_center[0]
            dy = live_pt[1] - ref_center[1]
            distance = int(np.sqrt(dx * dx + dy * dy))

            if distance > DISTANCE_THRESHOLD:
                reset_tracking()
    # ----- end tracking path -----

    if live_pt is not None:
        cv2.circle(display, live_pt, MARKER_SIZE, (0, 255, 0), -1)
        cv2.arrowedLine(display, live_pt, ref_center,
                        get_distance_color(distance), 3, tipLength=0.3)

    draw_scale_bar(display, distance)
    draw_axial_bar(display, last_scale_est)

    if overlay_counter > 0:
        overlay = display.copy()
        red = np.zeros_like(display)
        red[:] = (0, 0, 255)
        cv2.addWeighted(red, 0.4, overlay, 0.6, 0, overlay)
        overlay_counter -= 1
        display = overlay

    cv2.putText(display, 'S: Set reference | R: Reset | Q: Quit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    cv2.imshow("Endoscopy SIFT Tracking", display)

    key = cv2.waitKey(1) & 0xFF

    # Block setting reference when hovering over feature-poor region
    if key == ord('s'):
        kp_ref, desc_ref = compute_keypoints_descriptors(gray)
        try:
            if not is_feature_rich(gray, kp_ref):
                print("Cannot set reference: insufficient features / low texture in current view.")
                overlay_counter = OVERLAY_FRAMES
                continue
        except Exception:
            print("Cannot set reference: semantic validity check failed.")
            overlay_counter = OVERLAY_FRAMES
            continue

        reference_frame = gray.copy()
        ref_kp, ref_desc = kp_ref, desc_ref
        ref_patch, ref_patch_center = extract_ref_patch(reference_frame)
        live_pt = ref_center
        prev_live_pt = ref_center
        prev_gray = gray.copy()
        last_scale_est = 1.0
        was_feature_rich = True
        lost_counter = 0
        print("Reference set. Tracking enabled.")

    elif key == ord('r'):
        reset_tracking()

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
