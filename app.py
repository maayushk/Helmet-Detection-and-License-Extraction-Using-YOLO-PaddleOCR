import os
import re
import tempfile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- Page Setup ---
st.set_page_config(page_title="Smart Helmet Detection", layout="wide")


# ── Load external CSS ────────────────────────────────────────────────────────
def load_css(file_path):
    """Read a CSS file from disk and inject it into the Streamlit app."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Input Source")
input_type = st.sidebar.radio("", ["Image", "Video"])

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⬡ HELMET VIOLATION SYSTEM")
st.markdown(
    "<p style='font-family:Arial, sans-serif; font-size:0.78rem; color:#a0b0c8; letter-spacing:1px;'>"
    "PIPELINE → DETECT RIDER → CHECK HELMET → IF MISSING → DETECT PLATE → LOG VIOLATION"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — tweak these values to adjust detection sensitivity
# ─────────────────────────────────────────────────────────────────────────────
# Model class IDs (from yolo_helmet_model.pt)
CLASS_RIDER     = 1
CLASS_HELMET    = 2
CLASS_NO_HELMET = 3

# Detection confidence thresholds (range: 0.0 to 1.0)
#   ↓ lower  = more detections (catches weak ones, but more false positives)
#   ↑ higher = fewer detections (only strong ones, but may miss some)
RIDER_CONF_THRESHOLD     = 0.40   # confidence to call something a rider
NO_HELMET_CONF_THRESHOLD = 0.60   # confidence to call a head "no helmet"
PLATE_CONF_THRESHOLD     = 0.25   # confidence for license plate detection
# ─────────────────────────────────────────────────────────────────────────────

# --- Initialize Folders ---
# Parent folder that holds one sub-folder per app session.
VIOLATIONS_PARENT = "Violations"
os.makedirs(VIOLATIONS_PARENT, exist_ok=True)

# Create a new session folder ONCE per app startup.
# @st.cache_resource caches the result for the lifetime of the Streamlit
# process, so Streamlit's normal reruns (interactions, uploads) all share
# the SAME session folder. Killing the app and running it again invalidates
# the cache, which creates a fresh folder for the next session.
@st.cache_resource
def create_session_folder():
    """Make a new timestamped sub-folder inside 'Violations/' for this session."""
    session_name = "Violation_" + datetime.now().strftime("%d-%m-%Y_%I-%M-%p")
    session_path = os.path.join(VIOLATIONS_PARENT, session_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path

# All rider crops, plate crops, and the CSV for THIS run go inside this folder.
VIOLATIONS_DIR = create_session_folder()
EXCEL_PATH = os.path.join(VIOLATIONS_DIR, "violations_log.csv")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    helmet_model = YOLO("yolo_helmet_model.pt")
    plate_model  = YOLO("plate_model.pt")
    ocr_model    = PaddleOCR(use_angle_cls=True, lang='en')
    return helmet_model, plate_model, ocr_model

loading_placeholder = st.empty()
loading_placeholder.markdown(
    "<div class='loading-box'>⟳ &nbsp; LOADING AI MODELS — PLEASE WAIT...</div>",
    unsafe_allow_html=True
)
helmet_model, plate_model, ocr_model = load_models()
loading_placeholder.empty()


# ── Shared helper: run OCR on plate crop ──────────────────────────────────────
def run_ocr(plate_crop_bgr, ocr_model):
    img = cv2.resize(plate_crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ocr_result = ocr_model.ocr(img, cls=True)
    if ocr_result and ocr_result[0]:
        # Sort lines top-to-bottom by Y position, join all
        lines = sorted(ocr_result[0], key=lambda l: l[0][0][1])
        texts = [line[1][0] for line in lines]
        raw   = " ".join(texts)
        cleaned = raw.replace("IND", "").replace("ind", "").replace(".", "").replace(" ", "").strip().upper()

        # Valid Indian state/UT codes
        VALID_STATES = {
            "AN","AP","AR","AS","BR","CH","CG","DN","DD","DL",
            "GA","GJ","HR","HP","JK","JH","KA","KL","LD","MP",
            "MH","MN","ML","MZ","NL","OD","PY","PB","RJ","SK",
            "TN","TS","TR","UP","UK","WB"
        }

        # Common OCR misread corrections for state codes
        STATE_FIXES = {
            "1N":"TN","IN":"TN","T0":"TN","HN":"TN",
            "KN":"KA","MA":"MH","0D":"OD","U0":"UP",
            "R1":"RJ","P8":"PB","G1":"GJ","H8":"HR",
        }

        # Try to fix misread state code
        if len(cleaned) >= 2:
            prefix = cleaned[:2]
            if prefix not in VALID_STATES and prefix in STATE_FIXES:
                cleaned = STATE_FIXES[prefix] + cleaned[2:]

        # Validate and format: 2 letters + 2 digits + 2 letters + 4 digits
        # e.g. TN09BJ4054
        pattern = re.compile(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$')
        match = pattern.match(cleaned)
        if match:
            state, dist, series, num = match.groups()
            if state in VALID_STATES:
                return f"{state}{dist}{series}{num}"

        # If no full match, still return cleaned if state code is valid
        if len(cleaned) >= 2 and cleaned[:2] in VALID_STATES:
            return cleaned

        return cleaned
    return "No Plate Detected"


# ── Helper: check if a no-helmet box belongs to a rider ──────────────────────
def no_helmet_on_rider(rider_box, no_helmet_boxes):
    """
    Returns True if any 'no-helmet' detection sits inside this rider's bounding box.
    Uses the center-point of the no-helmet box to test containment — this is how
    we filter out pedestrians (who have heads but no surrounding rider box).
    """
    rx1, ry1, rx2, ry2 = map(int, rider_box.xyxy[0])
    for nh in no_helmet_boxes:
        nx1, ny1, nx2, ny2 = map(int, nh.xyxy[0])
        n_cx = (nx1 + nx2) // 2
        n_cy = (ny1 + ny2) // 2
        if rx1 <= n_cx <= rx2 and ry1 <= n_cy <= ry2:
            return True
    return False


# ── Shared helper: render violation log table from CSV ────────────────────────
def render_violation_log():
    st.markdown(
        "<h3 style='font-family:Arial, sans-serif; color:#e0e0e0; font-size:0.85rem;"
        "letter-spacing:2px; text-transform:uppercase;'>VIOLATION LOG — ALL RECORDS</h3>",
        unsafe_allow_html=True
    )
    if not os.path.exists(EXCEL_PATH):
        st.markdown(
            "<div class='no-violation'>NO RECORDS YET — VIOLATIONS WILL APPEAR HERE</div>",
            unsafe_allow_html=True
        )
        return
    df_all = pd.read_csv(EXCEL_PATH)
    if df_all.empty:
        st.markdown("<div class='no-violation'>NO RECORDS YET</div>", unsafe_allow_html=True)
        return

    hdr1, hdr2, hdr3, hdr4, hdr5 = st.columns([1.2, 1.2, 0.9, 0.9, 1.5])
    for hdr, label in zip(
        [hdr1, hdr2, hdr3, hdr4, hdr5],
        ["STATUS", "TIMESTAMP", "RIDER", "PLATE IMAGE", "PLATE NUMBER"]
    ):
        hdr.markdown(
            f"<p style='font-family:Arial, sans-serif; font-size:0.7rem; letter-spacing:1.5px;"
            f"color:#c0c8d8; margin-bottom:4px; font-weight:bold;'>{label}</p>",
            unsafe_allow_html=True
        )
    st.markdown("<hr style='border-color:#2e3145; margin:4px 0 10px 0;'>", unsafe_allow_html=True)

    for i, row in df_all.iloc[::-1].reset_index(drop=True).iterrows():
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 0.9, 0.9, 1.5])
        with c1:
            st.markdown("<span class='status-badge'>⚠ NO HELMET</span>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<p class='timestamp'>{row['Date and Time']}</p>", unsafe_allow_html=True)
        with c3:
            rp = str(row.get("Rider Image Path", ""))
            if rp and os.path.exists(rp):
                st.image(rp, channels="BGR", width=80)
            else:
                st.markdown("<p style='color:#606880; font-size:0.7rem;'>—</p>", unsafe_allow_html=True)
        with c4:
            pp = str(row.get("Plate Image Path", "None"))
            if pp != "None" and os.path.exists(pp):
                st.image(pp, channels="BGR", width=80)
            else:
                st.markdown("<p style='color:#606880; font-size:0.7rem;'>—</p>", unsafe_allow_html=True)
        with c5:
            st.markdown(
                f"<p class='plate-number'>{row.get('Extracted Text', '—')}</p>",
                unsafe_allow_html=True
            )
        if i < len(df_all) - 1:
            st.markdown("<hr style='border-color:#2e3145; margin:6px 0;'>", unsafe_allow_html=True)


# ── Shared helper: save violations to CSV ────────────────────────────────────
def save_to_csv(violations_found):
    if not violations_found:
        return
    df_new = pd.DataFrame(violations_found)
    if os.path.exists(EXCEL_PATH):
        pd.concat(
            [pd.read_csv(EXCEL_PATH), df_new], ignore_index=True
        ).to_csv(EXCEL_PATH, index=False)
    else:
        df_new.to_csv(EXCEL_PATH, index=False)


# Lowest threshold to feed YOLO — we predict at this and filter per-class after.
HELMET_PREDICT_CONF = min(RIDER_CONF_THRESHOLD, NO_HELMET_CONF_THRESHOLD)


# ════════════════════════════════════════════
# IMAGE MODE
# ════════════════════════════════════════════
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image     = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        with st.spinner("Running detection..."):
            # Predict at the lowest needed threshold; per-class filtering happens below
            helmet_results = helmet_model.predict(img_bgr, conf=HELMET_PREDICT_CONF)
            plate_results  = plate_model.predict(img_bgr, conf=PLATE_CONF_THRESHOLD)

        # Use YOLO default plot — no custom colors
        helmet_annotated = helmet_results[0].plot()
        plate_annotated  = plate_results[0].plot()

        def resize_to_height(img_rgb, height=420):
            h, w = img_rgb.shape[:2]
            scale = height / h
            return cv2.resize(img_rgb, (int(w * scale), height))

        img1 = resize_to_height(img_array)
        img2 = resize_to_height(cv2.cvtColor(helmet_annotated, cv2.COLOR_BGR2RGB))
        img3 = resize_to_height(cv2.cvtColor(plate_annotated,  cv2.COLOR_BGR2RGB))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='img-label'>① INPUT IMAGE</div>", unsafe_allow_html=True)
            st.image(img1, use_container_width=True)
        with col2:
            st.markdown("<div class='img-label'>② HELMET DETECTION</div>", unsafe_allow_html=True)
            st.image(img2, use_container_width=True)
        with col3:
            st.markdown("<div class='img-label'>③ LICENSE PLATE DETECTION</div>", unsafe_allow_html=True)
            st.image(img3, use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        violations_found = []

        def find_nearest_plate(rider_box, plate_boxes, img_bgr):
            """Find the plate closest to this rider horizontally."""
            if len(plate_boxes) == 0:
                return None, "No Plate Detected"
            rx1, ry1, rx2, ry2 = map(int, rider_box.xyxy[0])
            r_cx = (rx1 + rx2) // 2
            best_plate = min(plate_boxes, key=lambda b: abs(((int(b.xyxy[0][0]) + int(b.xyxy[0][2])) // 2) - r_cx))
            px1, py1, px2, py2 = map(int, best_plate.xyxy[0])
            crop = img_bgr[py1:py2, px1:px2]
            if crop.size == 0:
                return None, "No Plate Detected"
            return crop, run_ocr(crop, ocr_model)

        # ── Rider-first filtering with per-class thresholds ──────────────
        all_boxes = helmet_results[0].boxes
        rider_boxes = [
            b for b in all_boxes
            if int(b.cls[0]) == CLASS_RIDER and float(b.conf[0]) >= RIDER_CONF_THRESHOLD
        ]
        no_helmet_boxes = [
            b for b in all_boxes
            if int(b.cls[0]) == CLASS_NO_HELMET and float(b.conf[0]) >= NO_HELMET_CONF_THRESHOLD
        ]

        # Show a quick debug count
        st.markdown(
            f"<p style='font-family:Arial, sans-serif; font-size:0.72rem; color:#a0b0c8;'>"
            f"DETECTED → {len(rider_boxes)} rider(s) · {len(no_helmet_boxes)} no-helmet · "
            f"{len(plate_results[0].boxes)} plate(s)</p>",
            unsafe_allow_html=True
        )

        for rider_box in rider_boxes:
            # Skip riders who are wearing a helmet (or where no no-helmet
            # detection falls inside their box)
            if not no_helmet_on_rider(rider_box, no_helmet_boxes):
                continue

            x1, y1, x2, y2 = map(int, rider_box.xyxy[0])
            rider_crop = img_bgr[y1:y2, x1:x2]
            if rider_crop.size == 0:
                continue

            # Find nearest plate to THIS rider
            plate_crop, extracted_text = find_nearest_plate(
                rider_box, plate_results[0].boxes, img_bgr
            )

            timestamp_clean = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            rider_img_path = os.path.join(VIOLATIONS_DIR, f"rider_{timestamp_clean}.jpg")
            cv2.imwrite(rider_img_path, rider_crop)
            plate_img_path = "None"
            if plate_crop is not None and plate_crop.size > 0:
                plate_img_path = os.path.join(VIOLATIONS_DIR, f"plate_{timestamp_clean}.jpg")
                cv2.imwrite(plate_img_path, plate_crop)
            violations_found.append({
                "Date and Time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Rider Image Path": rider_img_path,
                "Plate Image Path": plate_img_path,
                "Extracted Text":   extracted_text,
            })

        save_to_csv(violations_found)

        if not violations_found:
            st.markdown(
                "<div class='no-violation'>✓ NO VIOLATIONS DETECTED IN THIS IMAGE</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='status-badge' style='display:inline-block; margin-bottom:8px;'>"
                f"⚠ {len(violations_found)} VIOLATION(S) DETECTED — LOGGED</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_violation_log()


# ════════════════════════════════════════════
# VIDEO MODE
# ════════════════════════════════════════════
elif input_type == "Video/Real-time":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:

        # Save uploaded video to a temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        video_path = tfile.name

        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        PROCESS_EVERY = 3

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-family:Arial, sans-serif; font-size:0.75rem; color:#a0b0c8;'>"
            f"VIDEO LOADED — {total_frames} FRAMES @ {fps:.1f} FPS — "
            f"PROCESSING EVERY {PROCESS_EVERY}RD FRAME</p>",
            unsafe_allow_html=True
        )

        # UI placeholders
        progress_bar    = st.progress(0)
        status_text     = st.empty()
        frame_display   = st.empty()
        violation_count = st.empty()

        violations_found  = []
        seen_plates       = set()   # unique plate texts already logged
        logged_positions  = set()   # center-x zones already fully logged
        pending           = {}
        frame_idx         = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress = int((frame_idx / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            status_text.markdown(
                f"<p style='font-family:Arial, sans-serif; font-size:0.72rem; color:#606880;'>"
                f"PROCESSING FRAME {frame_idx}/{total_frames}</p>",
                unsafe_allow_html=True
            )

            if frame_idx % PROCESS_EVERY != 0:
                continue

            # ── Run detection ─────────────────────────────────────────────
            helmet_results = helmet_model.predict(frame, conf=HELMET_PREDICT_CONF, verbose=False)
            plate_results  = plate_model.predict(frame, conf=PLATE_CONF_THRESHOLD, verbose=False)

            # ── Annotated preview — YOLO default colors ───────────────────
            helmet_ann = helmet_results[0].plot()
            plate_ann  = plate_results[0].plot()
            annotated  = cv2.addWeighted(helmet_ann, 0.5, plate_ann, 0.5, 0)
            preview_w  = 350
            preview_h  = int(annotated.shape[0] * preview_w / annotated.shape[1])
            preview    = cv2.resize(annotated, (preview_w, preview_h))
            frame_display.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width=preview_w)

            # ── Rider-first filtering with per-class thresholds ───────────
            all_boxes = helmet_results[0].boxes
            rider_boxes = [
                b for b in all_boxes
                if int(b.cls[0]) == CLASS_RIDER and float(b.conf[0]) >= RIDER_CONF_THRESHOLD
            ]
            no_helmet_boxes = [
                b for b in all_boxes
                if int(b.cls[0]) == CLASS_NO_HELMET and float(b.conf[0]) >= NO_HELMET_CONF_THRESHOLD
            ]

            violating_riders = [
                b for b in rider_boxes
                if no_helmet_on_rider(b, no_helmet_boxes)
            ]

            # ── Update pending buffer ─────────────────────────────────────
            for box in violating_riders:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx   = (x1 + x2) // 2
                matched_key = None
                for key in pending:
                    if abs(key - cx) < 80:
                        matched_key = key
                        break
                rider_crop = frame[y1:y2, x1:x2]
                if rider_crop.size == 0:
                    continue

                # Find nearest plate to THIS rider
                frame_plate_crop = None
                frame_plate_text = None
                frame_plate_conf = 0.0
                if len(plate_results[0].boxes) > 0:
                    nearest = min(plate_results[0].boxes,
                                  key=lambda b: abs(((int(b.xyxy[0][0]) + int(b.xyxy[0][2])) // 2) - cx))
                    npx1, npy1, npx2, npy2 = map(int, nearest.xyxy[0])
                    ncrop = frame[npy1:npy2, npx1:npx2]
                    if ncrop.size > 0:
                        ntext = run_ocr(ncrop, ocr_model)
                        if ntext != "No Plate Detected":
                            frame_plate_crop = ncrop
                            frame_plate_text = ntext
                            frame_plate_conf = float(nearest.conf[0])

                if matched_key is None:
                    pending[cx] = {
                        "best_rider_crop":  rider_crop,
                        "best_rider_conf":  conf,
                        "plate_crop":       frame_plate_crop,
                        "plate_text":       frame_plate_text,
                        "best_plate_conf":  frame_plate_conf,
                        "frames_seen":      1,
                    }
                else:
                    entry = pending[matched_key]
                    entry["frames_seen"] += 1
                    if conf > entry["best_rider_conf"]:
                        entry["best_rider_crop"] = rider_crop
                        entry["best_rider_conf"] = conf
                    if frame_plate_crop is not None and frame_plate_conf > entry["best_plate_conf"]:
                        entry["plate_crop"]      = frame_plate_crop
                        entry["plate_text"]      = frame_plate_text
                        entry["best_plate_conf"] = frame_plate_conf
                    pending[cx] = entry
                    if matched_key != cx:
                        del pending[matched_key]

            # ── Log fully captured entries ────────────────────────────────
            # Wait at least 5 processed frames before logging so we collect
            # the highest confidence plate across multiple frames
            fully_captured = []
            for key, entry in pending.items():
                plate_text = entry["plate_text"] or "No Plate Detected"
                already_logged = any(abs(key - lp) < 80 for lp in logged_positions)
                if (entry["plate_crop"] is not None
                        and entry["frames_seen"] >= 5
                        and plate_text not in seen_plates
                        and not already_logged):
                    seen_plates.add(plate_text)
                    logged_positions.add(key)
                    fully_captured.append(key)
                    timestamp_clean = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    rider_img_path  = os.path.join(VIOLATIONS_DIR, f"rider_{timestamp_clean}.jpg")
                    plate_img_path  = os.path.join(VIOLATIONS_DIR, f"plate_{timestamp_clean}.jpg")
                    cv2.imwrite(rider_img_path, entry["best_rider_crop"])
                    cv2.imwrite(plate_img_path, entry["plate_crop"])
                    violations_found.append({
                        "Date and Time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Rider Image Path": rider_img_path,
                        "Plate Image Path": plate_img_path,
                        "Extracted Text":   plate_text,
                    })
                    violation_count.markdown(
                        f"<div class='status-badge' style='display:inline-block; margin-bottom:4px;'>"
                        f"⚠ {len(violations_found)} VIOLATION(S) LOGGED SO FAR</div>",
                        unsafe_allow_html=True
                    )
            for key in fully_captured:
                pending.pop(key, None)

        cap.release()
        try:
            os.unlink(video_path)
        except PermissionError:
            pass
        progress_bar.progress(100)
        status_text.markdown(
            f"<p style='font-family:Arial, sans-serif; font-size:0.72rem; color:#52c0a0;'>"
            f"✓ VIDEO PROCESSING COMPLETE — {total_frames}/{total_frames} FRAMES DONE</p>",
            unsafe_allow_html=True
        )

        save_to_csv(violations_found)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        if not violations_found:
            st.markdown(
                "<div class='no-violation'>✓ NO VIOLATIONS DETECTED IN THIS VIDEO</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='status-badge' style='display:inline-block; margin-bottom:8px;'>"
                f"⚠ {len(violations_found)} TOTAL VIOLATION(S) DETECTED — LOGGED</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_violation_log()
