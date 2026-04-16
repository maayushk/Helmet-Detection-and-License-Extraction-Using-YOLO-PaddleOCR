# ⬡ Helmet Violation Detection System

An AI-powered traffic violation detection system that identifies motorcycle riders without helmets, extracts their license plate numbers, and automatically logs violations — built using YOLOv11, YOLOv8, and PaddleOCR.

---

## 🎯 What It Does

- Detects riders **with** and **without** helmets using a custom YOLOv11 model
- Detects and reads **license plates** using a custom YOLOv8 model + PaddleOCR
- Logs all violations with **rider image**, **plate image**, **plate number**, and **timestamp**
- Works on both **images** and **videos**
- Saves a persistent violation log to CSV across sessions

---

## 📁 Project Structure

```
HelmetDetection/
│
├── app.py                  # Main Streamlit application
├── install.bat             # Run ONCE to install all packages
├── run.bat                 # Double-click to launch the app
├── requirements.txt        # Python package list
├── yolo_helmet_model.pt    # Helmet detection model (YOLOv11)
├── plate_model.pt          # License plate detection model (YOLOv8)
│
├── Screenshots/            # App screenshots
│
└── Violations/
    ├── Violation_16-04-2026_11-18-AM/
    │   ├── rider_20260416_111823_123.jpg
    │   ├── plate_20260416_111823_123.jpg
    │   └── violations_log.csv
    ├── Violation_16-04-2026_02-45-PM/
    │   ├── rider_20260416_144502_456.jpg
    │   ├── plate_20260416_144502_456.jpg
    │   └── violations_log.csv
    └── Violation_17-04-2026_09-30-AM/
        └── violations_log.csv
```

---

## 🔧 How It Works

### Detection Pipeline

```
Input            ──►  YOLOv11 Medium     ──►  YOLOv8 Nano       ──►  PaddleOCR        ──►  Violation Logger
(Image/Video)         Helmet Detection        Plate Detection         Text Extraction        Save & Log
                       • Rider                 • Finds nearest         • 3x upscale           • Rider crop
                       • Helmet                  plate to rider        • Read all lines        • Plate crop
                       • No_Helmet             • Per-rider             • Validate Indian       • Plate number
                      ─────────────            assignment              state code              • CSV log
                      No_Helmet + Rider?
```

### Smart Detection Features

- **Rider-Helmet Linking** — No_Helmet is only flagged if it spatially overlaps with a Rider box, eliminating false positives like pedestrians or traffic police
- **Nearest Plate Assignment** — Each rider is assigned the closest license plate to their bounding box, preventing wrong plate assignment when multiple bikes are in frame
- **Duplicate Prevention** — Same person is never logged twice using plate text tracking and position zone tracking across frames
- **Best Frame Selection (Video)** — Waits across multiple frames and picks the highest confidence plate reading before logging
- **Indian Plate Validation** — OCR output is corrected for common misreads and validated against all 36 official Indian state/UT codes and the standard format (`XX00XX0000`)

---

## 🤖 Model Details

| Model | Architecture | Classes | Epochs | Dataset |
|---|---|---|---|---|
| `yolo_helmet_model.pt` | YOLOv11 Medium | Motorcycle, Rider, Helmet, No_Helmet | 100 | Custom Indian traffic images |
| `plate_model.pt` | YOLOv8 Nano | License_Plate | 50 | Indian Car Bike Number Plate v2 + ANPR Dataset |

**OCR** — PaddleOCR 2.7.3 with DB detection + CRNN recognition, 3× upscale preprocessing, Indian state code validation.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Web interface for image/video upload and results display |
| Helmet Detection | YOLOv11 Medium | Detects riders, helmets, and no-helmet violations in real time |
| Plate Detection | YOLOv8 Nano | Locates license plates in the frame |
| OCR | PaddleOCR 2.7.3 | Extracts alphanumeric text from detected plate crops |
| Deep Learning | PyTorch via Ultralytics | Runs YOLO model inference on CPU or GPU |
| Image Processing | OpenCV | Frame reading, cropping, resizing, and color conversion |
| Data Handling | Pandas, NumPy | Violation log management and array operations |
| Training Platform | Google Colab + NVIDIA T4 | GPU-accelerated model training environment |

---

## 🖥️ System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10 / 11 |
| Python | 3.9, 3.10, or 3.11 |
| RAM | 8 GB |
| GPU | Optional (NVIDIA recommended for faster video processing) |
| Storage | ~3 GB (for models + packages) |

---

## 📦 Installation & Setup

### Step 1 — Install Python

Download and install Python 3.11 from:
👉 https://www.python.org/downloads/

> ⚠️ During installation, **check the box that says "Add Python to PATH"**

---

### Step 2 — Download This Project

Click the green **Code** button on this GitHub page → **Download ZIP**

Extract the ZIP to a folder on your computer (e.g. `C:\HelmetDetection`)

---

### Step 3 — Model Files

Both model files are included in this repository — no extra setup needed.

---

### Step 4 — Install Required Packages (First Time Only)

Double-click **`install.bat`**

- Detects your GPU (NVIDIA / AMD / Intel) automatically
- Installs all required packages
- Takes about 5–10 minutes

> ✅ Only do this once. Skip if you already have all packages installed.

---

### Step 5 — Run the App

Double-click **`run.bat`**

- Checks everything is installed and model files are present
- Opens the app at `http://localhost:8501`

---

## 🚀 How to Use

### Image Mode
1. Select **Image** from the sidebar
2. Upload a `.jpg`, `.jpeg`, or `.png` file
3. The system shows three panels — input, helmet detection, plate detection
4. Violations are automatically logged in the table below

### Video Mode
1. Select **Video/Real-time** from the sidebar
2. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` file
3. Violation log appears after processing completes

---

## 🗒️ Notes

- First run takes longer as PaddleOCR downloads its internal models (~500MB) — one-time only
- NVIDIA GPU is automatically used if available — no manual setup needed
- The `Violations/` folder and CSV log are created automatically and persist across sessions

---

## 🛠️ Manual Installation (if install.bat fails)

Open **Command Prompt** and run:

```bash
pip install streamlit
pip install opencv-python
pip install "numpy>=1.24,<2.0"
pip install pandas Pillow
pip install ultralytics --upgrade
pip install paddlepaddle==2.6.2
pip install paddleocr==2.7.3
```

Then launch:
```bash
streamlit run app.py
```

---

## 📸 Screenshots

<p align="center">
  <img src="Screenshots/Home.png" width="700"/><br><em>Home Screen</em>
</p>

<p align="center">
  <img src="Screenshots/Image%20Input.png" width="700"/><br><em>Image Mode — Detection Results</em>
</p>

<p align="center">
  <img src="Screenshots/Image%20Output.png" width="700"/><br><em>Image Mode — Violation Logged</em>
</p>

<p align="center">
  <img src="Screenshots/Video%20Input.png" width="700"/><br><em>Video Mode — Live Processing</em>
</p>

<p align="center">
  <img src="Screenshots/Video%20Output.png" width="700"/><br><em>Video Mode — Violation Log</em>
</p>

---

## ⚠️ Known Limitations

- **Blurry or distant plates** — OCR accuracy drops on small or blurry plates; LPR cameras would solve this
- **Angled plates** — Plates tilted more than ~30° may be misread or partially extracted
- **Overlapping riders** — When multiple riders are very close, plate assignment may occasionally be incorrect
- **Low light** — Detection confidence drops in poor lighting or night-time footage without infrared cameras
- **Non-standard plates** — Fancy fonts, stickers, or damaged plates may not be read correctly

---

## 🔮 Future Improvements

### Accuracy Improvements
- Train a dedicated Indian license plate OCR model instead of general-purpose PaddleOCR
- Train the plate model specifically on bike plates (currently trained on mixed car/bike data)
- Integrate LPR cameras with infrared sensors for accurate plate reading at high speeds, long distances, and in low light
- Add plate tilt correction (deskewing) before OCR for angled plates

### Feature Additions
- Real-time CCTV/webcam stream support via RTSP instead of uploaded files only
- RTO e-Challan integration — query VAHAN API for the owner's registered mobile number and auto-send a digital challan via SMS with a UPI payment link, completely eliminating paperwork
- Dashboard with violation statistics — graphs by time, location, and plate frequency
- Multiple camera support for full intersection monitoring
- Automatic PDF report generation with rider photo, plate image, timestamp, and location

### Technical Improvements
- TensorRT optimization for YOLO models — 2-3x faster inference on NVIDIA GPUs enabling true real-time processing
- ByteTrack/SORT integration for proper multi-object tracking instead of the current center-X approach
- SQLite database instead of CSV for better querying, filtering, and scalability

---
