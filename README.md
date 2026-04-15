# ⬡ Helmet Violation Detection System

An AI-powered system that detects motorcycle riders without helmets, extracts their license plate numbers, and logs violations automatically — using YOLOv8 and PaddleOCR.

---

## 📸 What It Does

- Detects riders **with** and **without** helmets using a custom YOLOv8 model
- Detects and reads **license plates** using a plate detection model + PaddleOCR
- Logs all violations with **rider image**, **plate image**, and **plate number**
- Works on both **images** and **videos**
- Saves a full violation log to CSV

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

Both model files are included in this repository:

| File | Description |
|---|---|
| `yolo_helmet_model.pt` | Helmet detection model |
| `plate_model.pt` | License plate detection model |

They are already in the correct location — no extra setup needed.

---

### Step 4 — Run the App

Double-click **`run.bat`**

This will:
1. Automatically install all required Python packages
2. Launch the app in your browser at `http://localhost:8501`

> If your browser doesn't open automatically, go to: **http://localhost:8501**

---

## 🚀 How to Use

### Image Mode
1. Select **Image** from the sidebar
2. Upload a `.jpg`, `.jpeg`, or `.png` file
3. The system will show:
   - Input image
   - Helmet detection result
   - License plate detection result
4. Violations are logged in the table below

### Video Mode
1. Select **Video/Real-time** from the sidebar
2. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` file
3. Full violation log appears after processing

---

## 📁 Project Structure

```
HelmetDetection/
│
├── app.py                  # Main application
├── run.bat                 # Double-click to launch
├── requirements.txt        # Python dependencies
├── yolo_helmet_model.pt    # Helmet detection model (download separately)
├── plate_model.pt          # Plate detection model (download separately)
│
└── Violations/             # Auto-created when app runs
    ├── violations_log.csv  # All logged violations
    ├── rider_*.jpg         # Saved rider face crops
    └── plate_*.jpg         # Saved plate crops
```

---

## 🛠️ Manual Installation (if run.bat fails)

Open **Command Prompt** and run these one by one:

```bash
pip install streamlit
pip install opencv-python
pip install "numpy>=1.24,<2.0"
pip install pandas Pillow ultralytics
pip install paddlepaddle==2.6.2
pip install paddleocr==2.7.3
```

Then launch the app:
```bash
streamlit run app.py
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Helmet Detection | YOLOv8 (Ultralytics) |
| Plate Detection | YOLOv8 (custom trained) |
| OCR | PaddleOCR 2.7.3 |
| Image Processing | OpenCV |

---

## 🗒️ Notes

- First run will take longer as PaddleOCR downloads its internal models (~500MB)
- GPU (NVIDIA) is automatically used if available — no setup needed
- The `Violations/` folder is created automatically in the same directory as `app.py`
- Violation log (`violations_log.csv`) persists across sessions

---

## 👨‍💻 Made By

**Aayush** — AI/ML Project, 2026
