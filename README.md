
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00FFAA,100:5C3EE8&height=300&section=header&text=ANPR%20System&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=YOLOv8%20%2B%20EasyOCR%20%7C%20Real-Time%20Detection&descSize=20&descAlignY=60" width="100%"/>
</p>



![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=for-the-badge&logo=github&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-FF6B6B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> **Real-time license plate detection and text recognition using state-of-the-art deep learning.**

<p>

<a href="https://automatic-number-plate-recognition-yolov8-gkdvntr7uccgaqpuci5m.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀_Live_Demo-Click_Here-success?style=for-the-badge">
</a>
<a href="https://github.com/MhmdGamalAskar/automatic-number-plate-recognition-yolov8">
  <img src="https://img.shields.io/github/stars/MhmdGamalAskar/automatic-number-plate-recognition-yolov8?style=for-the-badge&logo=github">
</a>
<a href="https://automatic-number-plate-recognition-yolov8-gkdvntr7uccgaqpuci5m.streamlit.app/">
  <img src="https://img.shields.io/badge/☁️_Streamlit_Cloud-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</a>

</p>

---

## 📸 Demo

![Demo](demo.gif)

---

## 📌 Overview

**ANPR (Automatic Number Plate Recognition)** is an end-to-end computer vision pipeline that automatically detects and reads license plates from images.

Built with a fine-tuned **YOLOv8** model for detection and **EasyOCR** for character recognition, wrapped in a clean **Streamlit** web interface — no setup needed for end users.

1. **Detects** license plates in uploaded images using a fine-tuned **YOLOv8** model
2. **Crops** the plate region with high precision
3. **Reads** the plate number using **EasyOCR**
4. **Displays** the result through a clean **Streamlit** interface

No prior setup needed — just upload an image and get results in seconds.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Plate Detection** | Fine-tuned YOLOv8 for high-accuracy bounding box localization |
| 🔤 **Text Recognition** | EasyOCR for robust multi-character extraction |
| 🖼️ **Simple UI** | Drag-and-drop image upload via Streamlit |
| ⚡ **Fast Inference** | Full pipeline runs in under a second |
| 📊 **Visual Output** | Annotated image with box overlay + extracted plate number |

---

## 🧠 Tech Stack

```
Computer Vision  →  YOLOv8 (Ultralytics) + OpenCV
OCR Engine       →  EasyOCR
Web Interface    →  Streamlit
Data Handling    →  NumPy · Pandas
Language         →  Python 3.12
```

---

## 📂 Project Structure

```
automatic-number-plate-recognition-yolov8/
│
├── app.py                  # Main Streamlit application
├── util.py                 # Helper functions (crop, preprocess, draw)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (for deployment)
│
├── model/
│   └── best.pt             # Fine-tuned YOLOv8 weights
│
└── README.md
```

---

## 🚀 How It Works

```
📤 Upload Image
      │
      ▼
🔍 YOLOv8 Detection ──► Bounding box around license plate
      │
      ▼
✂️  Crop Plate Region
      │
      ▼
🔤 EasyOCR Recognition ──► Extract text characters
      │
      ▼
📊 Display Result ──► Annotated image + plate number
```

---

## ▶️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/MhmdGamalAskar/automatic-number-plate-recognition-yolov8.git
cd automatic-number-plate-recognition-yolov8
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## ☁️ Live Demo

👉 **[Try it on Streamlit Cloud](https://automatic-number-plate-recognition-yolov8-gkdvntr7uccgaqpuci5m.streamlit.app/)**

---


## 🙌 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR by JaidedAI](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

---


## 📬 Contact

<div align="center">

**Mohamed Gamal Askar**

[![GitHub](https://img.shields.io/badge/GitHub-MhmdGamalAskar-181717?style=for-the-badge&logo=github)](https://github.com/MhmdGamalAskar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mohamed_Gamal_Askar-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamedgamalaskar/)
[![Gmail](https://img.shields.io/badge/Gmail-mg.askar.ai@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mg.askar.ai@gmail.com)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-%2B201102644939-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://wa.me/201102644939)

</div>

---

<div align="center">

**⭐ If this project helped you  drop a star — it keeps the motivation going!**

![Wave](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)

</div>
