# Passive Fatigue Detection for Autonomous Vehicles 🚗👁️

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![University of Michigan](https://img.shields.io/badge/UMich-00274C?style=flat)](https://umich.edu)

> **Master's Capstone Project** — University of Michigan, Dearborn | Summer 2024  
> M.S. Artificial Intelligence — Knowledge Management and Reasoning

---

## Overview

Passive fatigue is one of the leading causes of accidents in semi-autonomous and Level 3 autonomous vehicles — unlike active fatigue, it occurs without the driver's awareness, making it significantly harder to detect and act on.

This project presents a **multimodal machine learning framework** that continuously monitors driver state using eye-tracking, facial cues, head pose estimation, and vehicle speed data to detect the onset of passive fatigue in real time — without requiring any active input from the driver.

The system achieved **92% fatigue detection accuracy** using a Random Forest classifier trained on multi-stream in-car video data, outperforming single-modality baselines across all tested conditions.

---

## Key Results

| Metric | Result |
|---|---|
| Fatigue detection accuracy | **92%** |
| Model | Random Forest (multi-modal features) |
| Feature streams | Eye tracking, facial landmarks, head pose, vehicle speed |
| Weather classification accuracy | 4-class environmental model (VGGNet) |
| Dataset | Dreyeve (environmental) + custom in-car video streams |

---

## System Architecture

The pipeline consists of three main components running in parallel:

```
In-car video stream
        │
        ├──► Gaze & Eye Tracker (MediaPipe)
        │         └── Blink rate, gaze deviation, pupil dilation
        │
        ├──► Facial Landmark Detector (MediaPipe)
        │         └── Eye aspect ratio, mouth openness, facial droop
        │
        ├──► Head Pose Estimator (OpenCV)
        │         └── Pitch, yaw, roll deviation from baseline
        │
        └──► Vehicle Speed Monitor
                  └── Speed anomalies, lane deviation signals
                           │
                           ▼
              Feature Fusion → Random Forest Classifier
                           │
                           ▼
              Fatigue Alert System
```

Additionally, a **VGGNet-based weather classifier** trained on the Dreyeve dataset runs in parallel to account for environmental factors (rain, fog, night driving) that influence driver alertness — ensuring the fatigue model adapts to external conditions.

---

## Features

- **Real-time gaze deviation detection** — tracks where the driver is looking and flags sustained off-road fixation
- **Blink rate monitoring** — identifies abnormal blink patterns associated with drowsiness
- **Head pose tracking** — detects micro-sleep indicators via sudden head drops or sustained tilt
- **Multimodal fusion** — combines all streams into a unified feature vector for classification
- **Environmental context** — weather classification adjusts alertness thresholds dynamically
- **Automated data pipelines** — processes raw in-car video streams end-to-end with no manual annotation step

---

## Tech Stack

| Category | Tools |
|---|---|
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | Scikit-learn (Random Forest), VGGNet |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset | Dreyeve (environmental), custom in-car video |

---

## Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe scikit-learn pandas numpy matplotlib seaborn
```

### Run the pipeline

```bash
# Clone the repository
git clone https://github.com/SaliElloh/Gaze-Detection-Final-
cd Gaze-Detection-Final-

# Run the main detection pipeline on a video file
python main.py --input path/to/video.mp4

# Run on live webcam feed
python main.py --input 0
```

---

## Project Structure

```
Gaze-Detection-Final-/
├── gaze_tracker/          # Eye tracking and gaze deviation modules
├── face_landmarks/        # Facial landmark extraction pipeline
├── head_pose/             # Head pose estimation
├── weather_classifier/    # VGGNet-based environmental model
├── data_pipeline/         # ETL pipeline for video stream processing
├── models/                # Trained model weights
├── notebooks/             # Exploratory analysis and results
└── main.py                # Main entry point
```

---

## Background & Motivation

Level 3 autonomous vehicles hand control back to the driver when needed — but passive fatigue means the driver may be physically present yet cognitively absent. Existing fatigue detection systems rely on intrusive sensors or require the driver to perform calibration tasks. This project explores a fully passive, vision-based approach that works with standard in-car cameras alone.

---

## Related Work

This project is part of a broader research direction on driver monitoring for autonomous systems. Related work includes:

- [Dreyeve Dataset](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8) — large-scale driver attention dataset
- VGGNet architecture for environmental classification
- MediaPipe Face Mesh for real-time landmark detection

---

## Author

**Sali El-loh**  
M.S. Artificial Intelligence | University of Michigan — Dearborn  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/salielloh12/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/SaliElloh)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:selloh@umich.edu)

---

## License

No license specified. Contact the author for usage permissions.
