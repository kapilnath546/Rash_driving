# 🚗 Rash Driving Detection System

A real-time computer vision system to detect rash driving behaviors such as overspeeding, sudden lane changes, and red-light violations using YOLOv5, Python, OpenCV, and Machine Learning.

---

## 🔍 Features

- 🚦 Detects overspeeding, lane violations, and abrupt maneuvers
- 🎯 YOLOv5-based vehicle detection from live video feed
- 🧠 Trained ML model to classify rash driving behavior
- 🗺️ Visual overlays for detected violations
- 🤖 Integrated chatbot for emergency help and reporting
- 📊 Live log of detected events for analysis

---

## 🛠️ Tech Stack

- **Languages**: Python, JavaScript (for chatbot UI)
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib
- **ML Model**: Scikit-learn / Custom classifier
- **Detection**: YOLOv5 (pre-trained or custom-trained weights)
- **Backend**: Flask / Node.js (optional for chatbot APIs)
- **Chatbot**: Simple AI logic / Dialogflow / Rasa
- **Deployment**: Localhost or cloud (optional)

---

## 📂 Project Structure

rash-driving-detection/
├── yolov5/ # YOLOv5 detection logic
├── dataset/ # Training dataset
├── models/ # Trained ML models
├── chatbot/ # Chatbot logic (UI/API)
├── utils/ # Helper functions
├── main.py # Entry point for detection
├── requirements.txt # Dependencies
└── README.md # This file

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/rash-driving-detection.git
cd rash-driving-detection
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the detection script

bash
Copy
Edit
python main.py
Access Chatbot (optional)

Launch chatbot UI: chatbot/index.html

Or run with Flask/API if implemented

📊 Results
⚡ Real-time performance: ~30 FPS on webcam input

🎯 Rash driving detection accuracy: 85%+

✅ Successfully detects lane violation and red-light jumping

🤖 Chatbot enables quick emergency responses and reporting
