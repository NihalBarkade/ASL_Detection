# 🧠 ASL Detection Using Deep Learning + Mediapipe

This project is a real-time American Sign Language (ASL) hand gesture recognition system using a trained deep learning model with Mediapipe for hand detection and OpenCV for webcam input.

---

## 📁 Project Structure

ASL_DETECTION_2/ │ ├── asl_model.h5 # Trained Keras model for ASL classification ├── data_loader.py # Loads and preprocesses the ASL dataset ├── data_split.py # Splits the dataset into training/validation/test ├── model_trainer.py # Model training script ├── evaluate_model.py # Model evaluation script ├── real_time.py # Real-time detection (base version) ├── real_time2.py # Real-time detection (with smoothing, latest version ✅) ├── test_images.py # Run detection on static images ├── trainer_visualizer.py # Plot training metrics ├── requirements.txt # Python dependencies ├── .gitignore # Git ignore file (excludes dataset and env) └── README.md # You're reading this!

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/NihalBarkade/ASL_Detection.git
cd ASL_Detection
```

### 2. Create and Activate Virtual Environment

python -m venv asl_env
source asl_env/bin/activate # Linux/Mac

# or

asl_env\Scripts\activate # Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run Real-Time ASL Detection

python real_time.py

# or

python real_time2.py is the improved version with smoothing and flicker reduction.
