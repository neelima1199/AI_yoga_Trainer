AI Yoga Trainer 🧘‍♀️
Overview

AI Yoga Trainer is an intelligent yoga posture analysis system that uses computer vision and machine learning techniques to detect, classify, and evaluate yoga poses in real time. The application assists users in performing yoga exercises correctly by providing posture recognition and feedback.

The project leverages pose estimation techniques to identify body landmarks and analyze user posture, helping improve yoga practice accuracy and reducing the risk of injury.

Features
Real-time yoga pose detection
Human pose estimation using computer vision
Yoga pose classification and recognition
Posture accuracy evaluation
Machine learning model training and testing
Interactive user interface
Extensible architecture for adding new yoga poses

Project Structure
AI_Yoga_Trainer/
│
├── app.py                     # Main application
├── config.py                  # Configuration settings
├── data_loader.py             # Dataset loading utilities
├── model.py                   # Machine learning model
├── pose_estimator.py          # Pose estimation module
├── pose_rules.py              # Yoga pose evaluation rules
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation script
├── tune.py                    # Hyperparameter tuning
├── requirements.txt           # Dependencies
└── report_file/               # Project documentation

**Technologies Used**
Python
OpenCV
MediaPipe
NumPy
Scikit-Learn
Machine Learning
Computer Vision
**Installation**
Clone the Repository
git clone https://github.com/neelima1199/AI_Yoga_Trainer.git
cd AI_Yoga_Trainer
Create Virtual Environment
python -m venv venv

**Activate Environment**

Windows:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt

**Running the Application**
python app.py

**Training the Model**
python train.py

**Evaluating the Model**
python evaluate.py
Future Enhancements
Voice-guided yoga instructions
Personalized workout recommendations
Mobile application support
Advanced posture correction feedback
Progress tracking dashboard

**Applications**
Home yoga training
Fitness coaching
Health and wellness monitoring
Educational yoga platforms
