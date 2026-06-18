# 🧘 Training on Google Colab — Step-by-Step Guide

## 1. Open Google Colab
Go to [Google Colab](https://colab.research.google.com/) and create a **new notebook**.

## 2. Enable GPU
Go to **Runtime → Change runtime type → GPU (T4)** and save.

Verify GPU is available:
```python
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

## 3. Upload Project Files
Upload the following files to Colab (use the file browser on the left sidebar, or the upload button):

### Option A: Upload via Google Drive (Recommended for large dataset)
```python
from google.colab import drive
drive.mount('/content/drive')

# If your project is on Drive:
# !cp -r "/content/drive/MyDrive/Project/" "/content/Project/"
```

### Option B: Upload directly
```python
from google.colab import files

# Upload these Python files:
# - config.py
# - data_loader.py
# - model.py
# - train.py
# - evaluate.py
```

Then upload your dataset. The easiest way is to zip the DATASET folder first:

**On your Windows machine:**
```
Right-click DATASET folder → Send to → Compressed (zipped) folder
```

Then in Colab:
```python
from google.colab import files
uploaded = files.upload()  # Upload DATASET.zip

# Unzip it
!unzip -q DATASET.zip -d .
```

After upload, your Colab file structure should look like:
```
/content/
├── config.py
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
└── DATASET/
    ├── TRAIN/
    │   ├── downdog/
    │   ├── goddess/
    │   ├── plank/
    │   ├── tree/
    │   └── warrior2/
    └── TEST/
        ├── downdog/
        ├── goddess/
        ├── plank/
        ├── tree/
        └── warrior2/
```

## 4. Install Dependencies
```python
!pip install -q tensorflow mediapipe scikit-learn seaborn tqdm
```

## 5. Fix Paths for Colab
In Colab, the working directory is `/content/`. Run this cell to update paths:

```python
import config
import os

# Override paths for Colab
config.BASE_DIR = "/content"
config.DATASET_DIR = "/content/DATASET"
config.TRAIN_DIR = "/content/DATASET/TRAIN"
config.TEST_DIR = "/content/DATASET/TEST"
config.MODELS_DIR = "/content/models"
config.RESULTS_DIR = "/content/results"
config.MODEL_SAVE_PATH = "/content/models/yoga_pose_cnn.keras"
config.BEST_MODEL_PATH = "/content/models/yoga_pose_best.keras"

os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("✅ Paths configured for Colab")
```

## 6. Train the Model
```python
from train import train

model, history = train()
```

Expected output:
- Stage 1 (Feature Extraction): ~2-3 minutes on T4 GPU
- Stage 2 (Fine-Tuning): ~5-8 minutes on T4 GPU
- Expected test accuracy: **85-95%**

## 7. Evaluate the Model
```python
from evaluate import evaluate

evaluate()
```

This generates:
- `results/confusion_matrix.png`
- `results/training_curves.png`
- `results/sample_predictions.png`
- `results/classification_report.txt`

View the results:
```python
from IPython.display import Image, display

display(Image("/content/results/confusion_matrix.png"))
display(Image("/content/results/training_curves.png"))
display(Image("/content/results/sample_predictions.png"))
```

## 8. Download the Trained Model

### Option A: Download directly
```python
from google.colab import files

# Download the model files
files.download("/content/models/yoga_pose_best.keras")
files.download("/content/models/yoga_pose_cnn.keras")

# Also download results
!zip -r results.zip results/
files.download("results.zip")
```

### Option B: Save to Google Drive
```python
!cp -r /content/models/ "/content/drive/MyDrive/Project/models/"
!cp -r /content/results/ "/content/drive/MyDrive/Project/results/"
print("✅ Saved to Google Drive")
```

## 9. Set Up Locally (Your Windows Machine)

1. Place the downloaded `models/` folder in your `Project/` directory:
   ```
   Project/
   ├── models/
   │   ├── yoga_pose_cnn.keras
   │   └── yoga_pose_best.keras
   ├── app.py
   ├── pose_estimator.py
   ├── pose_rules.py
   ├── config.py
   └── ...
   ```

2. Install local dependencies:
   ```
   pip install tensorflow opencv-python mediapipe numpy
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Stand in front of your camera and try a yoga pose! 🧘

## Keyboard Controls
| Key | Action |
|-----|--------|
| `Q` | Quit the app |
| `S` | Take a screenshot |
| `R` | Reset prediction buffer & stats |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Model not found" | Make sure `models/yoga_pose_best.keras` exists in the Project folder |
| "Could not open webcam" | Close other apps using the camera, try `CAMERA_INDEX = 1` in config.py |
| Low FPS | Use `tensorflow-cpu` if you don't have a local GPU (model still runs fine) |
| Low accuracy | Try training for more epochs — increase `STAGE2_EPOCHS` to 40 in config.py |
