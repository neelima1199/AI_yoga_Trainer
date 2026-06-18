---
description: Workflow for Training and Implementing the Yoga AI Pose Trainer
---

## 1. Data Preparation
Before training, ensure your dataset is organized.

// turbo
1. Verify the structure of the `DATASET` directory.
   ```powershell
   ls DATASET/TRAIN
   ls DATASET/TEST
   ```
2. Each folder inside `TRAIN` and `TEST` should represent a yoga pose.

## 2. Model Training (Google Colab Recommended)
Training requires a GPU for efficient processing.

1. Zip your project files for upload:
   ```powershell
   Compress-Archive -Path config.py, data_loader.py, model.py, train.py, DATASET -DestinationPath YogaProject.zip
   ```
2. Upload `YogaProject.zip` to Google Colab.
3. Unzip and run the training script:
   ```python
   !unzip YogaProject.zip
   !python train.py
   ```
4. Download the resulting `models/yoga_pose_best.keras` file.

## 3. Local Implementation & Running
Once you have the trained model, you can run the trainer locally.

1. Ensure the model is in the `models/` directory:
   ```powershell
   ls models/
   ```
2. Launch the real-time application:
   ```powershell
   python app.py
   ```

## 4. Evaluation
To see how well your model is performing with detailed metrics:

1. Run the evaluation script:
   ```powershell
   python evaluate.py
   ```
2. Check the `results/` folder for confusion matrices and training curves.
