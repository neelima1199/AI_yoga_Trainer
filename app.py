"""
Real-Time Yoga Pose Detection & Correction App (NO MIC VERSION)
- Voice OUTPUT only (no microphone needed)
- CNN classification + MediaPipe skeleton + correction feedback
"""

import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
import pyttsx3
import random
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import *
from pose_estimator import PoseEstimator
from pose_rules import (
    get_corrections, get_joint_statuses, get_pose_description, 
    get_beginner_instructions, POSE_RULES
)


# ================= VOICE ENGINE =================
class VoiceEngine:
    def __init__(self):
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 160)
        self.tts.setProperty("volume", 1.0)

        self._speech_queue = queue.Queue()
        self._speaking = False
        self._last_spoken_time = 0
        self.SPEAK_INTERVAL = 4.0  # Seconds between each spoken correction
        self._correction_index = 0

        threading.Thread(target=self._worker, daemon=True).start()
        print("[Voice] Engine ready (no mic)")

    def _worker(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
            
        engine = pyttsx3.init()
        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            self._speaking = True
            engine.say(text)
            engine.runAndWait()
            self._speaking = False

        self._speaking = False

    def speak(self, text, clear=False):
        if clear:
            while not self._speech_queue.empty():
                try:
                    self._speech_queue.get_nowait()
                except queue.Empty:
                    break
        self._speech_queue.put(text)

    def queue_message(self, text):
        """Force queue a message to be spoken sequentially."""
        self._speech_queue.put(text)

    def speak_blocking(self, text):
        self._speech_queue.put(text)
        time.sleep(0.1)
        while self._speaking or not self._speech_queue.empty():
            time.sleep(0.05)

    def speak_correction_if_due(self, corrections):
        """Speak the most critical correction based on the user's CURRENT live position."""
        now = time.time()

        # Wait until TTS engine is free — never interrupt ongoing speech
        if self._speaking or not self._speech_queue.empty():
            return

        # Enforce minimum interval between corrections
        if now - self._last_spoken_time < self.SPEAK_INTERVAL:
            return

        incorrect = [c for c in corrections if not c["is_correct"]]

        if not incorrect:
            self.speak("Perfect form! Stay in this position.")
            self._last_spoken_time = now
            return

        # Sort by deviation — most wrong joint gets spoken first
        def deviation_score(c):
            angle = c.get("current_angle", 0)
            ideal_range = c.get("ideal_range", "")  # e.g. "80°-100°"
            try:
                parts = ideal_range.replace("°", "").split("-")
                ideal = (int(parts[0]) + int(parts[1])) / 2
                return abs(angle - ideal)
            except Exception:
                return 0

        incorrect.sort(key=deviation_score, reverse=True)
        c = incorrect[0]  # Most deviant joint right now

        spoken_msg = c.get("fix_msg", "").strip()
        if not spoken_msg:
            msg = c["message"].replace("❌ ", "").replace("✅ ", "")
            if "(now:" in msg:
                spoken_msg = msg[:msg.index("(now:")].strip()
            else:
                spoken_msg = msg

        if spoken_msg:
            self.speak(spoken_msg)  # No clear — let it queue naturally
            self._last_spoken_time = now


# ================= MAIN APP =================
class YogaPoseApp:
    def __init__(self):
        print("\n[Yoga AI Trainer] Starting...\n")

        self.model = self._load_model()
        self.estimator = PoseEstimator()
        self.voice = VoiceEngine()

        self.detected_pose = None
        self.confidence = 0.0
        self.corrections = []
        self.overall_score = 0.0

        self.prediction_buffer = []
        self.BUFFER_SIZE = 10

        # Guided Routine State Machine
        self.routine = CLASS_NAMES.copy()
        self.routine_idx = self.routine.index("tree") if "tree" in self.routine else 0
        self.target_pose = self.routine[self.routine_idx]

        self.state = "INTRO"
        self.state_start_time = time.time()
        self.INTRO_DURATION = 5
        self.SHOW_DURATION = 10 # Total 15s before COACHING starts
        self.HOLD_DURATION = 10
        self.COMPLETED_DURATION = 3

        self.load_reference_images()
        self._last_frame_warning_time = 0

    def load_reference_images(self):
        self.reference_images = {}
        for p in self.routine:
            folder = os.path.join(TRAIN_DIR, p)
            if os.path.exists(folder):
                files = os.listdir(folder)
                if files:
                    img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    if img_files:
                        img_path = os.path.join(folder, img_files[0])
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.reference_images[p] = img

    def _load_model(self):
        model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_SAVE_PATH
        print(f"[Model] Loading: {model_path}")
        return tf.keras.models.load_model(model_path)

    def _draw_hud(self, frame, score, state, time_left, corrections):
        """Draw a premium HUD overlay."""
        h, w, _ = frame.shape
        
        # Header bar (White background, Black text - Reel 1 style)
        cv2.rectangle(frame, (0, 0), (w, UI_HEADER_HEIGHT), COLOR_WHITE, -1)
        
        # Main Title
        cv2.putText(frame, "yoga trainer", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_BLACK, 2, cv2.LINE_AA)
        
        # Pose Name
        pose_text = f"TARGET: {self.target_pose.upper()}"
        cv2.putText(frame, pose_text, (400, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_BLACK, 1, cv2.LINE_AA)
        
        # Score
        score_str = f"SCORE: {score*100:.0f}%"
        cv2.putText(frame, score_str, (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2, cv2.LINE_AA)
        
        # Timer Display
        if state == "INTRO":
            timer_text = "WELCOME..."
            timer_color = COLOR_BLACK
        elif state == "SHOW_POSE":
            timer_text = f"GET READY: {int(time_left)}s"
            timer_color = (0, 0, 150)
        elif state == "GUIDE":
            timer_text = "ANALYZE & CORRECT"
            timer_color = COLOR_MAGENTA
        elif state == "HOLD":
            timer_text = f"HOLD IT! {int(time_left)}s"
            timer_color = (0, 200, 0)
        else:
            timer_text = "POSE COMPLETED!"
            timer_color = (255, 0, 0)
            
        cv2.putText(frame, timer_text, (w // 2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2, cv2.LINE_AA)
        
        # Footer — show ALL corrections as a list
        COLOR_YELLOW_FEEDBACK = (0, 220, 255)  # BGR
        
        # Calculate footer height needed for all corrections
        incorrect = [c for c in corrections if not c["is_correct"]]
        num_lines = max(1, len(incorrect))
        footer_h = max(UI_FOOTER_HEIGHT, 30 + num_lines * 30)
        footer_y = h - footer_h
        
        cv2.rectangle(frame, (0, footer_y), (w, h), COLOR_YELLOW_FEEDBACK, -1)

        if state == "HOLD":
            cv2.putText(frame, "✓  Perfect! Hold this position!", (30, footer_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLACK, 2, cv2.LINE_AA)
        elif state == "COMPLETED":
            cv2.putText(frame, "✓  Great job! Get ready for the next pose.", (30, footer_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLACK, 2, cv2.LINE_AA)
        elif score == 1.0:
            cv2.putText(frame, "✓  Perfect form! Hold steady.", (30, footer_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 0), 2, cv2.LINE_AA)
        elif incorrect:
            for i, c in enumerate(incorrect):
                msg = c.get("fix_msg", c["message"].replace("❌ ", "").replace("✅ ", ""))
                line_y = footer_y + 28 + i * 30
                if line_y < h - 5:
                    cv2.putText(frame, f"{i+1}. {msg.strip()}", (30, line_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Please position yourself in front of the camera.",
                        (30, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLACK, 2, cv2.LINE_AA)

    def _classify_pose(self, frame):
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = preprocess_input(img)

        pred = self.model.predict(img, verbose=0)[0]
        self.prediction_buffer.append(pred)

        if len(self.prediction_buffer) > self.BUFFER_SIZE:
            self.prediction_buffer.pop(0)

        avg = np.mean(self.prediction_buffer, axis=0)
        idx = np.argmax(avg)
        conf = avg[idx]

        if conf >= CNN_CONFIDENCE_THRESHOLD:
            return CLASS_NAMES[idx], conf
        return None, conf

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("[ERROR] Webcam error")
            return

        self.voice.speak("Welcome to your guided yoga session. I will be your trainer.")
        self.state_start_time = time.time()

        cv2.namedWindow("Yoga AI Mentor - Split Screen", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Yoga AI Mentor - Split Screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            # Handle Timers & States for Guided Mode
            now = time.time()
            elapsed = now - self.state_start_time
            time_left = 0
            
            if self.state == "INTRO":
                time_left = max(0, self.INTRO_DURATION - elapsed)
                if time_left == 0:
                    self.state = "SHOW_POSE"
                    self.state_start_time = now
                    self.voice.queue_message(f"Let's do the {self.target_pose} pose. Please look at the reference image.")
                    instructions = get_beginner_instructions(self.target_pose)
                    # Only queue 2 steps — rest will be said during GUIDE phase
                    for step in instructions[:2]:
                        self.voice.queue_message(step)
            
            elif self.state == "SHOW_POSE":
                time_left = max(0, self.SHOW_DURATION - elapsed)
                if time_left == 0:
                    self.state = "GUIDE"
                    self.state_start_time = now
                    # Clear any pending instructions, start coaching immediately
                    self.voice.speak("Now I will check your pose and guide you.", clear=True)
                    self.voice._last_spoken_time = 0  # Force correction on very next frame
                    
            elif self.state == "GUIDE":
                # Move to HOLD when score >= 75%
                if self.overall_score >= 0.75:
                    self.state = "HOLD"
                    self.state_start_time = now
                    self.voice.speak("Perfect! Stay in the pose for 10 seconds.")
                    
            elif self.state == "HOLD":
                time_left = max(0, self.HOLD_DURATION - elapsed)
                if self.overall_score < 0.65:  # Drop back if score falls below 65%
                    self.state = "GUIDE"
                    self.state_start_time = now
                    self.voice.speak("You lost it. Let's fix your form.")
                elif time_left == 0:
                    self.state = "COMPLETED"
                    self.state_start_time = now
                    self.voice.speak("Done.")
                    
            elif self.state == "COMPLETED":
                time_left = max(0, self.COMPLETED_DURATION - elapsed)
                if time_left == 0:
                    self.routine_idx = (self.routine_idx + 1) % len(self.routine)
                    self.target_pose = self.routine[self.routine_idx]
                    self.state = "SHOW_POSE"
                    self.state_start_time = now
                    self.voice.speak(f"Next up is the {self.target_pose} pose. Get into position.")

            ret, frame = cap.read()
            if not ret: continue

            # Ensure frame matches config dimensions
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            frame = cv2.flip(frame, 1)
            
            # 1. Classification & Pose Matching (Ignored in favor of target_pose)
            _ = self._classify_pose(frame) # Keep running to update buffer if needed
            self.detected_pose = self.target_pose
            
            # 2. Extract User Landmarks
            landmarks, raw = self.estimator.process_frame(frame)
            angles = self.estimator.calculate_joint_angles(landmarks)

            # 3. Handle Corrections & Voice for Target Pose
            if angles:
                self.corrections, self.overall_score = get_corrections(self.target_pose, angles)
                joint_status = get_joint_statuses(self.target_pose, angles)
                
                # Check for full-body visibility
                is_full_body = self.estimator.check_full_body_visibility(raw)
                
                if self.state in ["GUIDE", "HOLD"]:
                    if not is_full_body:
                        # Prioritize full-body warning
                        if now - self._last_frame_warning_time > 5.0:
                            self.voice.speak("I can only see your upper body. Please adjust your camera so I can see your full body.", clear=True)
                            self._last_frame_warning_time = now
                    else:
                        # Normal coaching
                        self.voice.speak_correction_if_due(self.corrections)
            else:
                joint_status = None
                self.overall_score = 0.0
                if self.state == "GUIDE" and (now - self._last_frame_warning_time > 5.0):
                    self.voice.speak("Please step fully into the camera view so I can see your body.", clear=True)
                    self._last_frame_warning_time = now

            # 4. PiP Canvas
            # Render User Side Full Screen
            main_view = self.estimator.draw_skeleton(frame.copy(), landmarks, joint_status)
            main_view = self.estimator.draw_angle_labels(main_view, landmarks, angles)

            # 5. Render Reference Image (Inset)
            h, w, _ = main_view.shape
            inset_w, inset_h = int(w * 0.25), int(h * 0.4) 
            
            if self.target_pose in self.reference_images and self.state != "INTRO":
                ref_img = self.reference_images[self.target_pose]
                twin_view_small = cv2.resize(ref_img, (inset_w, inset_h))
            else:
                twin_view_full = np.full((h, w, 3), COLOR_DARK_BG, dtype=np.uint8)
                msg = f"{self.target_pose.upper()} POSE" if self.state != "INTRO" else "GET READY"
                cv2.putText(twin_view_full, msg, (w // 2 - 100, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
                twin_view_small = cv2.resize(twin_view_full, (inset_w, inset_h))
            
            # Place the inset in the top left, under the header (Reel 1 aesthetic)
            inset_y = UI_HEADER_HEIGHT + 10
            inset_x = 10
            
            # Draw a border around the inset
            cv2.rectangle(twin_view_small, (0,0), (inset_w-1, inset_h-1), COLOR_WHITE, 2)
            cv2.putText(twin_view_small, "TARGET POSE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

            main_view[inset_y:inset_y+inset_h, inset_x:inset_x+inset_w] = twin_view_small
            
            # 7. Combine HUD & Final Polish
            self._draw_hud(main_view, self.overall_score, self.state, time_left, self.corrections)

            cv2.imshow("Yoga AI Mentor - Split Screen", main_view)

            # Debug detection status in console
            if landmarks:
                print(f"[OK] Landmarks: Yes | Target Pose: {self.target_pose.upper()} | Score: {self.overall_score:.2f}", end="\r")
            else:
                print("[..] Landmarks: No  | Waiting for body...                       ", end="\r")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    YogaPoseApp().run()


if __name__ == "__main__":
    main()