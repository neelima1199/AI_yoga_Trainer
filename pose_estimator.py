"""
Pose Estimator using MediaPipe Tasks API (v0.10.33+).
Extracts 33 body landmarks, calculates joint angles, and draws skeleton overlay.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from config import (
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    COLOR_ACCENT, COLOR_IDEAL, COLOR_BLUE, COLOR_WHITE, COLOR_DARK_BG
)


# MediaPipe landmark indices for readability
class LandmarkIndex:
    """Named indices for MediaPipe Pose landmarks (33 total)."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# Joint definitions: (name, point_a, point_b (joint), point_c)
JOINT_DEFINITIONS = {
    "left_elbow":    (LandmarkIndex.LEFT_SHOULDER,  LandmarkIndex.LEFT_ELBOW,    LandmarkIndex.LEFT_WRIST),
    "right_elbow":   (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW,   LandmarkIndex.RIGHT_WRIST),
    "left_shoulder":  (LandmarkIndex.LEFT_HIP,       LandmarkIndex.LEFT_SHOULDER,  LandmarkIndex.LEFT_ELBOW),
    "right_shoulder": (LandmarkIndex.RIGHT_HIP,      LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
    "left_hip":      (LandmarkIndex.LEFT_SHOULDER,  LandmarkIndex.LEFT_HIP,      LandmarkIndex.LEFT_KNEE),
    "right_hip":     (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP,     LandmarkIndex.RIGHT_KNEE),
    "left_knee":     (LandmarkIndex.LEFT_HIP,       LandmarkIndex.LEFT_KNEE,     LandmarkIndex.LEFT_ANKLE),
    "right_knee":    (LandmarkIndex.RIGHT_HIP,      LandmarkIndex.RIGHT_KNEE,    LandmarkIndex.RIGHT_ANKLE),
}

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A-B-C.

    Args:
        a: (x, y) coordinates of first point
        b: (x, y) coordinates of the joint (vertex)
        c: (x, y) coordinates of third point

    Returns:
        Angle in degrees [0, 180]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return round(angle, 1)


class PoseEstimator:
    """
    Wrapper around MediaPipe PoseLandmarker (Tasks API) for landmark detection.
    """

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "pose_landmarker_lite.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at {model_path}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', "
                "'models/pose_landmarker_lite.task')\""
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # Smoothing state (EMA filter)
        self.prev_landmarks_px = None
        self.prev_landmarks_float = None  # Float precision for accurate blending
        self.SMOOTH_ALPHA = 0.1  # Very stable — only 10% of new frame affects position
        
        print("[OK] MediaPipe Pose Estimator initialized (Tasks API with smoothing)")

    def process_frame(self, frame):
        """
        Process a single BGR frame and extract pose landmarks.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            landmarks_px: List of (x, y) pixel coordinates for each landmark,
                          or None if no pose detected
            raw_landmarks: Raw normalized landmarks list, or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            # Don't reset prev — keeps last good frame to avoid snapping
            return None, None

        # Get the first detected pose
        pose_landmarks = result.pose_landmarks[0]

        # Convert normalized landmarks to pixel coordinates (as floats for precision)
        h, w, _ = frame.shape
        raw_float = [(lm.x * w, lm.y * h) for lm in pose_landmarks]

        # Apply EMA smoothing in float space to avoid integer rounding jitter
        if self.prev_landmarks_float is None:
            self.prev_landmarks_float = raw_float
            smoothed_px = [(int(x), int(y)) for x, y in raw_float]
            self.prev_landmarks_px = smoothed_px
            return smoothed_px, pose_landmarks

        alpha = self.SMOOTH_ALPHA
        smoothed_float = []
        for i in range(len(raw_float)):
            cx, cy = raw_float[i]
            px, py = self.prev_landmarks_float[i]
            sx = cx * alpha + px * (1 - alpha)
            sy = cy * alpha + py * (1 - alpha)
            smoothed_float.append((sx, sy))

        self.prev_landmarks_float = smoothed_float
        smoothed_px = [(int(x), int(y)) for x, y in smoothed_float]
        self.prev_landmarks_px = smoothed_px
        return smoothed_px, pose_landmarks

    def calculate_joint_angles(self, landmarks_px):
        """
        Calculate all defined joint angles from pixel landmarks.

        Args:
            landmarks_px: List of (x, y) pixel coordinates

        Returns:
            dict: {joint_name: angle_in_degrees}
        """
        if landmarks_px is None:
            return {}

        angles = {}
        for joint_name, (idx_a, idx_b, idx_c) in JOINT_DEFINITIONS.items():
            try:
                angle = calculate_angle(
                    landmarks_px[idx_a],
                    landmarks_px[idx_b],
                    landmarks_px[idx_c]
                )
                angles[joint_name] = angle
            except (IndexError, TypeError):
                angles[joint_name] = None

        return angles

    def draw_skeleton(self, frame, landmarks_px, joint_statuses=None):
        """
        Draw the pose skeleton on the frame matching Instagram aesthetic.
        """
        if landmarks_px is None:
            return frame

        BONE_COLOR = (255, 255, 255) # White
        DEFAULT_JOINT_OUTER = (180, 50, 0)  # Dark Blue (BGR)

        # Draw connections (bones)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start = landmarks_px[start_idx]
                end = landmarks_px[end_idx]
                cv2.line(frame, start, end, BONE_COLOR, 3, cv2.LINE_AA)

        # Build status mapping for joint overrides
        status_map = {}
        if joint_statuses:
            joint_to_landmark = {
                "left_elbow": LandmarkIndex.LEFT_ELBOW,
                "right_elbow": LandmarkIndex.RIGHT_ELBOW,
                "left_shoulder": LandmarkIndex.LEFT_SHOULDER,
                "right_shoulder": LandmarkIndex.RIGHT_SHOULDER,
                "left_hip": LandmarkIndex.LEFT_HIP,
                "right_hip": LandmarkIndex.RIGHT_HIP,
                "left_knee": LandmarkIndex.LEFT_KNEE,
                "right_knee": LandmarkIndex.RIGHT_KNEE,
            }
            for j_name, is_curr in joint_statuses.items():
                if j_name in joint_to_landmark:
                    status_map[joint_to_landmark[j_name]] = is_curr

        # Draw landmark points
        for i, (x, y) in enumerate(landmarks_px):
            if i < 11 or i > 28:  # Skip face and extremities
                continue
                
            outer_color = DEFAULT_JOINT_OUTER
            if i in status_map:
                outer_color = (0, 255, 0) if status_map[i] else (0, 0, 255) # Green / Red
                
            # Aesthetic ring with inner core
            cv2.circle(frame, (x, y), 8, outer_color, 3, cv2.LINE_AA) # Thick colored ring
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA) # Solid white center dot

        return frame

    def draw_angle_labels(self, frame, landmarks_px, angles):
        """
        Draw angle values and arcs near their respective joints.
        """
        if landmarks_px is None or not angles:
            return frame

        TEXT_COLOR = (255, 0, 255) # Bright Magenta (BGR)
        ARC_COLOR = (255, 255, 255)

        for joint_name, angle in angles.items():
            if angle is not None and joint_name in JOINT_DEFINITIONS:
                idx_a, idx_b, idx_c = JOINT_DEFINITIONS[joint_name]
                
                if idx_b < len(landmarks_px) and idx_a < len(landmarks_px) and idx_c < len(landmarks_px):
                    p_a = landmarks_px[idx_a]
                    p_b = landmarks_px[idx_b]
                    p_c = landmarks_px[idx_c]
                    x, y = p_b
                    
                    # Compute rotation path for the arc
                    theta1 = np.degrees(np.arctan2(p_a[1] - y, p_a[0] - x))
                    theta2 = np.degrees(np.arctan2(p_c[1] - y, p_c[0] - x))
                    
                    t_min = min(theta1, theta2)
                    t_max = max(theta1, theta2)
                    
                    if t_max - t_min > 180:
                        start_angle = t_max
                        end_angle = t_min + 360
                    else:
                        start_angle = t_min
                        end_angle = t_max
                        
                    # Draw Arc inside the joint
                    cv2.ellipse(frame, (x, y), (15, 15), 0, start_angle, end_angle, ARC_COLOR, 2, cv2.LINE_AA)
                    
                    # Draw Text like "32deg"
                    text = f"{angle:.0f}deg"
                    cv2.putText(
                        frame, text, (x + 18, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA  # Thick Black shadow/outline
                    )
                    cv2.putText(
                        frame, text, (x + 18, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        TEXT_COLOR, 1, cv2.LINE_AA # Magenta fill
                    )

        return frame

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    def get_corrected_landmarks(self, landmarks_px, pose_rules, current_angles):
        """
        Creates a 'Perfect' version of the user's landmarks by adjusting joints 
        to match the target angles from pose_rules.
        """
        if not landmarks_px or not pose_rules:
            return landmarks_px

        corrected = list(landmarks_px)
        
        # Simple bone-based adjustment for key joints
        # Only adjusting the end-points of the bones defined in rules
        for joint_name, rule in pose_rules.items():
            if joint_name not in JOINT_DEFINITIONS:
                continue
            
            # Root, Pivot, End
            idx_a, idx_p, idx_e = JOINT_DEFINITIONS[joint_name]
            
            p = np.array(landmarks_px[idx_p])
            a = np.array(landmarks_px[idx_a])
            e = np.array(landmarks_px[idx_e])
            
            # Target angle (midpoint of rule)
            target_angle = (rule["min"] + rule["max"]) / 2
            
            # Calculate current vectors
            v_pa = a - p
            v_pe = e - p
            
            # Use current length of bone p->e
            length_pe = np.linalg.norm(v_pe)
            if length_pe < 1: continue
            
            # Current angle
            curr_angle = current_angles.get(joint_name)
            if curr_angle is None: continue
            
            # Rotation needed (in radians)
            # This is a simplification: we rotate v_pe in the 2D plane to match target_angle
            angle_diff = np.radians(target_angle - curr_angle)
            
            # Determine direction of rotation based on cross product or current relative angle
            # For simplicity in 2D, we try alternating and check which one gets closer
            # (In a more robust version, we'd use atan2 more carefully)
            
            # Rotate vector v_pe by angle_diff
            c, s = np.cos(angle_diff), np.sin(angle_diff)
            rot_matrix = np.array(((c, -s), (s, c)))
            v_pe_new = rot_matrix.dot(v_pe)
            
            # Update the end landmark
            corrected[idx_e] = (int(p[0] + v_pe_new[0]), int(p[1] + v_pe_new[1]))
            
        return corrected

    def check_full_body_visibility(self, raw_landmarks):
        """
        Check if the essential lower body landmarks (knees/ankles) are visible.
        
        Returns:
            bool: True if ankles are visible, False otherwise.
        """
        if not raw_landmarks:
            return False
            
        # Landmarks: 25, 26 (Knees), 27, 28 (Ankles)
        # We prioritize Ankles as they represent the 'full' extent of the visible body
        ankle_l = raw_landmarks[27]
        ankle_r = raw_landmarks[28]
        
        VIS_THRESHOLD = 0.5
        if ankle_l.visibility < VIS_THRESHOLD or ankle_r.visibility < VIS_THRESHOLD:
            return False
            
        return True

    def draw_solid_body(self, frame, landmarks_px, color=(200, 160, 120)):
        """
        Draw a 'handsome' athletic mannequin body with V-taper proportions.
        """
        if landmarks_px is None:
            return frame

        overlay = frame.copy()
        
        # Premium Colors
        SKIN_COLOR = (210, 180, 140)  # Tan/Skin tone
        ACCENT_COLOR = (255, 215, 0)   # Gold highlight
        
        # 1. TORSO (Polygonal Fill with Athletic V-Taper)
        try:
            ls, rs = np.array(landmarks_px[11]), np.array(landmarks_px[12])
            lh, rh = np.array(landmarks_px[23]), np.array(landmarks_px[24])
            
            # Slightly widen shoulders for athletic look
            shoulder_vec = rs - ls
            ls_wide = ls - (shoulder_vec * 0.1)
            rs_wide = rs + (shoulder_vec * 0.1)
            
            torso_pts = np.array([ls_wide, rs_wide, rh, lh], np.int32)
            cv2.fillPoly(overlay, [torso_pts], SKIN_COLOR)
            
            # Chest highlight
            mid_shoulder = (ls_wide + rs_wide) / 2
            mid_torso = (lh + rh) / 2
            chest_pts = np.array([ls_wide, rs_wide, mid_torso + (mid_shoulder - mid_torso) * 0.3], np.int32)
            cv2.fillPoly(overlay, [chest_pts], (SKIN_COLOR[0]+10, SKIN_COLOR[1]+10, SKIN_COLOR[2]+10))
        except (IndexError, TypeError):
            pass

        # 2. LIMBS (Thick, muscular rendering)
        limb_thickness = 28
        # Arms (Upper & Lower)
        for start_idx, end_idx in [(11, 13), (13, 15), (12, 14), (14, 16)]:
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start, end = np.array(landmarks_px[start_idx]), np.array(landmarks_px[end_idx])
                # Draw thick muscular segment
                cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), SKIN_COLOR, limb_thickness, cv2.LINE_AA)
                cv2.circle(overlay, tuple(start.astype(int)), limb_thickness // 2, SKIN_COLOR, -1, cv2.LINE_AA)

        # Legs
        for start_idx, end_idx in [(23, 25), (25, 27), (24, 26), (26, 28)]:
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start, end = np.array(landmarks_px[start_idx]), np.array(landmarks_px[end_idx])
                cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), SKIN_COLOR, limb_thickness + 8, cv2.LINE_AA)
                cv2.circle(overlay, tuple(start.astype(int)), (limb_thickness + 8) // 2, SKIN_COLOR, -1, cv2.LINE_AA)

        # 3. HEAD & NECK
        try:
            nose = np.array(landmarks_px[LandmarkIndex.NOSE])
            ls, rs = np.array(landmarks_px[11]), np.array(landmarks_px[12])
            shoulder_mid = (ls + rs) / 2
            
            # Neck
            neck_top = nose + (shoulder_mid - nose) * 0.6
            cv2.line(overlay, tuple(neck_top.astype(int)), tuple(shoulder_mid.astype(int)), SKIN_COLOR, 15, cv2.LINE_AA)
            
            # Handsome jawline head shape
            head_w = int(np.linalg.norm(ls - rs) * 0.25)
            head_h = int(head_w * 1.4)
            cv2.ellipse(overlay, tuple(nose.astype(int)), (head_w, head_h), 0, 0, 360, SKIN_COLOR, -1, cv2.LINE_AA)
            
            # Add stylized hair/eye shadow for 'handsome' look
            cv2.ellipse(overlay, tuple(nose.astype(int)), (head_w, head_h // 2), 0, 180, 360, (50, 50, 50), -1, cv2.LINE_AA)
        except (IndexError, TypeError):
            pass

        # 4. BLEND & GLOW
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Outlines
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start, end = landmarks_px[start_idx], landmarks_px[end_idx]
                cv2.line(frame, start, end, (255, 255, 255), 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    # Quick test: run pose estimation on webcam
    print("[Camera] Testing Pose Estimator -- press 'q' to quit")

    estimator = PoseEstimator()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_px, _ = estimator.process_frame(frame)
        angles = estimator.calculate_joint_angles(landmarks_px)

        frame = estimator.draw_solid_body(frame, landmarks_px)
        frame = estimator.draw_angle_labels(frame, landmarks_px, angles)

        cv2.imshow("Pose Estimator Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    estimator.release()
