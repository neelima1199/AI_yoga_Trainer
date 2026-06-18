"""
Pose Rules Engine — Ideal Angle Database & Correction Logic.
Defines what "correct" looks like for each yoga pose and generates
human-readable corrective feedback.
"""
from config import ANGLE_TOLERANCE


# ==============================================================================
# IDEAL ANGLE DATABASE
# ==============================================================================
# Each pose defines ideal angle ranges for key joints.
# Format: { joint_name: (min_angle, max_angle, friendly_name, correction_tip) }
# Only joints that are critical for that specific pose are included.
# ==============================================================================

POSE_RULES = {
    # ------------------------------------------------------------------
    # DOWNDOG (Adho Mukha Svanasana) — Inverted V shape
    # ------------------------------------------------------------------
    "downdog": {
        "left_elbow": {
            "min": 160, "max": 180,
            "name": "Left Elbow",
            "correct_msg": "Arms are straight",
            "fix_msg": "Straighten your left arm fully",
        },
        "right_elbow": {
            "min": 160, "max": 180,
            "name": "Right Elbow",
            "correct_msg": "Arms are straight",
            "fix_msg": "Straighten your right arm fully",
        },
        "left_knee": {
            "min": 155, "max": 180,
            "name": "Left Knee",
            "correct_msg": "Legs are straight",
            "fix_msg": "Try to straighten your left leg",
        },
        "right_knee": {
            "min": 155, "max": 180,
            "name": "Right Knee",
            "correct_msg": "Legs are straight",
            "fix_msg": "Try to straighten your right leg",
        },
        "left_hip": {
            "min": 50, "max": 100,
            "name": "Left Hip",
            "correct_msg": "Great hip angle — nice V shape!",
            "fix_msg": "Push your hips up and back to form a V",
        },
        "right_hip": {
            "min": 50, "max": 100,
            "name": "Right Hip",
            "correct_msg": "Great hip angle — nice V shape!",
            "fix_msg": "Push your hips up and back to form a V",
        },
        "left_shoulder": {
            "min": 155, "max": 180,
            "name": "Left Shoulder",
            "correct_msg": "Arms aligned with torso",
            "fix_msg": "Extend your arms in line with your back",
        },
        "right_shoulder": {
            "min": 155, "max": 180,
            "name": "Right Shoulder",
            "correct_msg": "Arms aligned with torso",
            "fix_msg": "Extend your arms in line with your back",
        },
    },
    
    # ------------------------------------------------------------------
    # GODDESS (Utkata Konasana) — Wide squat with bent arms
    # ------------------------------------------------------------------
    "goddess": {
        "left_knee": {
            "min": 75, "max": 120,
            "name": "Left Knee",
            "correct_msg": "Good knee bend",
            "fix_msg": "Bend your left knee deeper — aim for 90 degrees",
        },
        "right_knee": {
            "min": 75, "max": 120,
            "name": "Right Knee",
            "correct_msg": "Good knee bend",
            "fix_msg": "Bend your right knee deeper — aim for 90 degrees",
        },
        "left_elbow": {
            "min": 70, "max": 110,
            "name": "Left Elbow",
            "correct_msg": "Arms bent correctly",
            "fix_msg": "Bend your left elbow to about 90 degrees",
        },
        "right_elbow": {
            "min": 70, "max": 110,
            "name": "Right Elbow",
            "correct_msg": "Arms bent correctly",
            "fix_msg": "Bend your right elbow to about 90 degrees",
        },
        "left_hip": {
            "min": 70, "max": 130,
            "name": "Left Hip",
            "correct_msg": "Wide stance looks good",
            "fix_msg": "Widen your stance and lower your hips",
        },
        "right_hip": {
            "min": 70, "max": 130,
            "name": "Right Hip",
            "correct_msg": "Wide stance looks good",
            "fix_msg": "Widen your stance and lower your hips",
        },
        "left_shoulder": {
            "min": 60, "max": 120,
            "name": "Left Shoulder",
            "correct_msg": "Arms at shoulder height",
            "fix_msg": "Raise your arms to shoulder height",
        },
        "right_shoulder": {
            "min": 60, "max": 120,
            "name": "Right Shoulder",
            "correct_msg": "Arms at shoulder height",
            "fix_msg": "Raise your arms to shoulder height",
        },
    },
    
    # ------------------------------------------------------------------
    # PLANK (Phalakasana) — Straight line from head to heels
    # ------------------------------------------------------------------
    "plank": {
        "left_elbow": {
            "min": 155, "max": 180,
            "name": "Left Elbow",
            "correct_msg": "Arms straight — good support",
            "fix_msg": "Straighten your left arm — lock the elbow",
        },
        "right_elbow": {
            "min": 155, "max": 180,
            "name": "Right Elbow",
            "correct_msg": "Arms straight — good support",
            "fix_msg": "Straighten your right arm — lock the elbow",
        },
        "left_hip": {
            "min": 155, "max": 180,
            "name": "Left Hip",
            "correct_msg": "Body is in a straight line!",
            "fix_msg": "Don't let your hips sag — engage your core",
        },
        "right_hip": {
            "min": 155, "max": 180,
            "name": "Right Hip",
            "correct_msg": "Body is in a straight line!",
            "fix_msg": "Don't let your hips sag — engage your core",
        },
        "left_knee": {
            "min": 155, "max": 180,
            "name": "Left Knee",
            "correct_msg": "Legs straight",
            "fix_msg": "Straighten your left leg fully",
        },
        "right_knee": {
            "min": 155, "max": 180,
            "name": "Right Knee",
            "correct_msg": "Legs straight",
            "fix_msg": "Straighten your right leg fully",
        },
        "left_shoulder": {
            "min": 60, "max": 110,
            "name": "Left Shoulder",
            "correct_msg": "Shoulders over wrists — great",
            "fix_msg": "Align your shoulders directly above your wrists",
        },
        "right_shoulder": {
            "min": 60, "max": 110,
            "name": "Right Shoulder",
            "correct_msg": "Shoulders over wrists — great",
            "fix_msg": "Align your shoulders directly above your wrists",
        },
    },
    
    # ------------------------------------------------------------------
    # TREE (Vrksasana) — Standing balance with one foot on inner thigh
    # ------------------------------------------------------------------
    "tree": {
        "left_knee": {
            "min": 155, "max": 180,
            "name": "Standing Knee (L)",
            "correct_msg": "Standing leg is strong and straight",
            "fix_msg": "Straighten your standing leg — keep it strong",
        },
        "right_knee": {
            "min": 155, "max": 180,
            "name": "Standing Knee (R)",
            "correct_msg": "Standing leg is strong and straight",
            "fix_msg": "Straighten your standing leg — keep it strong",
        },
        "left_shoulder": {
            "min": 150, "max": 180,
            "name": "Left Shoulder",
            "correct_msg": "Arms reaching overhead nicely",
            "fix_msg": "Reach your arms straight up overhead",
        },
        "right_shoulder": {
            "min": 150, "max": 180,
            "name": "Right Shoulder",
            "correct_msg": "Arms reaching overhead nicely",
            "fix_msg": "Reach your arms straight up overhead",
        },
        "left_elbow": {
            "min": 155, "max": 180,
            "name": "Left Elbow",
            "correct_msg": "Arms straight overhead",
            "fix_msg": "Straighten your arms overhead",
        },
        "right_elbow": {
            "min": 155, "max": 180,
            "name": "Right Elbow",
            "correct_msg": "Arms straight overhead",
            "fix_msg": "Straighten your arms overhead",
        },
        # Hip angle will vary depending on which leg is the standing leg
        "left_hip": {
            "min": 140, "max": 180,
            "name": "Left Hip",
            "correct_msg": "Hips level — great balance",
            "fix_msg": "Keep your hips level and square",
        },
        "right_hip": {
            "min": 140, "max": 180,
            "name": "Right Hip",
            "correct_msg": "Hips level — great balance",
            "fix_msg": "Keep your hips level and square",
        },
    },
    
    # ------------------------------------------------------------------
    # WARRIOR II (Virabhadrasana II) — Wide lunge with arms extended
    # ------------------------------------------------------------------
    "warrior2": {
        "left_elbow": {
            "min": 155, "max": 180,
            "name": "Left Elbow",
            "correct_msg": "Left arm extended — great",
            "fix_msg": "Fully extend your left arm out to the side",
        },
        "right_elbow": {
            "min": 155, "max": 180,
            "name": "Right Elbow",
            "correct_msg": "Right arm extended — great",
            "fix_msg": "Fully extend your right arm out to the side",
        },
        "left_shoulder": {
            "min": 70, "max": 120,
            "name": "Left Shoulder",
            "correct_msg": "Arms at shoulder height",
            "fix_msg": "Raise your arms to shoulder height",
        },
        "right_shoulder": {
            "min": 70, "max": 120,
            "name": "Right Shoulder",
            "correct_msg": "Arms at shoulder height",
            "fix_msg": "Raise your arms to shoulder height",
        },
        # Front knee should be deeply bent
        "left_knee": {
            "min": 75, "max": 120,
            "name": "Front Knee (L)",
            "correct_msg": "Good front knee bend",
            "fix_msg": "Bend your front knee to about 90 degrees",
        },
        "right_knee": {
            "min": 75, "max": 120,
            "name": "Front Knee (R)",
            "correct_msg": "Good front knee bend",
            "fix_msg": "Bend your front knee to about 90 degrees",
        },
        "left_hip": {
            "min": 70, "max": 130,
            "name": "Left Hip",
            "correct_msg": "Hips open — beautiful form",
            "fix_msg": "Open your hips more to the side",
        },
        "right_hip": {
            "min": 70, "max": 130,
            "name": "Right Hip",
            "correct_msg": "Hips open — beautiful form",
            "fix_msg": "Open your hips more to the side",
        },
    },
}


def get_corrections(pose_name, current_angles):
    """
    Compare current joint angles against ideal angles for the detected pose.
    
    Args:
        pose_name: str — one of the pose names (e.g., "warrior2")
        current_angles: dict — {joint_name: measured_angle}
    
    Returns:
        corrections: list of dicts with keys:
            - joint: str (joint name)
            - is_correct: bool
            - message: str (feedback message)
            - current_angle: float
            - ideal_range: str (e.g., "80°-100°")
        overall_score: float (0.0 - 1.0) — proportion of joints in correct position
    """
    if pose_name not in POSE_RULES:
        return [], 0.0
    
    rules = POSE_RULES[pose_name]
    corrections = []
    correct_count = 0
    total_count = 0
    
    for joint_name, rule in rules.items():
        if joint_name not in current_angles or current_angles[joint_name] is None:
            continue
        
        total_count += 1
        angle = current_angles[joint_name]
        min_angle = rule["min"] - ANGLE_TOLERANCE
        max_angle = rule["max"] + ANGLE_TOLERANCE
        
        is_correct = min_angle <= angle <= max_angle
        
        if is_correct:
            correct_count += 1
            message = f"✅ {rule['correct_msg']}"
        else:
            # Dynamic correction based on me/the user's specific angle
            friendly_name = rule['name'].lower().replace(" (l)", "").replace(" (r)", "")
            if angle < min_angle:
                # Angle is too tight
                if "shoulder" in joint_name:
                    action = "open or raise"
                elif "hip" in joint_name:
                    action = "straighten or push up" if pose_name == "plank" else "straighten"
                else:
                    action = "straighten"
            else:
                # Angle is too wide
                if "shoulder" in joint_name:
                    action = "lower"
                elif "hip" in joint_name:
                    action = "bend or lower" if pose_name == "plank" else "bend"
                else:
                    action = "bend"
                    
            message = f"❌ {action} your {friendly_name} (now: {angle:.0f}°, target: {rule['min']}°-{rule['max']}°)"
        
        corrections.append({
            "joint": joint_name,
            "friendly_name": rule["name"],
            "is_correct": is_correct,
            "message": message,
            "fix_msg": rule["fix_msg"],
            "current_angle": angle,
            "ideal_range": f"{rule['min']}°-{rule['max']}°",
        })
    
    overall_score = correct_count / total_count if total_count > 0 else 0.0
    
    return corrections, overall_score


def get_joint_statuses(pose_name, current_angles):
    """
    Get a simple dict of {joint_name: is_correct} for skeleton coloring.
    
    Args:
        pose_name: str
        current_angles: dict
    
    Returns:
        dict {joint_name: bool}
    """
    corrections, _ = get_corrections(pose_name, current_angles)
    return {c["joint"]: c["is_correct"] for c in corrections}


def get_pose_description(pose_name):
    """Get a short description of the ideal form for a pose."""
    descriptions = {
        "downdog":  "Inverted V — straight arms & legs, hips high",
        "goddess":  "Wide squat — knees bent, arms at 90 degrees",
        "plank":    "Straight line — head to heels, core engaged",
        "tree":     "Balance — standing leg straight, arms overhead",
        "warrior2": "Wide lunge — front knee 90°, arms extended",
    }
    return descriptions.get(pose_name, "Unknown pose")


# ==============================================================================
# BEGINNER STEP-BY-STEP INSTRUCTIONS (spoken aloud via TTS)
# ==============================================================================

BEGINNER_INSTRUCTIONS = {
    "downdog": [
        "Let's learn Downward Dog, also called Adho Mukha Svanasana.",
        "Step 1: Start on your hands and knees on the floor. Place your hands shoulder width apart.",
        "Step 2: Tuck your toes under and slowly lift your knees off the floor.",
        "Step 3: Push your hips up and back toward the ceiling, forming an upside down V shape with your body.",
        "Step 4: Straighten your arms fully. Press your palms flat into the ground.",
        "Step 5: Try to straighten your legs. It's okay if your heels don't touch the floor.",
        "Step 6: Let your head hang naturally between your arms. Look toward your belly button.",
        "Step 7: Hold this position. I will now check your form and guide you.",
    ],

    "goddess": [
        "Let's learn Goddess Pose, also called Utkata Konasana.",
        "Step 1: Stand up straight with your feet wide apart, about 3 to 4 feet.",
        "Step 2: Turn your toes outward at about 45 degrees.",
        "Step 3: Bend your knees deeply, lowering your hips. Try to get your thighs parallel to the floor.",
        "Step 4: Make sure your knees point in the same direction as your toes.",
        "Step 5: Raise your arms out to the sides at shoulder height.",
        "Step 6: Bend your elbows to 90 degrees, with your palms facing forward.",
        "Step 7: Keep your back straight and your core engaged.",
        "Step 8: Hold this position. I will now check your form and guide you.",
    ],

    "plank": [
        "Let's learn Plank Pose, also called Phalakasana.",
        "Step 1: Start by getting on your hands and knees on the floor.",
        "Step 2: Place your hands directly under your shoulders, shoulder width apart.",
        "Step 3: Step your feet back one at a time, extending your legs fully behind you.",
        "Step 4: Keep your arms straight. Do not bend your elbows.",
        "Step 5: Your body should form one straight line from your head to your heels.",
        "Step 6: Engage your core muscles. Don't let your hips sag down or pike up.",
        "Step 7: Look at the floor slightly ahead of your hands.",
        "Step 8: Hold this position. I will now check your form and guide you.",
    ],

    "tree": [
        "Let's learn Tree Pose, also called Vrksasana. This is a balance pose.",
        "Step 1: Stand up straight with both feet together.",
        "Step 2: Shift your weight onto your left foot. Feel the ground firmly under you.",
        "Step 3: Slowly lift your right foot and place it on your inner left thigh or calf. Avoid placing it on your knee.",
        "Step 4: Press your right foot into your left leg, and your left leg back into your foot.",
        "Step 5: Find a spot on the wall to focus your eyes on. This helps with balance.",
        "Step 6: Bring your hands together at your chest in a prayer position.",
        "Step 7: When you feel stable, slowly raise your arms straight up overhead.",
        "Step 8: Hold this position. I will now check your form and guide you.",
    ],

    "warrior2": [
        "Let's learn Warrior 2, also called Virabhadrasana 2.",
        "Step 1: Stand up straight, then step your feet wide apart, about 4 feet.",
        "Step 2: Turn your right foot out 90 degrees so your toes point to the right.",
        "Step 3: Keep your left foot slightly angled inward.",
        "Step 4: Bend your right knee deeply until it is directly above your right ankle. Aim for a 90 degree bend.",
        "Step 5: Keep your left leg completely straight and strong.",
        "Step 6: Extend both arms out to the sides at shoulder height, parallel to the floor.",
        "Step 7: Keep your arms straight. Turn your head and gaze over your right fingertips.",
        "Step 8: Keep your hips open facing the side, not the front.",
        "Step 9: Hold this position. I will now check your form and guide you.",
    ],
}


def get_beginner_instructions(pose_name):
    """
    Get step-by-step beginner instructions for a pose.

    Args:
        pose_name: str — one of the 5 pose names

    Returns:
        list of instruction strings, or empty list if pose unknown
    """
    return BEGINNER_INSTRUCTIONS.get(pose_name, [])


if __name__ == "__main__":
    # Quick test: simulate angle checking
    print("=" * 60)
    print("POSE RULES ENGINE — TEST")
    print("=" * 60)
    
    # Simulate a warrior2 pose with some incorrect angles
    test_angles = {
        "left_elbow": 172.0,      # Correct (160-180 range)
        "right_elbow": 145.0,     # Incorrect — not fully extended
        "left_shoulder": 90.0,    # Correct (70-120 range)
        "right_shoulder": 85.0,   # Correct
        "left_knee": 130.0,       # Incorrect — not bent enough
        "right_knee": 92.0,       # Correct (75-120 range)
        "left_hip": 110.0,        # Correct (70-130 range)
        "right_hip": 105.0,       # Correct
    }
    
    corrections, score = get_corrections("warrior2", test_angles)
    
    print(f"\nPose: WARRIOR II")
    print(f"Overall Score: {score:.0%}\n")
    
    for c in corrections:
        status = "✅" if c["is_correct"] else "❌"
        print(f"  {status} {c['friendly_name']:20s} | "
              f"Current: {c['current_angle']:6.1f}° | "
              f"Ideal: {c['ideal_range']}")
    
    print(f"\n--- Correction Messages ---")
    for c in corrections:
        if not c["is_correct"]:
            print(f"  {c['message']}")
