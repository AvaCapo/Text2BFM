import numpy as np

body_ids = np.array([
    1,   # Pelvis
    2,   # L_Hip
    6,   # R_Hip
    10,  # Torso  -> spine1
    3,   # L_Knee
    7,   # R_Knee
    11,  # Spine  -> spine2
    4,   # L_Ankle
    8,   # R_Ankle
    12,  # Chest  -> spine3
    5,   # L_Toe  -> left_foot
    9,   # R_Toe  -> right_foot
    13,  # Neck
    15,  # L_Thorax -> left_collar
    20,  # R_Thorax -> right_collar
    14,  # Head
    16,  # L_Shoulder
    21,  # R_Shoulder
    17,  # L_Elbow
    22,  # R_Elbow
    18,  # L_Wrist
    23,  # R_Wrist
    19,  # L_Hand
    24,  # R_Hand
], dtype=np.int32)
