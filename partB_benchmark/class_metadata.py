"""
Anatomical taxonomy, lateral groupings, and spatial craniocaudal hierarchy
for the 117 TotalSegmentator v2 classes.
"""

# The 117 classes mapped from totalseg_v2_classes.json
CLASSES = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus",
    16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum", 20: "colon",
    21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left", 24: "kidney_cyst_right",
    25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4",
    29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1", 32: "vertebrae_T12",
    33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9", 36: "vertebrae_T8",
    37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
    41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7",
    45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3",
    49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart", 52: "aorta", 53: "pulmonary_vein",
    54: "brachiocephalic_trunk", 55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
    79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right", 84: "gluteus_minimus_left",
    85: "gluteus_minimus_right", 86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull",
    92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7", 99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages"
}

# Anatomical categories
CATEGORIES = {
    "organs_abdominal": [
        "liver", "spleen", "stomach", "pancreas", "gallbladder",
        "kidney_left", "kidney_right", "adrenal_gland_left", "adrenal_gland_right",
        "duodenum", "small_bowel", "colon"
    ],
    "organs_thoracic": [
        "heart", "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        "trachea", "esophagus", "thyroid_gland"
    ],
    "organs_pelvic": [
        "urinary_bladder", "prostate"
    ],
    "vascular": [
        "aorta", "superior_vena_cava", "inferior_vena_cava",
        "portal_vein_and_splenic_vein", "pulmonary_vein",
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right"
    ],
    "skeletal_axial": [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6",
        "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5", "vertebrae_S1", "sacrum", "sternum"
    ],
    "skeletal_appendicular": [
        "clavicula_left", "clavicula_right", "scapula_left", "scapula_right",
        "humerus_left", "humerus_right", "hip_left", "hip_right", "femur_left", "femur_right"
    ]
}

# Bilateral organ and bone pairs (Left, Right, readable plural name)
BILATERAL_PAIRS = [
    ("kidney_left", "kidney_right", "kidneys"),
    ("adrenal_gland_left", "adrenal_gland_right", "adrenal glands"),
    ("lung_upper_lobe_left", "lung_upper_lobe_right", "upper lung lobes"),
    ("lung_lower_lobe_left", "lung_lower_lobe_right", "lower lung lobes"),
    ("clavicula_left", "clavicula_right", "clavicles"),
    ("scapula_left", "scapula_right", "scapulae"),
    ("humerus_left", "humerus_right", "humeral bones"),
    ("hip_left", "hip_right", "hip bones"),
    ("femur_left", "femur_right", "femurs"),
    ("iliopsoas_left", "iliopsoas_right", "iliopsoas muscles"),
    ("gluteus_maximus_left", "gluteus_maximus_right", "gluteus maximus muscles")
]

# Canonical anatomical levels (superior to inferior ordering)
CRANIOCAUDAL_LEVELS = [
    ("brain", 10),
    ("skull", 10),
    ("thyroid_gland", 9),
    ("trachea", 8),
    ("heart", 7),
    ("lung_upper_lobe_left", 7),
    ("aorta", 6),
    ("esophagus", 6),
    ("liver", 5),
    ("spleen", 5),
    ("stomach", 5),
    ("pancreas", 4),
    ("kidney_left", 3),
    ("kidney_right", 3),
    ("duodenum", 3),
    ("colon", 2),
    ("small_bowel", 2),
    ("urinary_bladder", 1),
    ("prostate", 1),
    ("femur_left", 0)
]
