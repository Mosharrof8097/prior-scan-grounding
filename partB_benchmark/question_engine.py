"""
Question Template Engine for TotalSegmentator-grounded 3D VQA.
Defines canonical templates, distractors, option randomization, and counterfactual probes.
"""
import random
from typing import Dict, List, Any, Tuple

from class_metadata import CLASSES, BILATERAL_PAIRS, CRANIOCAUDAL_LEVELS, CATEGORIES


def format_structure_name(name: str) -> str:
    """Converts raw label name like 'adrenal_gland_left' to readable 'left adrenal gland'."""
    tokens = name.split("_")
    if tokens[-1] in ("left", "right"):
        return f"{tokens[-1]} " + " ".join(tokens[:-1])
    return " ".join(tokens)


class QuestionEngine:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_existence_question(self, scan_id: str, structure: str, is_present: bool) -> Dict[str, Any]:
        """Family 1: Binary Existence / FOV detection (balanced 50% chance)."""
        readable = format_structure_name(structure)
        phrasings = [
            f"Based on the anatomical coverage of this 3D CT scan, is the {readable} visualized within the field of view?",
            f"Is the {readable} completely or partially present within this volumetric CT exam?",
            f"Reviewing the full field of view, does this CT volume capture the {readable}?"
        ]
        q_text = self.rng.choice(phrasings)
        gold_answer = "Yes" if is_present else "No"
        
        # 50/50 randomized placement
        options = ["Yes", "No"]
        if self.rng.random() > 0.5:
            options = ["No", "Yes"]
        
        answer_letter = chr(ord('A') + options.index(gold_answer))
        return {
            "family": "existence",
            "scan_id": scan_id,
            "target_structure": structure,
            "question": q_text,
            "choices": {"A": options[0], "B": options[1]},
            "answer_choice": answer_letter,
            "answer_text": gold_answer,
            "chance_floor": 0.50
        }

    def build_laterality_question(self, scan_id: str, pair: Tuple[str, str, str], 
                                  vol_left: float, vol_right: float) -> Dict[str, Any]:
        """Family 2: Laterality / Bilateral volume comparison (4-choice MCQ)."""
        left_name, right_name, plural_name = pair
        phrasings = [
            f"Comparing bilateral {plural_name}, which structure exhibits the greater 3D volumetric space in this scan?",
            f"Between the left and right {plural_name}, which occupies a larger anatomical volume?",
            f"Based on 3D volumetric assessment, is the left {plural_name[:-1] if plural_name.endswith('s') else plural_name} larger than the right, or vice versa?"
        ]
        q_text = self.rng.choice(phrasings)

        if vol_left <= 0 and vol_right <= 0:
            gold = "Neither is visualized in the field of view"
        elif abs(vol_left - vol_right) / max(vol_left, vol_right, 1e-5) < 0.05:
            gold = "Approximately equal volume (within 5%)"
        elif vol_left > vol_right:
            gold = f"Left {plural_name[:-1] if plural_name.endswith('s') else plural_name}"
        else:
            gold = f"Right {plural_name[:-1] if plural_name.endswith('s') else plural_name}"

        candidate_pool = [
            f"Left {plural_name[:-1] if plural_name.endswith('s') else plural_name}",
            f"Right {plural_name[:-1] if plural_name.endswith('s') else plural_name}",
            "Approximately equal volume (within 5%)",
            "Neither is visualized in the field of view"
        ]
        
        # Ensure gold is in pool
        if gold not in candidate_pool:
            candidate_pool[2] = gold
        
        self.rng.shuffle(candidate_pool)
        letter_map = {chr(ord('A') + i): candidate_pool[i] for i in range(4)}
        gold_letter = next(k for k, v in letter_map.items() if v == gold)

        return {
            "family": "laterality_volume",
            "scan_id": scan_id,
            "structures": [left_name, right_name],
            "question": q_text,
            "choices": letter_map,
            "answer_choice": gold_letter,
            "answer_text": gold,
            "chance_floor": 0.25
        }

    def build_spatial_topography_question(self, scan_id: str, organ_a: str, organ_b: str, 
                                          z_centroid_a: float, z_centroid_b: float) -> Dict[str, Any]:
        """Family 3: 3D Relative Craniocaudal Topography (4-choice MCQ)."""
        name_a, name_b = format_structure_name(organ_a), format_structure_name(organ_b)
        phrasings = [
            f"Along the craniocaudal (superior-inferior) anatomical axis, is the centroid of the {name_a} positioned superior or inferior relative to the {name_b}?",
            f"In 3D axial coordinates, does the {name_a} lie superior or inferior to the {name_b}?",
            f"Relative to the {name_b}, what is the spatial craniocaudal orientation of the {name_a}?"
        ]
        q_text = self.rng.choice(phrasings)

        diff = z_centroid_a - z_centroid_b
        if abs(diff) < 2.0:
            gold = "At the same axial slice level"
        elif diff > 0:
            gold = "Superior"
        else:
            gold = "Inferior"

        candidates = ["Superior", "Inferior", "At the same axial slice level", "Not both visualized in the scan"]
        self.rng.shuffle(candidates)
        letter_map = {chr(ord('A') + i): candidates[i] for i in range(4)}
        gold_letter = next(k for k, v in letter_map.items() if v == gold)

        return {
            "family": "spatial_topography",
            "scan_id": scan_id,
            "structures": [organ_a, organ_b],
            "question": q_text,
            "choices": letter_map,
            "answer_choice": gold_letter,
            "answer_text": gold,
            "chance_floor": 0.25
        }

    def build_volume_ranking_question(self, scan_id: str, 
                                      four_structures_and_vols: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Family 4: Multi-structure volume ranking (4-choice MCQ, 25% chance)."""
        structures = [s for s, _ in four_structures_and_vols]
        readable = [format_structure_name(s) for s in structures]
        vols = [v for _, v in four_structures_and_vols]
        
        largest_idx = vols.index(max(vols))
        gold_structure = readable[largest_idx]

        q_text = "Which of the following segmented anatomical structures occupies the largest 3D volumetric space in this CT scan?"
        
        # Balance options
        options = list(readable)
        self.rng.shuffle(options)
        letter_map = {chr(ord('A') + i): options[i] for i in range(4)}
        gold_letter = next(k for k, v in letter_map.items() if v == gold_structure)

        return {
            "family": "volume_ranking",
            "scan_id": scan_id,
            "candidate_structures": structures,
            "question": q_text,
            "choices": letter_map,
            "answer_choice": gold_letter,
            "answer_text": gold_structure,
            "chance_floor": 0.25
        }
