import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Optional

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# ------------------------------------------------
# Load SAM model
# ------------------------------------------------
def _resolve_sam_checkpoint(model_path=None) -> Path:
    """Resolve SAM checkpoint path with deterministic priority.

    Priority:
    1) explicit `model_path` argument
    2) `SAM_CHECKPOINT` environment variable
    3) `<repo>/models/sam_vit_b.pth`
    """
    if model_path is not None:
        return Path(model_path).expanduser().resolve()

    env_ckpt = os.getenv("SAM_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt).expanduser().resolve()

    root = Path(__file__).resolve().parents[4]
    return (root / "models" / "sam_vit_b.pth").resolve()


def load_sam(model_path=None):
    model_path = _resolve_sam_checkpoint(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {model_path}. "
            "Expected file: C:/Users/robyn/PlantDisease/models/sam_vit_b.pth"
        )
    if not model_path.is_file():
        raise ValueError(f"SAM checkpoint path is not a file: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,
        points_per_batch=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=256,
    )
    predictor = SamPredictor(sam)

    return {
        "generator": mask_generator,
        "predictor": predictor,
        "device": device,
        "checkpoint_path": str(model_path),
    }


# ------------------------------------------------
# Generate masks
# ------------------------------------------------
def generate_sam_masks(mask_generator, image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)

    return masks


def _extract_generator(sam_bundle):
    if isinstance(sam_bundle, dict):
        return sam_bundle.get("generator")
    return sam_bundle


def _extract_predictor(sam_bundle):
    if isinstance(sam_bundle, dict):
        return sam_bundle.get("predictor")
    return None


def _is_cuda_bundle(sam_bundle) -> bool:
    if isinstance(sam_bundle, dict):
        return sam_bundle.get("device") == "cuda"
    return torch.cuda.is_available()


def _predict_leaf_mask_fast(image: np.ndarray, sam_bundle) -> Optional[np.ndarray]:
    """Fast SAM inference using point prompts (CPU-friendly)."""
    predictor = _extract_predictor(sam_bundle)
    if predictor is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    points = np.array(
        [
            [w // 2, h // 2],
            [6, 6],
            [w - 7, 6],
            [6, h - 7],
            [w - 7, h - 7],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        return None

    best_idx = int(np.argmax(scores))
    return (masks[best_idx].astype(np.uint8) * 255)


def _cleanup_leaf_mask(mask: np.ndarray) -> np.ndarray:
    """Clean SAM mask and keep the dominant connected leaf component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if n_labels <= 1:
        return cleaned

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dominant = np.where(labels == largest, 255, 0).astype(np.uint8)

    dominant = cv2.morphologyEx(dominant, cv2.MORPH_CLOSE, kernel, iterations=1)
    return dominant


# ------------------------------------------------
# Select best leaf mask
# ------------------------------------------------
def select_leaf_mask(masks, image_shape):

    H, W = image_shape[:2]
    center = np.array([H/2, W/2])

    best_score = -1.0
    best_mask: Optional[np.ndarray] = None
    image_area = float(H * W)

    for m in masks:

        area = float(m["area"])
        mask = m["segmentation"].astype(np.uint8)

        area_ratio = area / image_area
        if area_ratio < 0.03 or area_ratio > 0.98:
            continue

        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)

        center_dist = np.linalg.norm(centroid - center)

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = float(cv2.contourArea(cnt)) / hull_area

        # Prioritize plausible leaf-like masks: sizeable, near-centre, reasonably solid.
        score = (
            (1.5 * area_ratio)
            + (0.4 * solidity)
            - (0.25 * (center_dist / max(H, W)))
        )

        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None

    return _cleanup_leaf_mask(best_mask.astype(np.uint8) * 255)


# ------------------------------------------------
# Full SAM segmentation
# ------------------------------------------------
def segment_leaf_sam(image, mask_generator, max_side: int = 384):

    h, w = image.shape[:2]
    run_img = image
    scale = 1.0
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        run_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    leaf_mask = None

    # 1) Fast point-prompt SAM path (works well on CPU)
    try:
        leaf_mask = _predict_leaf_mask_fast(run_img, mask_generator)
    except Exception:
        leaf_mask = None

    # 2) Automatic mask generation fallback (prefer on CUDA due runtime)
    if leaf_mask is None:
        try:
            if _is_cuda_bundle(mask_generator):
                generator = _extract_generator(mask_generator)
                if generator is not None:
                    masks = generate_sam_masks(generator, run_img)
                    leaf_mask = select_leaf_mask(masks, run_img.shape)
        except Exception:
            leaf_mask = None

    if leaf_mask is None:
        return None

    if scale != 1.0:
        leaf_mask = cv2.resize(leaf_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        leaf_mask = _cleanup_leaf_mask(leaf_mask)

    return leaf_mask
