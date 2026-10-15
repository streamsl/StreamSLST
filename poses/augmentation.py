# Pose augmentation approach from best solution of Kaggle's GISLR competition (--pose_augment, train only, off by default)
# https://www.kaggle.com/code/hoyso48/1st-place-solution-training?scriptVersionId=128283887&cellId=9
# Only the augmentations that keep the window length (and so the subtitle timing) are used: augment_dvc_window.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def flip_lr(keypoints: np.ndarray) -> np.ndarray:
    '''Flip horizontally and swap left/right for COCO-WholeBody 133 keypoints.

    Notes:
    - Mirrors x around the image centerline (x -> width - x).
    - Swaps the left/right hand groups to preserve semantic indices.
    - Swaps a small set of body left/right pairs (eyes/shoulders/elbows/wrists).
    '''
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    keypoints = np.asarray(keypoints).copy()
    conf = keypoints[..., 2]
    valid = conf > 0
    keypoints[..., 0] = np.where(valid, WIDTH - keypoints[..., 0], keypoints[..., 0]) # Horizontal flip in pixel coordinates

    # Swap hands as groups (left 91-111, right 112-132)
    left_hand = keypoints[:, LEFT_HAND_IDS, :].copy()
    right_hand = keypoints[:, RIGHT_HAND_IDS, :].copy()
    keypoints[:, LEFT_HAND_IDS, :] = right_hand
    keypoints[:, RIGHT_HAND_IDS, :] = left_hand

    # Swap common body left/right semantic pairs.
    body_lr_pairs = [
        (1, 2),  # left/right eye
        (5, 6),  # left/right shoulder
        (7, 8),  # left/right elbow
        (9, 10),  # left/right wrist
    ]
    for li, ri in body_lr_pairs:
        tmp = keypoints[:, li, :].copy()
        keypoints[:, li, :] = keypoints[:, ri, :]
        keypoints[:, ri, :] = tmp

    # FACE_IDS = [23..39 jaw contour] + [53 nose]. Mirror ONLY the jaw contour among itself (23<->39,
    # 24<->38, ... chin-center fixed); the nose is on the centerline and its x was already mirrored above.
    # Reversing the full FACE_IDS slice wrongly swaps the nose into the jaw contour.
    face_contour = FACE_IDS[:-1]
    keypoints[:, face_contour] = keypoints[:, face_contour[::-1]].copy()
    # Inner mouth (ids 83..90 = 68-landmark inner lip 60..67) is a LOOP, not a left->right strip, so a
    # plain reversal pairs a corner with a bottom-lip point. Correct mirror: 60<->64, 61<->63, 65<->67;
    # 62 (top-center) and 66 (bottom-center) stay fixed.
    mouth_mirror = [MOUTH_IDS[i] for i in (4, 3, 2, 1, 0, 7, 6, 5)]
    keypoints[:, MOUTH_IDS] = keypoints[:, mouth_mirror].copy()
    return keypoints


def spatial_random_affine(
	keypoints: np.ndarray,
	scale: tuple[float, float] | None = (0.8, 1.2),
	shear: tuple[float, float] | None = (-0.15, 0.15),
	shift: tuple[float, float] | None = (-0.1, 0.1),
	degree: tuple[float, float] | None = (-30, 30),
	rng: np.random.Generator | None = None,
) -> np.ndarray:
    '''Apply random affine to x/y (leaves conf intact).

    Designed for raw pixel coordinates (0..WIDTH, 0..HEIGHT).
    Only transforms keypoints with conf > 0 to avoid moving padded/invalid points.
    '''
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3: raise ValueError(f'Expected (T, K, 3), got {keypoints.shape}')
    if rng is None: rng = np.random.default_rng()
    keypoints = np.asarray(keypoints).copy()
    xy = keypoints[..., :2]
    conf = keypoints[..., 2]
    valid = conf > 0
    center = np.array([WIDTH / 2.0, HEIGHT / 2.0], dtype=np.float32)

    if scale is not None: # Scale around the canvas center
        s = float(rng.uniform(scale[0], scale[1]))
        xy[valid] = (xy[valid] - center) * s + center

    if shear is not None: # Shear around the canvas center
        shear_x = shear_y = float(rng.uniform(shear[0], shear[1]))
        if rng.random() < 0.5: shear_x = 0.0
        else: shear_y = 0.0
        shear_mat = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float32)
        xy[valid] = (xy[valid] - center) @ shear_mat + center
        center = center + np.array([shear_y, shear_x], dtype=np.float32)

    if degree is not None: # Rotate around center
        deg = float(rng.uniform(degree[0], degree[1]))
        rad = deg * np.pi / 180.0 
        cos_v, sin_v = float(np.cos(rad)), float(np.sin(rad))
        rot = np.array([[cos_v, sin_v], [-sin_v, cos_v]], dtype=np.float32)
        xy[valid] = (xy[valid] - center) @ rot + center

    if shift is not None: # Shift in pixels (fraction of WIDTH/HEIGHT)
        sh = float(rng.uniform(shift[0], shift[1]))
        xy[valid, 0] = xy[valid, 0] + sh * WIDTH
        xy[valid, 1] = xy[valid, 1] + sh * HEIGHT

    keypoints[..., :2] = xy
    return keypoints


# The masks write 0 (x, y and confidence), the loader's 'missing keypoint': input version 2 drops a group whose root or length
# keypoints are masked.
def temporal_mask(
	keypoints: np.ndarray,
	size: tuple[float, float] = (0.2, 0.4),
	mask_value: float = 0.0,
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	keypoints = np.asarray(keypoints).copy()
	l = int(keypoints.shape[0])
	if l <= 1: return keypoints

	frac = float(rng.uniform(size[0], size[1]))
	mask_size = int(max(1, round(l * frac)))
	max_offset = max(1, l - mask_size)
	offset = int(rng.integers(0, max_offset))
	keypoints[offset : offset + mask_size, :, :] = mask_value
	return keypoints


def spatial_mask(
	keypoints: np.ndarray,
	size: tuple[float, float] = (0.2, 0.4),
	mask_value: float = 0.0,
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	if rng is None: rng = np.random.default_rng()
	keypoints = np.asarray(keypoints).copy()

	offset_y = float(rng.uniform(0.0, HEIGHT))
	offset_x = float(rng.uniform(0.0, WIDTH))
	frac = float(rng.uniform(size[0], size[1]))
	box = frac * min(WIDTH, HEIGHT)

	mask_x = (offset_x < keypoints[..., 0]) & (keypoints[..., 0] < offset_x + box)
	mask_y = (offset_y < keypoints[..., 1]) & (keypoints[..., 1] < offset_y + box)
	mask = mask_x & mask_y
	keypoints[mask, :] = mask_value
	return keypoints


def augment_dvc_window(keypoints: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
	'''Augmentations safe for window-level DVC (do not change sequence length).

	Applies: flip_lr, spatial_random_affine, temporal_mask, spatial_mask.
	'''
	if rng is None: rng = np.random.default_rng()
	if rng.random() < 0.5: keypoints = flip_lr(keypoints)
	if rng.random() < 0.75: keypoints = spatial_random_affine(keypoints, rng=rng)
	if rng.random() < 0.5: keypoints = temporal_mask(keypoints, rng=rng)
	if rng.random() < 0.5: keypoints = spatial_mask(keypoints, rng=rng)
	return keypoints