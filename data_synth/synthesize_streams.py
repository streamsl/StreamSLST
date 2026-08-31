'''Synthesize streaming sign-language datasets from offline pre-segmented pose data.

Single entry-point for all four corpora. Two source layouts are supported, auto-selected from
`SYNTH_META`:

  (A) Legacy big-pickle  -- one {.train,.dev,.test} pickle per split. Each is a dict
    `{name: {'keypoint': (T,133,3), 'text': str, 'num_frames': int}}`. Used by PHOENIX.
    Triggered when `SYNTH_META['pickle_prefix']` is present.

  (B) Per-sample pose `.pkl` (normalized [0,1] x,y + confidence + `w_h`, with optional `start`/`end` 
    for CSL) + a gzipped label dict keyed by sample name. Triggered when `pickle_prefix` is absent.

Both paths feed the **same BOBSL-style synthesis**: same-signer (CSL: `P\\d+`; H2S: VIDEO_ID
prefix) concatenation, BOBSL-derived pause sampling, Hermite C¹ bridges.

Co-articulated streams via five minimal-knob mechanisms:
  1. **trim_rest** strips leading/trailing HELD-rest frames (below-shoulder AND near-still, capped per side).
  2. **Empirical pause sampling** from BOBSL manual annotations (majority zero, heavy positive tail;
     the exact zero fraction is whatever analyze_bobsl_gaps.py measures -- report it from the stats file,
     do not quote a hard-coded number).
  3. **BG_pre / BG_post** sampled from positives only (silent broadcast lead-in / lead-out).
  4. **K-per-stream from a 60s sliding window** (= 4x training window) so streaming actually fires.
  5. **Donor-texture bridges**: every synthesized frame = Hermite C1 MEAN trajectory + real high-frequency
     (xy-jitter, confidence) texture transplanted from the adjacent clips of the same signer/recording.

Mechanism 5 exists because a reviewer probe proved the submitted bridges leaked: confidence was the frozen
elementwise min of the endpoints (a one-line rule found seams at P=1.00/R=0.97) and the noise-free polynomial
coordinates let a kinematics-only GBM segment at ~90 avg F1. Donor transplant carries real marginals, temporal
autocorrelation and conf<->jitter coupling with zero fitted constants. Run data_synth/verify_seam_probe.py after
every synthesis to quantify residual seam detectability -- its output is part of the benchmark card.

A short Hermite bridge (>= MIN_BRIDGE_FRAMES) connects every seam smoothly even when the
sampled pause is 0; tangent magnitude is damped inversely with bridge length so long BG
segments do not extrapolate clip-end momentum into visible overshoot.

Outputs (drop-in BOBSL-style):
    POSE_ROOT/<stream_id>.npy               # (T, 133, 3) float32 at FPS in pixel coords (canvas = SYNTH_META src_w/h)
    VTT_DIR/<stream_id>.vtt                 # WEBVTT, one sentence per cue (single-line text)
    SUBSET_JSON                             # {"train": [...], "val": [...], "test": [...]}
    manifest.json                           # provenance: signer/clip ids per stream + seed

Run:
    DATASET=PHOENIX python -m data_synth.synthesize_streams --out_root data/synth/phoenix
    DATASET=CSL     python -m data_synth.synthesize_streams --out_root data/synth/csl
    DATASET=H2S     python -m data_synth.synthesize_streams --out_root data/synth/h2s

Shorter / easier streams (3..5 sentences each; also yields more total streams):
    DATASET=CSL     python -m data_synth.synthesize_streams --out_root data/synth/csl --k_range 3 5

Carve a val split from train (H2S has no Uni-Sign dev set; CSL/PHOENIX already have one but you
can override with this flag for a custom split):
    DATASET=H2S     python -m data_synth.synthesize_streams --out_root data/synth/h2s --val_frac 0.05

Caveats from the Uni-Sign / How2Sign upstreams (handled here):
  - ZechengLi19/Uni-Sign#2  : per-split counts differ from the Uni-Sign paper -- we trust the
                              gzipped labels file as source-of-truth and print the actual counts.
  - ZechengLi19/Uni-Sign#34 : Uni-Sign has no H2S dev split (loader raises NotImplementedError);
                              use --val_frac to carve val from train if you need one.
  - how2sign/how2sign-data#4: ~117-sample drift across CVPR21 / CSV / clip folders. We trust the
                              labels file but skip entries whose pose .pkl is absent on disk
                              (user-downloaded subsets are common) and log the count.
'''
import argparse, gzip, json, pickle, random, re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET, SYNTH_META, WINDOW_DURATION_SECONDS, FPS, MAX_SUB_DURATION

# Minimum signer-pure pool size to be included in synthesis
#   (otherwise the signer is dropped entirely since we can't form a multi-clip stream from them). 
# This is a tuning knob to trade off diversity (# signers) vs co-articulation realism (multi-clip streams per signer). 
# Setting it to 2 is a minimal requirement for co-articulation, 
#   and already excludes some very small pools with only one clip. 
# Setting it higher would exclude more signers and reduce diversity, 
#   but might improve co-articulation if those signers had many single-clip pools. 
# Setting it to 1 would include all signers but allow some single-clip streams with no co-articulation.
MIN_POOL_SIZE = 2  
MIN_BRIDGE_FRAMES = 2  # Numerical floor for Hermite interp to define a transition (not a tuning knob)

# Trim-rest geometry: COCO-WholeBody-133 indices for shoulders and wrists. A frame is "rest" when
# both wrists sit BELOW (image y-down: larger y) the shoulder line. Pure geometric rule, scales
# automatically across canvases by being relative to each signer's own shoulders.
SHOULDER_IDS = [5, 6]
WRIST_IDS = [9, 10]

# --- Loading & signer grouping ---------------------------------------------

def signer_id(dataset: str, clip_name: str) -> str:
    if dataset == 'PHOENIX': return re.sub(r'-\d+$', '', clip_name.split('/', 1)[-1])
    if dataset == 'CSL':
        m = re.search(r'_P(\d+)_', clip_name)
        return f'P{m.group(1)}' if m else clip_name
    if dataset == 'H2S':
        # How2Sign sentence names follow `<YouTubeID>_<sentence>-<take>-rgb_<view>.mp4`. Sentences from the same source 
        # YouTube ID share a signer + recording setup, so the YouTube ID is the natural "signer" pool unit. Stripping 
        # `_<digits>-...` isolates it whether or not the .mp4 suffix is present.
        base = clip_name.rsplit('/', 1)[-1]
        m = re.match(r'^(.+?)_\d+-\d+-rgb_(?:front|side)(?:\.mp4)?$', base)
        return m.group(1) if m else base.rsplit('.', 1)[0]
    raise ValueError(dataset)


def group_by_signer(clips: Dict[str, dict], dataset: str) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = defaultdict(list)
    for name in clips.keys():
        g[signer_id(dataset, name)].append(name)
    return g


# --- Pose handling: native pixel space throughout --------------------------

def to_numpy_kpts(rec: dict) -> np.ndarray:
    kp = rec['keypoint']
    if hasattr(kp, 'cpu'): kp = kp.cpu().numpy()
    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] != 133 or kp.shape[2] != 3:
        raise ValueError(f'Unexpected keypoint shape {kp.shape}')
    kp[..., 2] = np.clip(kp[..., 2], 0.0, 1.0)
    return kp


def trim_rest(kpts: np.ndarray, rest_speed_sw_per_s: float = 0.1875, max_trim_s: float = 0.75) -> np.ndarray:
    '''Drop contiguous leading/trailing HELD-REST frames at clip boundaries.

    Real continuous signing exhibits **co-articulation**: signers do not return to a full rest pose between adjacent
    sentences. Each pose clip is recorded in isolation, so it may begin with a "preparation" phase (hands rising from lap
    to first sign) and end with a "retraction" phase (hands falling back to lap). Concatenating non-trimmed clips produces
    an obvious "hands-down -> pause -> hands-up" boundary the localization head can latch onto trivially. Trimming makes
    boundaries kinematically continuous, so they must be inferred from sign content, not a free rest cue.

    BUG FIX (reported): the previous rule flagged a frame as "rest" on POSITION ALONE (both wrists below the shoulder
    line). But signers — CSL-Daily for example — produce most of their signing BELOW shoulder level (measured: ~87-97% of
    frames are below-shoulder, moving at full signing speed), so that rule deleted active signing: on the CSL test split it
    trimmed EVERY clip, up to ~9 s of a ~10 s signing segment, while the caption stayed the full sentence — corrupting the
    translation supervision at the source. A rest pose is a HELD low pose, so we now additionally require the wrists to be
    near-STILL: motion below `rest_speed_sw_per_s` (shoulder-widths per SECOND — a physical rate, converted per frame
    inside so the rule is frame-rate independent). Moving low hands = signing, kept. A per-side `max_trim_s` cap
    guarantees a mis-classified slow passage can never delete more than that many seconds. The rule is still
    signer-relative (shoulder-width normalized) and needs no per-dataset threshold; `rest_speed_sw_per_s` / `max_trim_s`
    are physical calibration knobs, not fit parameters. (Datasets that ship a signing segment — CSL start/end — are already
    tight to signing, so this now trims only ~0.05 s there instead of destroying the clip.)
    '''
    T = kpts.shape[0]
    if T < 3: return kpts
    sh = kpts[:, SHOULDER_IDS, :2]
    wr = kpts[:, WRIST_IDS, :2].mean(axis=1)                         # (T, 2) mean wrist position
    sh_y = sh[..., 1].mean(axis=1)
    wr_y = kpts[:, WRIST_IDS, 1].mean(axis=1)                        # image y-down: larger y == lower == below shoulders
    sh_w = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1)               # shoulder width per frame = signer scale
    sh_w = max(float(np.median(sh_w[sh_w > 0])) if np.any(sh_w > 0) else 1.0, 1e-3)
    valid = (kpts[:, SHOULDER_IDS, 2].min(axis=1) > 0) & (kpts[:, WRIST_IDS, 2].min(axis=1) > 0)
    speed = np.zeros(T, dtype=np.float32)
    speed[1:] = np.linalg.norm(np.diff(wr, axis=0), axis=1) / sh_w   # wrist speed, normalized by signer scale
    if T > 1: speed[0] = speed[1]                                    # frame 0 has no predecessor -> use frame 1's motion
    # Rest = hands below the shoulder line AND essentially still (a held pose). Moving low hands are signing.
    # The stillness threshold is a PHYSICAL rate (shoulder-widths per SECOND), converted to a per-frame value here:
    # a per-frame constant would make "still" twice as permissive at 30 fps as at 15 fps (per-frame displacement
    # halves when the frame rate doubles), silently classifying slow signing as rest on native-fps builds.
    # 0.1875 sw/s is the original 0.015 sw/frame calibration expressed at the 12.5 fps it was calibrated on.
    is_rest = (wr_y > sh_y) & (speed < rest_speed_sw_per_s / FPS) & valid
    start = 0
    while start < T and is_rest[start]: start += 1
    end = T
    while end > start and is_rest[end - 1]: end -= 1
    cap = int(max_trim_s * FPS)                                      # never delete more than max_trim_s per side
    if start >= end: return kpts[:0]  # every frame is held rest: a caption over motionless frames is not a sentence -> drop
    start = min(start, cap)
    end = max(end, T - cap)
    # Near-all-rest clips (fewer than 3 non-rest frames) return the short slice rather than the old "give back the clip
    # untrimmed" safeguard: the caller's >=1s duration filter then discards them, instead of keeping a captioned rest
    # clip whose hands-down frames would reintroduce the trivial boundary cue at its seams.
    return kpts[start:end]


def resample_to_fps(kpts: np.ndarray, src_fps: float, tgt_fps: float = FPS) -> np.ndarray:
    '''Vectorized linear resample on (x, y), nearest-neighbour on confidence.

    Replaces a 133-joint Python loop (~266 per-clip np.interp calls) with a single set of
    NumPy index/broadcast ops. Result is mathematically identical to the prior per-joint
    implementation (linear interp on x/y with side='left' nearest-neighbour for confidence).
    '''
    if abs(src_fps - tgt_fps) < 1e-6 or kpts.shape[0] < 2: return kpts
    T_src = kpts.shape[0]
    duration = T_src / src_fps
    T_tgt = max(1, int(round(duration * tgt_fps)))
    src_t = np.arange(T_src, dtype=np.float64) / src_fps
    tgt_t = np.clip(np.arange(T_tgt, dtype=np.float64) / tgt_fps, src_t[0], src_t[-1])

    # Linear interp on x, y: locate bracketing source indices and per-target weight.
    lo = np.clip(np.searchsorted(src_t, tgt_t, side='right') - 1, 0, T_src - 2)
    hi = lo + 1
    span = src_t[hi] - src_t[lo]
    span[span == 0] = 1.0
    frac = ((tgt_t - src_t[lo]) / span).astype(np.float32)[:, None, None]
    out = np.empty((T_tgt, 133, 3), dtype=np.float32)
    out[..., :2] = (1.0 - frac) * kpts[lo][..., :2] + frac * kpts[hi][..., :2]
    # Confidence: TRUE nearest-neighbour (paper A.2 step 1). searchsorted side='left' returns the CEILING
    # (later) source frame for any target strictly between two frames, biasing confidence to the later frame;
    # rounding the fractional source index picks the genuinely nearest frame (source frames are uniform).
    near = np.clip(np.round(tgt_t * src_fps).astype(np.int64), 0, T_src - 1)
    out[..., 2] = kpts[near, :, 2]
    return out


# --- Stream synthesis ------------------------------------------------------

def hermite_interp_segment(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, n: int) -> np.ndarray:
    '''Cubic Hermite spline interpolation per joint on (x, y): the MEAN trajectory of a bridge only.

    Endpoints `p0`, `p1` are real frames from the data (last frame of clip A, first frame of clip B). Tangents `v0`, `v1`
    are the per-frame displacements at those endpoints (clip_A[-1] - clip_A[-2] and clip_B[1] - clip_B[0]) so interpolation
    matches the signer's actual velocity at each side of the seam. This gives C1 continuity at boundaries: the hand motion
    at the end of the spoken sentence flows smoothly into the pause and out into the next sentence.

    Tangents are scaled to the spline parameter s in [0, 1] (so segment length-aware) and clamped to at most 2 * |p1 - p0|
    per joint to prevent overshoot when endpoint velocities are large.

    The returned confidence channel is 0: a spline has no generative model for estimator confidence. Callers must go
    through `make_bridge`, which transplants real (xy-jitter, confidence) texture from the adjacent clips -- a raw spline
    is a noise-free constant-jerk polynomial with degenerate confidence, both of which a shallow per-frame probe can
    detect (reported by a reviewer: a frozen-confidence rule recovers seams at P=1.00/R=0.97).

    Zero free parameters: everything is derived from the data.
    '''
    if n <= 0: return np.zeros((0, 133, 3), dtype=np.float32)
    span = (n + 1.0)  # parameter step is 1/span between frames

    # Damp endpoint-velocity contribution for LONG bridges. Short bridges (n <= MIN_BRIDGE_FRAMES) keep full clip-end momentum 
    # -- this IS the co-articulation continuity we want when the sampled pause was 0. For longer bridges (e.g. multi-second BG 
    # segments) the full per-frame velocity propagated across many frames extrapolates the clip's last motion far beyond where
    # the hand should physically end up, producing visible overshoot. Damping inversely with n makes the spline smoothly converge 
    # to a position-only interpolation as the bridge grows.
    v_scale = float(MIN_BRIDGE_FRAMES) / max(n, MIN_BRIDGE_FRAMES)
    m0 = (v0[:, :2] * span * v_scale).astype(np.float32)
    m1 = (v1[:, :2] * span * v_scale).astype(np.float32)

    # Clamp tangent magnitude to 2 * |p1 - p0| per joint to prevent overshoot
    delta = (p1[:, :2] - p0[:, :2]).astype(np.float32)
    max_mag = 2.0 * np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-3
    for m in (m0, m1):
        mag = np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, max_mag / mag)
        m *= scale  # in-place clamp

    out = np.zeros((n, 133, 3), dtype=np.float32)
    for t in range(n):
        s = (t + 1.0) / span  # s in (0, 1)
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out[t, :, :2] = h00 * p0[:, :2] + h10 * m0 + h01 * p1[:, :2] + h11 * m1
    return out


def _hf_residual(xy: np.ndarray) -> np.ndarray:
    '''High-frequency residual of a real pose track: xy minus its 3-frame moving average.

    Separates the pose estimator's frame-to-frame jitter (the component a cubic spline has none of) from the smooth
    articulated motion. Width 3 is the smallest odd window, i.e. the weakest possible smoother: whatever a 3-frame mean
    already explains is treated as motion, the remainder as sensor texture. Not a tuning knob.
    '''
    if xy.shape[0] < 3: return np.zeros_like(xy)
    pad = np.pad(xy, ((1, 1), (0, 0), (0, 0)), mode='edge')
    return xy - (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0


class TextureReservoir:
    '''Real low-level texture source for a stream's bridges: hf-residual + confidence tracks from donor clips.

    Donors are same-signer clips that are NOT part of the stream (the phantom pool). Every transplanted statistic is
    therefore matched by construction -- marginal distributions, temporal autocorrelation, per-keypoint heterogeneity,
    heavy tails from detection failures, and the cross-channel coupling between confidence dips and coordinate jumps
    all come from real frames of the same source footage rather than from a parametric noise model. (A Gaussian
    noise-match was tried first and rejected: a gradient-boosted probe still recovered seams from the residual
    distributional mismatch; see data_synth/verify_seam_probe.py.)

    Two value-reuse signatures are designed out:
      - donors outside the stream: copying texture from a clip that also appears in the stream would let a
        duplicate-row detector (conf vector seen at two time indices of one stream) recover every seam;
      - non-overlapping draws: two bridges sharing reservoir frames would duplicate each other the same way.
    When the signer pool has no spare clip both protections degrade (in-stream donors, possible overlap); this is
    recorded per stream in the manifest (`texture_donors_in_stream`) so the fraction is disclosed, not hidden.

    Each bridge takes one contiguous slice. Among 8 candidate starts the one whose first confidence row is closest
    (mean |diff|) to the anchoring real frame is chosen -- an argmin over measured values, not a fitted constant --
    to keep the bridge-entry confidence step at the scale of an ordinary frame-to-frame step.
    '''

    def __init__(self, donor_clips: List[np.ndarray], donor_provider, rng: np.random.Generator,
                 exclude_rows: set = None):
        # `donor_provider` yields further out-of-stream resampled clips on demand; the reservoir tops itself up
        # rather than EVER reusing frames -- two bridges sharing reservoir rows duplicate each other exactly the
        # way in-stream donors did (measured: overlap/tiling left 40% of bridge frames row-duplicated).
        # `exclude_rows` is the set of confidence-row hashes already present in the stream's kept clips: a donor
        # sharing any of them is REJECTED. This matters on How2Sign, whose sentence clips are cut from one
        # continuous video and can overlap in source frames -- a same-video donor then carries frames that are
        # byte-identical to in-stream clip frames, recreating the duplicate-row seam signature through the back
        # door (measured: 4.6% of bridge frames row-duplicated at distant indices before this rejection).
        self.rng = rng
        self.provider = donor_provider
        self.res = np.zeros((0, 133, 2), np.float32)
        self.conf = np.zeros((0, 133), np.float32)
        self.used: List[Tuple[int, int]] = []
        self.reused_frames = 0  # >0 only when the whole split ran out of donor material (disclosed via manifest)
        self.skipped_overlap_donors = 0
        self.seen = set(exclude_rows) if exclude_rows else set()
        for d in donor_clips: self._append(d)

    def _append(self, clip: np.ndarray) -> None:
        if clip.shape[0] < 3: return
        if self.seen:
            rows = [clip[t, :, 2].tobytes() for t in range(clip.shape[0])]
            if any(r in self.seen for r in rows):
                self.skipped_overlap_donors += 1
                return
            self.seen.update(rows)
        self.res = np.concatenate([self.res, _hf_residual(clip[..., :2])])
        self.conf = np.concatenate([self.conf, clip[..., 2]])

    def draw(self, n: int, anchor_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            L = self.conf.shape[0]
            if L >= n:
                cands = self.rng.integers(0, L - n + 1, size=8)
                free = [int(c) for c in cands if all(c + n <= a or c >= b for a, b in self.used)]
                if free:
                    steps = [float(np.abs(self.conf[c] - anchor_conf).mean()) for c in free]
                    c = free[int(np.argmin(steps))]
                    self.used.append((c, c + n))
                    return self.res[c:c + n].copy(), self.conf[c:c + n].copy()
            nxt = next(self.provider, None)
            if nxt is None: break
            self._append(nxt)
        # Donor material exhausted (tiny split): accept overlapped reuse, but count it so it is never silent.
        if L == 0: return np.zeros((n, 133, 2), np.float32), np.zeros((n, 133), np.float32)
        self.reused_frames += n
        c = int(self.rng.integers(0, max(1, L - n + 1)))
        reps = int(np.ceil((c + n) / L))
        res = np.tile(self.res, (reps, 1, 1))[c:c + n].copy()
        conf = np.tile(self.conf, (reps, 1))[c:c + n].copy()
        return res, conf


def make_bridge(left: np.ndarray, right: np.ndarray, n: int, reservoir: 'TextureReservoir') -> np.ndarray:
    '''One non-transcribed bridge segment between two real clips: Hermite mean trajectory + donor texture.

    Fixes the two artifact channels a reviewer's shallow probe exploited in the submitted benchmarks:
      (1) confidence was the frozen elementwise min of the endpoints -> a "conf unchanged since previous frame" rule
          found bridge frames at P=1.00/R=0.97; now the bridge carries a real confidence track from a donor clip.
      (2) coordinates were a noise-free polynomial -> jerk statistics separated bridges from real DWPose/RTMPose
          output; now the donors' measured high-frequency residual rides on top of the spline.
    '''
    seg = hermite_interp_segment(left[-1], right[0],
                                 endpoint_velocity(left, 'last'), endpoint_velocity(right, 'first'), n)
    res, conf = reservoir.draw(n, left[-1][:, 2])
    seg[..., :2] += res
    seg[..., 2] = conf
    return seg


def endpoint_velocity(clip: np.ndarray, end: str) -> np.ndarray:
    '''Per-frame (x, y, conf) velocity at a clip endpoint. `end` in {'last', 'first'}.

    Returns a (133, 3) array with v[:, :2] = displacement, v[:, 2] = 0 (unused).
    '''
    if clip.shape[0] < 2: return np.zeros((133, 3), dtype=np.float32)
    if end == 'last': v = (clip[-1] - clip[-2]).astype(np.float32)
    elif end == 'first': v = (clip[1] - clip[0]).astype(np.float32)
    else: raise ValueError(end)
    v[:, 2] = 0.0
    return v


def synth_one_stream(
    rng: np.random.Generator, texture_rng: np.random.Generator, signer_pool: List[str],
    clips: Dict[str, dict], src_fps: float, pause: dict, k_range: Tuple[int, int],
) -> Tuple[np.ndarray, List[Tuple[float, float, str]], dict]:
    # `rng` drives stream COMPOSITION (clip choice, pause lengths); `texture_rng` drives only the donor-texture draws
    # inside bridges. Keeping them separate means regenerating with/without a texture change reproduces byte-identical
    # composition (same clips, same cue times), so any downstream metric delta is attributable to the texture alone.
    k_lo, k_hi = k_range
    K_target = int(rng.integers(min(k_lo, len(signer_pool)), min(k_hi, len(signer_pool)) + 1))

    # Top-up loop: many CSL clips are very short (<1s after [start:end] slicing + trim_rest), so a plain rng.choice(size=K) 
    # frequently lands on clips that get dropped by the <1s filter, producing streams with fewer than k_lo sentences. We 
    # instead shuffle the whole signer pool once and walk it in random order, keeping every clip that passes the filters 
    # until we hit K_target (or exhaust the pool). This guarantees `len(keep_names) >= min(K_target, n_valid)`
    # while preserving determinism (rng seeded per stream).
    pool_order = list(np.asarray(signer_pool)[rng.permutation(len(signer_pool))])
    resampled, texts, keep_names = [], [], []
    attempted: set = set()
    # Duration window a cue must satisfy to ever be supervised downstream:
    #  - lower bound: ceil(1s * FPS), NOT int(1s * FPS). int(1.0 * 12.5) = 12 frames = 0.96 s -- such a clip passes the
    #    old filter but is silently dropped by the loader's MIN_SUB_DURATION=1.0 s check, so its signing trains as
    #    background while its text vanishes: corrupted supervision. ceil makes generator and loader agree.
    #  - upper bound: a cue longer than the training/eval window can NEVER be fully contained in any window, so its text
    #    is never supervised and its frames are permanent false background. Cap at min(window, MAX_SUB_DURATION).
    min_frames_after_resample = int(np.ceil(1.0 * FPS))
    max_frames_after_resample = int(min(WINDOW_DURATION_SECONDS, MAX_SUB_DURATION) * FPS)
    for name in pool_order:
        if len(resampled) >= K_target: break
        if name in attempted: continue
        attempted.add(name)
        rec = clips[name]
        text = str(rec.get('text', '')).strip()
        if not text: continue
        kp = trim_rest(resample_to_fps(to_numpy_kpts(rec), src_fps))
        if not (min_frames_after_resample <= kp.shape[0] <= max_frames_after_resample): continue
        resampled.append(kp)
        texts.append(text)
        keep_names.append(name)

    if not resampled: return np.zeros((0, 133, 3), dtype=np.float32), [], {'clips': [], 'K': 0}

    # Phantom clips for BG_pre and BG_post: random clips from this signer's pool that were NOT selected as kept content. 
    # We deliberately do NOT trim_rest the phantoms -- BG segments represent the broadcast lead-in / lead-out where signer 
    # naturally holds a rest pose, so retaining the rest frames at phantom endpoint is realistic. If the pool has no spare 
    # clips, fallback to 1st/last selected clip itself (the seam will still be Hermite-interpolated rather than a held frame).
    chosen_set = set(keep_names)
    spare = [n for n in signer_pool if n not in chosen_set]
    if spare:
        phantom_left_name = str(rng.choice(spare))
        phantom_right_name = str(rng.choice(spare))
        phantom_left = resample_to_fps(to_numpy_kpts(clips[phantom_left_name]), src_fps)
        phantom_right = resample_to_fps(to_numpy_kpts(clips[phantom_right_name]), src_fps)
    else:
        phantom_left_name = phantom_right_name = '<self>'
        phantom_left = resampled[-1] if len(resampled) > 1 else resampled[0]
        phantom_right = resampled[0] if len(resampled) > 1 else resampled[-1]

    # Texture donor pool: the phantoms (already loaded, guaranteed outside the stream when spares exist) plus up to two
    # more spare clips drawn via texture_rng (texture choices never consume the composition rng). When the signer pool
    # has NO spare (common on PHOENIX, whose per-broadcast pools hold only 2-5 clips), donors are drawn from the whole
    # split instead -- same corpus, same pose extractor, same canvas, just a different signer. That keeps every donor
    # OUT of the stream: falling back to the stream's own clips (the first implementation) left verbatim confidence
    # rows duplicated inside the stream, and a one-line duplicate-row detector recovered every seam from them
    # (measured P=0.65 / R=1.00). Only if the split itself has no out-of-stream clip do we accept in-stream donors,
    # flagged per stream in the manifest.
    donor_names = [nm for nm in clips.keys() if nm not in chosen_set and nm not in (phantom_left_name, phantom_right_name)]
    donor_order = texture_rng.permutation(len(donor_names))
    def _donor_iter():
        for j in donor_order:
            yield resample_to_fps(to_numpy_kpts(clips[donor_names[int(j)]]), src_fps)
    donors_in_stream = not (spare or donor_names)
    initial = [phantom_left, phantom_right] if spare else (list(resampled) if donors_in_stream else [])
    # Row hashes of the stream's own clips: donors sharing any of them are rejected (overlapping H2S sentences).
    # The degenerate in-stream fallback necessarily shares rows, so exclusion is disabled there (flag discloses it).
    stream_rows = None if donors_in_stream else {c[t, :, 2].tobytes() for c in resampled for t in range(c.shape[0])}
    reservoir = TextureReservoir(initial, _donor_iter(), texture_rng, exclude_rows=stream_rows)

    segments: List[np.ndarray] = []
    durations: dict = {'pre_s': 0.0, 'pauses_s': [], 'post_s': 0.0, 'coart_joins': 0, 'pause_joins': 0}
    cur = 0

    # BG_pre = Hermite interp from phantom_left[-1] -> resampled[0][0] (signer "transitioning into" the first sentence). 
    # Animated, biomechanically grounded in real signer poses. Sampled from the POSITIVE-only subset of BOBSL gaps so BG 
    # segments are always visibly present in stream -- representing the silent broadcast lead-in, not an inter-sentence join.
    L_pre_s = float(rng.choice(pause['bg_samples_s']))
    n_pre = max(MIN_BRIDGE_FRAMES, int(round(L_pre_s * FPS)))
    bg_pre = make_bridge(phantom_left, resampled[0], n_pre, reservoir)
    segments.append(bg_pre)
    durations['pre_s'] = L_pre_s
    cur += bg_pre.shape[0]

    # Build physical stream (clip | bridge | clip | ...), recording each clip's physical frame span and, per seam, whether the sampled 
    # BOBSL gap was ZERO (co-articulated) or POSITIVE (a real pause). We still insert a >=2-frame Hermite bridge at EVERY  seam -- it is 
    # a numerical necessity for a C1-continuous transition, because clip[-1] and next_clip[0] are kinematically  independent and a hard 
    # concat would teleport (an even more trivial boundary cue than a gap). But the ORACLE cue boundaries are computed in a second pass 
    # below: co-articulated seams are *absorbed* into 2 adjacent sentences so their boundaries are frame-adjacent.
    clip_spans: List[Tuple[int, int]] = []          # physical [start, end) frame range of each real clip
    joins: List[dict] = []                          # per seam i (between clip i and i+1): {'n': bridge_frames, 'coart': bool}
    for i, clip in enumerate(resampled):
        c_start = cur
        segments.append(clip)
        cur += clip.shape[0]
        clip_spans.append((c_start, cur))
        if i < len(resampled) - 1:
            L_pause_s = float(rng.choice(pause['samples_s']))
            n_content = int(round(L_pause_s * FPS))         # frames the sampled gap actually spans (0 for the majority of BOBSL gaps)
            is_coart = (n_content == 0)                     # zero-gap -> co-articulated: sentences must be frame-adjacent
            # Positive gaps are rendered at EXACTLY their sampled length (a 1-frame gap stays 1 frame -- the Hermite
            # evaluates fine at a single interior sample), so the oracle gap distribution matches the BOBSL empirical
            # distribution it is sampled from. Only co-articulated (zero-gap) seams get the MIN_BRIDGE_FRAMES epenthesis
            # floor, and those frames are absorbed into the adjacent cues below, leaving no oracle gap at all.
            n_bridge = n_content if n_content >= 1 else MIN_BRIDGE_FRAMES
            pause_seg = make_bridge(clip, resampled[i + 1], n_bridge, reservoir)
            segments.append(pause_seg)
            cur += pause_seg.shape[0]
            joins.append({'n': n_bridge, 'coart': is_coart})
            durations['pauses_s'].append(L_pause_s)
            durations['coart_joins' if is_coart else 'pause_joins'] += 1

    # Oracle cues: Emitting cue_i = [clip_i.start, clip_i.end] will leave EVERY =2-frame bridge as a non-transcribed gap, so adjacent 
    # sentences were never frame-adjacent, even for the ~74% of BOBSL joins whose gap is exactly 0. That contradicts real continuous 
    # signing, where a sign sentence can start immediately after another (co-articulation / movement epenthesis), and it hands the 
    # localization head a trivial "there's always a small non-signing sliver between sentences" cue that doesn't exist in real streams.
    # Correct behaviour: a co-articulated (zero-gap) seam is signing content, not background. Its Hermite epenthesis frames are split 
    # at the midpoint and absorbed into 2 adjacent sentences so cue_i.end == cue_{i+1}.start (frame-adjacent). Only a genuine pause 
    # (sampled gap > 0) stays a non-transcribed background gap between sentences. This makes the synthetic oracle match the BOBSL gap 
    # statistics we sample from (analyze_bobsl_gaps.py) and the co-articulation goal stated in paper App. A.2, and it forces the model 
    # to infer boundaries from sign content.
    cues: List[Tuple[float, float, str]] = []
    for i, ((c_start, c_end), text) in enumerate(zip(clip_spans, texts)):
        left = c_start
        if i > 0 and joins[i - 1]['coart']:                        # take ceil(n/2) from the (co-articulated) preceding bridge
            n_prev = joins[i - 1]['n']
            left = c_start - (n_prev - n_prev // 2)
        right = c_end
        if i < len(clip_spans) - 1 and joins[i]['coart']:          # take floor(n/2) from the (co-articulated) following bridge
            right = c_end + joins[i]['n'] // 2
        cues.append((left / FPS, right / FPS, text))

    # BG_post = Hermite interp from resampled[-1][-1] -> phantom_right[0] 
    # (broadcast lead-out; same positive-only sampling as BG_pre).
    L_post_s = float(rng.choice(pause['bg_samples_s']))
    n_post = max(MIN_BRIDGE_FRAMES, int(round(L_post_s * FPS)))
    bg_post = make_bridge(resampled[-1], phantom_right, n_post, reservoir)
    segments.append(bg_post)
    cur += bg_post.shape[0]
    durations['post_s'] = L_post_s

    poses = np.concatenate(segments, axis=0).astype(np.float32)
    # NO canvas clamp. The old clip-to-[0, src_w/h] step (a) silently altered real PHOENIX frames -- DWPose legitimately
    # predicts keypoints outside the canvas, exactly as raw BOBSL npy files do on the same downstream path -- and (b)
    # pinned clamped bridge frames to the exact border value, an isolated-border-point contrast a probe can exploit.
    # Hermite overshoot beyond the canvas (~2% worst case) is harmless downstream: group-relative normalization has no
    # hard range requirement, and low-confidence outliers are zeroed by threshold_confidence anyway.
    prov = {
        'clips': keep_names, 'K': len(cues),
        'phantom_left': phantom_left_name, 'phantom_right': phantom_right_name,
        'texture_donors_in_stream': donors_in_stream, 'texture_reused_frames': int(reservoir.reused_frames),
        'texture_donors_skipped_overlap': int(reservoir.skipped_overlap_donors),
        'duration_s': float(poses.shape[0] / FPS), **durations,
    }
    return poses, cues, prov


# --- Output writers --------------------------------------------------------

def fmt_vtt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def write_vtt(path: Path, cues: List[Tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['WEBVTT', '']
    for s, e, text in cues:
        lines.append(f'{fmt_vtt_time(s)} --> {fmt_vtt_time(e)}')
        lines.append(' '.join(text.split()))
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_pose(out_dir: Path, stream_id: str, poses: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{stream_id}.npy', poses.astype(np.float32))


# --- Per-split orchestration -----------------------------------------------

def load_pause_dist(stats_path: str, allow_fallback: bool = False) -> dict:
    '''Load BOBSL empirical inter-subtitle gap samples (replaces the old LogNormal parameters).

    Looks for `bobsl_gap_samples.npy` next to the stats JSON. If absent, HARD-FAILS unless
    `allow_fallback` is set: a benchmark silently built on the analytic emulation would not be the
    benchmark described in the paper (empirical BOBSL gaps), and the difference would be invisible
    in the outputs. The emulation exists only for smoke tests (--allow_fallback_pauses).

    Returned dict: {'samples_s': np.ndarray, 'frac_zero': float, 'median_s': float, 'p90_s': float}
    '''
    samples_path = Path(stats_path).with_name('bobsl_gap_samples.npy')
    if samples_path.exists():
        samples = np.load(samples_path).astype(np.float32)
        samples = np.clip(samples, 0.0, None)  # negative gaps (overlapping subs) = co-articulated, treat as 0
        src = f'BOBSL empirical n={samples.size}'
    elif not allow_fallback:
        raise FileNotFoundError(
            f'{samples_path} not found. Run data_synth/analyze_bobsl_gaps.py to produce the empirical gap '
            f'samples, or pass --allow_fallback_pauses to build a NON-benchmark smoke-test stream set from '
            f'an analytic emulation of the gap distribution.')
    else:
        rng = np.random.default_rng(0)
        n = 10000
        is_pos = rng.random(n) > 0.74
        samples = np.where(is_pos, rng.lognormal(np.log(2.0), 0.83), 0.0).astype(np.float32)
        src = f'fallback (no {samples_path.name}; emulating 74% zero + LogNormal positive tail) -- SMOKE TEST ONLY'

    # Positive-only subset: used for BG_pre/BG_post (broadcast lead-in / lead-out). Inter-clip pauses use full empirical (which is 
    # ~74% zeros, capturing co-articulation), but BG segments are conceptually different -- they represent silent broadcast preamble 
    # where there is no caption, not an inter-sentence join. Sampling them from full empirical collapses ~74% of streams to a 2-frame 
    # BG (invisible). Sampling from positives only keeps them broadcast-realistic (median ~2s, heavy tail) without adding new knobs.
    bg_samples = samples[samples > 0]
    if bg_samples.size == 0: bg_samples = np.array([2.0], dtype=np.float32)  # degenerate fallback
    info = {
        'samples_s': samples,
        'bg_samples_s': bg_samples,
        'frac_zero': float((samples == 0).mean()),
        'median_s': float(np.median(samples)),
        'p90_s': float(np.percentile(samples, 90)),
        'bg_median_s': float(np.median(bg_samples)),
        'bg_p90_s': float(np.percentile(bg_samples, 90)),
    }
    print(f"(pause samples: {src}; frac_zero={info['frac_zero']:.3f}, "
          f"median={info['median_s']:.2f}s, p90={info['p90_s']:.2f}s; "
          f"BG (positive only): median={info['bg_median_s']:.2f}s, p90={info['bg_p90_s']:.2f}s)")
    return info


def load_k_range(stats_path: str) -> Tuple[int, int]:
    '''Sentences-per-stream range derived from BOBSL "subs per 15s sliding window" stats.

    If `bobsl_gap_stats.json` does not contain `subs_per_window_p10` / `subs_per_window_p90`
    (older versions), fall back to deriving K from `sub_dur_median_s` + `gap_median_s`:
        K_typical = WINDOW_DURATION_SECONDS / (sub_dur_median + max(gap_median_pos, 0))
        K_lo = max(1, K_typical // 2);  K_hi = K_typical * 2
    Otherwise use the empirical p10..p90 range from the BOBSL distribution.
    '''
    p = Path(stats_path)
    if p.exists():
        s = json.loads(p.read_text())
        # Prefer subs_per_STREAM_window (60s) — anchors K to a stream length that spans multiple
        # 15s training windows so the streaming inference behaviour actually fires at eval time.
        if 'subs_per_stream_window_p10' in s and 'subs_per_stream_window_p90' in s:
            lo = max(2, int(s['subs_per_stream_window_p10']))
            hi = max(lo + 1, int(s['subs_per_stream_window_p90']))
            sw = float(s.get('stream_window_seconds', 60.0))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_{sw:.0f}s_window p10..p90)')
            return lo, hi
        if 'subs_per_window_p10' in s and 'subs_per_window_p90' in s:
            lo = max(1, int(s['subs_per_window_p10']))
            hi = max(lo + 1, int(s['subs_per_window_p90']))
            print(f'(BOBSL K range: [{lo}, {hi}] from subs_per_15s_window p10..p90 [no 60s stats found])')
            return lo, hi
        sub_med = float(s.get('sub_dur_median_s', 4.0))
        # gap_median_s is often 0 in BOBSL because most subs are adjacent; use exp(log_pos_mean) as positive median
        gap_med_pos = float(np.exp(s.get('log_pos_mean', np.log(2.0))))
        cycle = max(2.0, sub_med + gap_med_pos)
        k_typ = max(2, int(round(WINDOW_DURATION_SECONDS / cycle)))
        lo, hi = max(1, k_typ // 2), max(k_typ + 1, k_typ * 2)
        print(f'(K range derived from sub_dur_median={sub_med:.1f}s + gap_median_pos={gap_med_pos:.1f}s '
              f'-> K_typical={k_typ}, range [{lo}, {hi}])')
        return lo, hi
    print('(no BOBSL stats; K range fallback [2, 10])')
    return 2, 10


def _load_per_sample_split(split_name_on_disk: str) -> Dict[str, dict]:
    '''Load a per-sample split (CSL or H2S) into the same `{name: {keypoint, text, ...}}` schema as the legacy big-pickle path:

        - parse gzipped labels.<split> (sample dict)
        - read poses/<name>.pkl (RTMPose normalized x,y in [0,1] + per-frame score + w_h)
        - CSL only: slice keypoints[start:end]
        - merge (T,1,133,2) xy with (T,1,133) confidence -> (T,133,3) on a unified canvas (multiplied by SYNTH_META src_w/src_h 
          so cross-clip endpoints are scale-comparable for Hermite bridging; group-relative normalization in poses/preprocessing.py 
          is scale- invariant downstream, so the choice of canvas only affects bridge endpoint deltas).
        - clip confidence to [0,1] (RTMPose occasionally returns ~1.1)
        - skip samples with empty text, missing pose .pkl, or fewer than 1s of frames after slicing.
    '''
    pickle_dir = Path(SYNTH_META['pickle_dir'])
    pose_dir = pickle_dir / SYNTH_META.get('pose_dir_name', 'poses')
    label_path = pickle_dir / f"{SYNTH_META.get('label_prefix', 'labels')}.{split_name_on_disk}"
    with gzip.open(label_path, 'rb') as f: labels = pickle.load(f)

    canvas = np.asarray([SYNTH_META['src_w'], SYNTH_META['src_h']], dtype=np.float32)
    min_src_frames = int(1.0 * SYNTH_META['src_fps'])
    out: Dict[str, dict] = {}
    n_missing_pose = n_empty_text = n_short = 0

    for sample_name, entry in tqdm(labels.items(), desc=f'[{DATASET}/{split_name_on_disk}] load Uni-Sign'):
        text = str(entry.get('text', '')).strip()
        if not text: n_empty_text += 1; continue
        # H2S keys carry .mp4; CSL keys do not. The pose file is always <name without .mp4>.pkl.
        pose_stem = sample_name[:-4] if sample_name.endswith('.mp4') else sample_name
        pose_path = pose_dir / f'{pose_stem}.pkl'
        if not pose_path.exists(): n_missing_pose += 1; continue
        try:
            with open(pose_path, 'rb') as f: pose_pkl = pickle.load(f)
        except Exception as e:
            print(f'  WARN: failed to read {pose_path.name}: {e}'); n_missing_pose += 1; continue

        kpts = np.asarray(pose_pkl['keypoints'])  # (T, 1, 133, 2) normalized [0, 1] xy
        scores = np.asarray(pose_pkl['scores'])   # (T, 1, 133) confidence (can exceed 1.0)
        if kpts.ndim == 4 and kpts.shape[1] == 1: kpts = kpts[:, 0]      # (T, 133, 2)
        if scores.ndim == 3 and scores.shape[1] == 1: scores = scores[:, 0]  # (T, 133)
        if 'start' in pose_pkl and 'end' in pose_pkl:
            s, e = int(pose_pkl['start']), int(pose_pkl['end'])
            if 0 <= s < e <= kpts.shape[0]:
                kpts = kpts[s:e]; scores = scores[s:e]
        if kpts.shape[0] < min_src_frames: n_short += 1; continue

        # Scale normalized [0,1] to a unified pixel-equivalent canvas. Per-sample w_h varies in H2S; rescaling each sample 
        # to (src_w, src_h) makes endpoint deltas (clip[-1] vs phantom[0]) comparable across signers without recovering each 
        # video's native resolution. We clip normalized coords to [0,1] before scaling: RTMPose occasionally predicts keypoints 
        # outside the frame (face/hands leaving view, or extrapolated low-confidence detections), which would otherwise produce 
        # pose values > canvas size. Clipping keeps output strictly within [0, src_w] x [0, src_h] -- the few clipped points are 
        # typically low-confidence anyway and get zeroed by threshold_confidence() downstream.
        kpts_norm = np.clip(kpts.astype(np.float32), 0.0, 1.0)
        xy = kpts_norm * canvas[None, None, :]
        conf = np.clip(scores.astype(np.float32), 0.0, 1.0)[..., None]
        keypoint = np.concatenate([xy, conf], axis=-1).astype(np.float32)  # (T, 133, 3)

        out[sample_name] = {
            'name': sample_name, 'text': text, 'keypoint': keypoint,
            'num_frames': int(keypoint.shape[0]), 'gloss': entry.get('gloss', ''),
        }
    print(f'  loaded {len(out)} samples (skipped: {n_missing_pose} missing pose, '
          f'{n_empty_text} empty text, {n_short} <1s after slicing)')
    return out


def _load_legacy_pickle_split(split_name_on_disk: str) -> Dict[str, dict]:
    pickle_path = SYNTH_META['pickle_dir'] / f"{SYNTH_META['pickle_prefix']}.{split_name_on_disk}"
    print(f'[{DATASET}/{split_name_on_disk}] loading legacy pickle {pickle_path}')
    with open(pickle_path, 'rb') as f: return pickle.load(f)


def synthesize_split(
    split: str, out_pose_dir: Path, out_vtt_dir: Path,
    base_seed: int, pause: dict, k_range: Tuple[int, int],
) -> Tuple[List[str], List[dict]]:
    # Map our split name to the on-disk split name (e.g. CSL/PHOENIX -> 'dev', H2S has no 'dev' key).
    split_map = SYNTH_META.get('splits', {'train': 'train', 'val': 'dev', 'test': 'test'})
    if split not in split_map:
        # Caller asked for a split this dataset's source does not provide (e.g. H2S 'val' when
        # --val_frac is 0). Synthesize an empty split so the orchestrator's output stays uniform.
        print(f'[{DATASET}/{split}] no source split mapping -- skipping')
        return [], []

    load_split = split_map[split]
    if 'pickle_prefix' in SYNTH_META: clips = _load_legacy_pickle_split(load_split)
    else: clips = _load_per_sample_split(load_split)

    min_src_frames = int(1.0 * SYNTH_META['src_fps'])
    clips = {k: v for k, v in clips.items() if str(v.get('text', '')).strip() and v.get('num_frames', 0) >= min_src_frames}
    groups = group_by_signer(clips, DATASET)
    groups = {sid: lst for sid, lst in groups.items() if len(lst) >= MIN_POOL_SIZE}
    n_usable = sum(len(v) for v in groups.values())
    k_avg = (k_range[0] + k_range[1]) / 2.0
    # Each clip used ~once on average across the split (n_usable / K_avg). Preserves the offline split's contract: no clip 
    # duplication beyond what's statistically inevitable. We deliberately do NOT inflate this with a permutation-multiplier knob 
    # -- on small signer pools (PHOENIX broadcasts have only 2-3 clips each) any multiplier above 1 saturates the K! permutation
    # space and biases training toward duplicated streams.
    n_streams = max(1, int(round(n_usable / k_avg)))
    print(f'[{DATASET}/{split}] usable clips: {n_usable}, signer-pure groups: {len(groups)} -> {n_streams} streams')
    if not groups: raise RuntimeError(f'No signer groups for {DATASET}/{split}')

    rng = np.random.default_rng(base_seed)
    signer_ids = list(groups.keys())
    stream_ids: List[str] = []
    manifest: List[dict] = []
    
    for i in tqdm(range(n_streams), desc=f'synth {split}'):
        stream_rng = np.random.default_rng([base_seed, i])
        # Independent per-stream generator for bridge donor-texture draws. Seeded from (base_seed, i) plus a fixed tag,
        # NOT from stream_rng's sequence, so (a) each stream is reproducible in isolation and (b) texture changes never
        # perturb composition draws (clips/pauses/cue times stay byte-identical across texture revisions).
        texture_rng = np.random.default_rng([base_seed, i, 0x7E])
        sid = signer_ids[int(rng.integers(0, len(signer_ids)))]
        poses, cues, prov = synth_one_stream(stream_rng, texture_rng, groups[sid], clips, SYNTH_META['src_fps'], pause, k_range)
        if not cues: continue
        stream_id = f'{split}_{i:05d}'
        write_pose(out_pose_dir, stream_id, poses)
        write_vtt(out_vtt_dir / f'{stream_id}.vtt', cues)
        prov.update({'stream_id': stream_id, 'signer': sid, 'split': split})
        manifest.append(prov)
        stream_ids.append(stream_id)
    return stream_ids, manifest


def _carve_val_from_train(subset: Dict[str, List[str]], full_manifest: Dict[str, List[dict]], val_frac: float, seed: int) -> None:
    '''Move a deterministic random fraction of train STREAMS into val, in-place.

    Mutates subset and full_manifest. Useful when the source corpus has no dev split (H2S) or when a custom split ratio is desired. 
    We carve at the STREAM level (not the source-clip level) so each carved stream is internally consistent and the per-window 
    evaluation logic needs no special-case handling.
    '''
    if val_frac <= 0.0: return
    if not (0.0 < val_frac < 1.0): raise ValueError(f'--val_frac must be in (0, 1); got {val_frac}')
    train_ids = list(subset.get('train', []))
    n_move = int(round(len(train_ids) * val_frac))
    if n_move <= 0:
        print(f'(--val_frac={val_frac:g} on {len(train_ids)} train streams rounds to 0 -- no carve)')
        return

    rng = random.Random(seed)
    move_set = set(rng.sample(train_ids, n_move))
    subset['train'] = [sid for sid in train_ids if sid not in move_set]
    subset['val']   = sorted(list(subset.get('val', [])) + [sid for sid in train_ids if sid in move_set])

    # Manifest: rebuild train+val entry lists keyed by stream_id (preserves provenance).
    train_manifest = full_manifest.get('train', [])
    keep, moved = [], []
    for m in train_manifest:
        (moved if m.get('stream_id') in move_set else keep).append(m)

    full_manifest['train'] = keep
    full_manifest['val'] = list(full_manifest.get('val', [])) + moved
    print(f'(carved {n_move}/{len(train_ids)} train streams into val [seed={seed}, frac={val_frac:g}])')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', required=True)
    p.add_argument('--bobsl_gap_stats', default='data_synth/stats/bobsl_gap_stats.json',
                   help='JSON of BOBSL manual-aligned pause statistics; if missing, use fallback.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k_range', type=int, nargs=2, metavar=('LO', 'HI'), default=None,
                   help='Override sentences-per-stream range [LO, HI]. e.g. --k_range 3 5 for shorter, '
                        'easier streams (also yields more total streams since n_streams = n_usable/K_avg).')
    p.add_argument('--val_frac', type=float, default=0.0,
                   help='Fraction of TRAIN streams to move into val after synthesis (default 0 = no carve). '
                        'Useful for H2S where the Uni-Sign labels file has no dev split. Deterministic '
                        'given --seed. Applied AFTER per-split synthesis so each carved stream is intact.')
    p.add_argument('--allow_fallback_pauses', action='store_true',
                   help='Permit the analytic pause-distribution emulation when bobsl_gap_samples.npy is missing. '
                        'Smoke tests only -- never for a released benchmark.')
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_pose_dir = out_root / 'poses'
    out_vtt_dir = out_root / 'vtt'
    out_pose_dir.mkdir(parents=True, exist_ok=True)
    out_vtt_dir.mkdir(parents=True, exist_ok=True)

    pause = load_pause_dist(args.bobsl_gap_stats, allow_fallback=args.allow_fallback_pauses)
    if args.k_range is not None:
        lo, hi = int(args.k_range[0]), int(args.k_range[1])
        if lo < 1 or hi < lo: raise ValueError(f'invalid --k_range {args.k_range}: need 1 <= LO <= HI')
        k_range = (lo, hi)
        print(f'(K range overridden by --k_range: [{lo}, {hi}])')
    else: k_range = load_k_range(args.bobsl_gap_stats)

    splits_cfg = [('train', args.seed), ('val', args.seed + 1), ('test', args.seed + 2)]
    subset: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    full_manifest: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}
    for split, seed in splits_cfg:
        ids, m = synthesize_split(split, out_pose_dir, out_vtt_dir, seed, pause, k_range)
        subset[split] = ids
        full_manifest[split] = m

    # Optional train -> val carve (default 0% = no carve). Done after synthesis so each carved
    # stream is fully written to disk and only the subset2episode.json mapping is rerouted.
    _carve_val_from_train(subset, full_manifest, args.val_frac, args.seed)
    (out_root / 'subset2episode.json').write_text(json.dumps(subset, indent=2))

    # Code / environment identity: "same seed => same benchmark" only holds for the same generator code, numpy version
    # (Generator bit-streams are version-dependent) and pause-sample file. Stamp all three so two archives that both say
    # `seed: 42` are distinguishable, and a benchmark built from modified code is self-documenting.
    import hashlib, subprocess
    def _md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else 'missing'
    try:
        git_rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                 cwd=Path(__file__).parent).stdout.strip() or 'unknown'
        git_dirty = bool(subprocess.run(['git', 'status', '--porcelain', str(Path(__file__).resolve())],
                                        capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip())
    except Exception:
        git_rev, git_dirty = 'unknown', True
    provenance = {
        'generator_md5': _md5(Path(__file__).resolve()),
        'git_rev': git_rev, 'git_dirty': git_dirty,
        'numpy_version': np.__version__,
        'pause_samples_md5': _md5(Path(args.bobsl_gap_stats).with_name('bobsl_gap_samples.npy')),
        'pause_source': 'empirical' if not args.allow_fallback_pauses else 'empirical_or_fallback',
    }
    (out_root / 'manifest.json').write_text(json.dumps({
        'dataset': DATASET,
        'provenance': provenance,
        'src_meta': {k: (str(v) if isinstance(v, Path) else v) for k, v in SYNTH_META.items()},
        'target_fps': FPS,
        # Drop the raw samples arrays from manifest -- store only the summary statistics that
        # describe the empirical pause distribution actually used during synthesis.
        'pause': {k: v for k, v in pause.items() if k not in ('samples_s', 'bg_samples_s')},
        'k_range': list(k_range), 'min_pool_size': MIN_POOL_SIZE, 'window_seconds': WINDOW_DURATION_SECONDS,
        'min_bridge_frames': MIN_BRIDGE_FRAMES, 'trim_rest': True, 'bridge_texture': 'donor_hf_v2',
        'val_frac': args.val_frac, 'seed': args.seed, 'streams': full_manifest,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nDone. Wrote {sum(len(v) for v in subset.values())} streams to {out_root}")
