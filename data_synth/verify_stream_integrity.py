'''Integrity gate: every synthesized stream cue must contain (nearly) the full source signing segment.

Catches the class of bug where trim_rest / slicing silently deletes signing while the caption stays the
full sentence — i.e. the pose span no longer matches the reference text. For each stream we pair its VTT
cues (in order) with the source clips recorded in manifest.json and compare:

    cue_span_s        = cue.end - cue.start                          (what the model actually sees)
    source_signing_s  = (end - start) / src_fps   [CSL/H2S start,end] (the signing the text refers to)
                      = num_frames    / src_fps   [PHOENIX, no start,end]

A small deficit is expected (legitimate held-rest trim, <~0.75 s/side). A large deficit means signing was
deleted. Exit code is non-zero on FAIL so this can gate a re-synthesis.

Run:  DATASET=CSL python -m data_synth.verify_stream_integrity --root data/synth/csl --split test
'''
import argparse, json, pickle, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYNTH_META, DATASET
from utils import parse_vtt


def _load_pickle_retry(path: Path, attempts: int = 4, delay_s: float = 2.0):
    # Source pose data lives on a cloud-synced volume (OneDrive); a cold, unhydrated file can raise
    # TimeoutError([Errno 60]) on first read. Bounded retries let the hydration finish instead of
    # failing the whole gate on transient I/O.
    for k in range(attempts):
        try:
            with open(path, 'rb') as f: return pickle.load(f)
        except (TimeoutError, OSError) as e:
            if k == attempts - 1: raise
            print(f'  retry {k + 1}/{attempts - 1} for {path.name} after {type(e).__name__}: {e}')
            time.sleep(delay_s * (k + 1))


def source_signing_seconds(name: str) -> float:
    # Duration (s) of the source clip's signing segment, matching what the loader feeds the synthesizer.
    src_fps = float(SYNTH_META['src_fps'])
    if 'pickle_prefix' in SYNTH_META:  # PHOENIX big-pickle: no start/end, full clip is the segment
        return None  # resolved from the pickle cache below
    pose_dir = Path(SYNTH_META['pickle_dir']) / SYNTH_META.get('pose_dir_name', 'poses')
    stem = name[:-4] if name.endswith('.mp4') else name
    p = _load_pickle_retry(pose_dir / f'{stem}.pkl')
    n = np.asarray(p['keypoints']).shape[0]
    if 'start' in p and 'end' in p:
        s, e = int(p['start']), int(p['end'])
        if 0 <= s < e <= n: n = e - s
    return n / src_fps


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='test')
    # Legitimate held-rest trimming is bounded by trim_rest's per-side cap (max_trim_s = 0.75 s), so a cue can
    # honestly lose up to 1.5 s when BOTH margins are true rest (common on H2S, whose source clips keep their
    # rest margins). The gate exists to catch deletion BEYOND that bound — the runaway class where signing is
    # eaten — so the default tolerance is 2 * max_trim_s plus resampling slack, not an arbitrary sub-cap value.
    ap.add_argument(
        '--tol_s', type=float, default=1.6, 
        help='max acceptable per-cue deficit (s); default 2*max_trim_s + slack'
    )
    args = ap.parse_args()
    root = Path(args.root)

    manifest = json.loads((root / 'manifest.json').read_text())
    streams = manifest['streams'][args.split]
    src_fps = float(SYNTH_META['src_fps'])

    # PHOENIX big-pickle: preload the split to look up num_frames per clip
    pheonix_cache = {}
    if 'pickle_prefix' in SYNTH_META:
        split_on_disk = SYNTH_META.get('splits', {}).get(args.split, args.split)
        pk = Path(SYNTH_META['pickle_dir']) / f"{SYNTH_META['pickle_prefix']}.{split_on_disk}"
        pheonix_cache = pickle.load(open(pk, 'rb'))

    n_spans = n_trimmed = n_count_mismatch = frames_lost = 0
    worst = []
    for st in streams:
        sid = st['stream_id']
        clips = st.get('clips', [])
        cues = parse_vtt(root / 'vtt' / f'{sid}.vtt')
        if len(cues) != len(clips):
            # A cue-count mismatch IS the silent-deletion failure this gate exists for (a cue vanished 
            # or was never written) -- it must fail the gate, not downgrade to a warning that exits 0.
            n_count_mismatch += 1
            print(f'  MISMATCH {sid}: {len(cues)} cues vs {len(clips)} manifest clips')
            continue
        for name, cue in zip(clips, cues):
            if pheonix_cache: src_s = pheonix_cache[name]['keypoint'].shape[0] / src_fps
            else: src_s = source_signing_seconds(name)
            span_s = cue['end'] - cue['start']
            deficit = src_s - span_s
            n_spans += 1
            if deficit > args.tol_s:
                n_trimmed += 1
                frames_lost += int(round(deficit * SYNTH_META['src_fps']))
                worst.append({'clip': name, 'stream': sid, 'source_s': round(src_s, 3),
                              'span_s': round(span_s, 3), 'deficit_s': round(deficit, 3)})

    worst.sort(key=lambda w: -w['deficit_s'])
    frac = n_trimmed / max(1, n_spans)
    verdict = 'PASS' if (frac < 0.02 and n_count_mismatch == 0) else 'FAIL'
    out = {'dataset': DATASET, 'split': args.split, 'spans': n_spans, 'over_trimmed_spans': n_trimmed,
           'over_trimmed_fraction': round(frac, 4), 'frames_lost_estimate': frames_lost,
           'cue_count_mismatched_streams': n_count_mismatch,
           'tol_s': args.tol_s, 'verdict': verdict, 'worst': worst[:15]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    sys.exit(0 if verdict == 'PASS' else 1)
