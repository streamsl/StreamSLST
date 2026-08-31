'''Seam-detectability probe for synthesized streaming benchmarks (benchmark-card tool).

Implements the shallow-classifier audit a NeurIPS reviewer requested: can a per-frame classifier on
low-level kinematic and confidence features identify the synthetic bridge frames (and therefore the
sentence boundaries) WITHOUT any linguistic signal? If yes, localization scores on the benchmark are
contaminated by a low-level shortcut and must be reported together with this probe's numbers.

Three escalating tests:
  T0  hard-rule audit : fraction of bridge frames whose confidence vector is IDENTICAL to the previous
                        frame (the frozen-conf artifact of the submitted generator), plus the P/R/F1 of
                        the resulting one-line seam detector.
  T1  frame classifier: logistic regression + gradient-boosted trees on per-frame features, trained on
                        the train split, evaluated on the test split. Feature ablations attribute the
                        leak: all / kinematics-only / confidence-only.
  T2  segmenter F1    : T1's frame posteriors converted into sentence proposals (contiguous not-bridge
                        runs >= 1 s) and scored with the SAME per-window-free localization protocol as
                        the paper (macro-averaged P/R per stream, F1 at tIoU {0.3,0.5,0.7,0.9} + avg),
                        so the number is directly comparable to the model's localization F1.

This tool is REPORT-ONLY by design: perfect statistical indistinguishability of synthesized frames is
not achievable (the bridges are non-transcribed by construction), so the pass criterion is a judgment
call made in the paper, not a threshold hidden here. The JSON it writes IS the disclosure.

Run after every synthesis:
    python -m data_synth.verify_seam_probe --root data/synth/phoenix
Writes <root>/seam_probe.json.
'''
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

TIOUS = (0.3, 0.5, 0.7, 0.9)
WRISTS = [9, 10]
HANDS = list(range(91, 133))


def parse_vtt(path: Path):
    subs, lines = [], path.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        if '-->' not in lines[i]:
            i += 1
            continue
        text = lines[i + 1].strip() if i + 1 < len(lines) else ''
        a, b = lines[i].split(' --> ')
        to_s = lambda s: sum(float(x) * 60 ** j for j, x in enumerate(reversed(s.replace(',', '.').split(':'))))
        if text: subs.append((to_s(a), to_s(b), text))
        i += 2
    return sorted(subs)


def compute_iou(p, g):
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = min(max(p[1], g[1]) - min(p[0], g[0]), (p[1] - p[0]) + (g[1] - g[0]))
    return inter / (union + 1e-8)


def pr_at_tiou(preds, gts, tiou):
    if not preds and not gts: return None, None
    if not preds or not gts: return 0.0, 0.0
    pc = sum(1 for p in preds if any(compute_iou(p, g) >= tiou for g in gts))
    gc = sum(1 for g in gts if any(compute_iou(p, g) >= tiou for p in preds))
    return pc / len(preds), gc / len(gts)


def _rolling(x: np.ndarray, w: int):
    '''Centered rolling mean and std of a 1-D series (edge-padded). Windowed features let the probe see
    sustained anomalies (e.g. an unusually still, band-limited multi-frame run) that per-frame features miss.'''
    pad = np.pad(x, (w // 2, w // 2), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    return win.mean(-1), win.std(-1)


def _dup_conf_rows(cf: np.ndarray) -> np.ndarray:
    '''1.0 for frames whose exact confidence vector occurs at ANOTHER time index of the same stream.

    Detects value-reuse artifacts (a bridge whose confidence track was copied verbatim from frames that also
    appear in the stream). Real estimator confidence is continuous, so exact 133-dim repeats essentially never
    happen in genuine footage.'''
    keys = [cf[t].tobytes() for t in range(cf.shape[0])]
    from collections import Counter
    counts = Counter(keys)
    return np.array([1.0 if counts[k] > 1 else 0.0 for k in keys], dtype=np.float32)


def frame_features(kp: np.ndarray):
    '''(T,133,3) raw stream -> (T,F) per-frame features; returns (features, n_kinematic).

    Kinematic block: speed/accel/jerk statistics + rolling (w=5, w=11) speed mean/std.
    Confidence block: level/delta statistics, frozen-row fraction, duplicate-row indicator,
                      + rolling (w=5, w=11) |dconf| mean/std.
    '''
    xy, cf = kp[..., :2], kp[..., 2]
    v = np.zeros_like(xy); v[1:] = xy[1:] - xy[:-1]
    a = np.zeros_like(xy); a[1:] = v[1:] - v[:-1]
    j = np.zeros_like(xy); j[1:] = a[1:] - a[:-1]
    sp, ac, jk = (np.linalg.norm(x, axis=-1) for x in (v, a, j))
    dcf = np.zeros_like(cf); dcf[1:] = cf[1:] - cf[:-1]
    sp_m, adc_m = sp.mean(1), np.abs(dcf).mean(1)
    sp_r5m, sp_r5s = _rolling(sp_m, 5); sp_r11m, sp_r11s = _rolling(sp_m, 11)
    dc_r5m, dc_r5s = _rolling(adc_m, 5); dc_r11m, dc_r11s = _rolling(adc_m, 11)
    kin = np.stack([
        sp[:, WRISTS].mean(1), sp[:, HANDS].mean(1), sp_m, sp.max(1), sp.std(1),
        ac[:, WRISTS].mean(1), ac.mean(1), ac.max(1),
        jk[:, WRISTS].mean(1), jk.mean(1), jk.max(1), jk.std(1),
        sp_r5m, sp_r5s, sp_r11m, sp_r11s,
    ], axis=1)
    conf = np.stack([
        cf.mean(1), cf.std(1), cf.min(1), cf.max(1),
        adc_m, np.abs(dcf).max(1),
        (np.abs(dcf) < 1e-9).mean(1),  # fraction of keypoints with conf identical to the previous frame
        _dup_conf_rows(cf),            # exact conf row seen elsewhere in this stream (value-reuse detector)
        dc_r5m, dc_r5s, dc_r11m, dc_r11s,
    ], axis=1)
    return np.nan_to_num(np.concatenate([kin, conf], axis=1)), kin.shape[1]


def frame_labels(T: int, cues, fps: float):
    '''1 = inside a non-transcribed bridge/background, 0 = inside a transcribed sentence span.'''
    y = np.ones(T, dtype=np.int8)
    for s, e, _ in cues:
        y[max(0, int(round(s * fps))):min(T, int(round(e * fps)))] = 0
    return y


def load_split(root: Path, split: str, fps: float):
    X, Y, G, meta = [], [], [], []
    nkin = 0
    for vp in sorted((root / 'poses').glob(f'{split}_*.npy')):
        cues = parse_vtt(root / 'vtt' / f'{vp.stem}.vtt')
        if not cues: continue
        kp = np.load(vp)
        f, nkin = frame_features(kp)
        y = frame_labels(kp.shape[0], cues, fps)
        X.append(f); Y.append(y); G.append(np.full(len(y), len(meta)))
        meta.append({'id': vp.stem, 'T': kp.shape[0], 'cues': cues})
    if not X: raise FileNotFoundError(f'no {split}_*.npy under {root}/poses')
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G), meta, nkin


def proposals_from_posterior(post: np.ndarray, fps: float, thr: float = 0.5):
    sig = post < thr
    out, i, T, min_len = [], 0, len(post), int(round(1.0 * fps))
    while i < T:
        if sig[i]:
            k = i
            while k < T and sig[k]: k += 1
            if k - i >= min_len: out.append((i / fps, k / fps))
            i = k
        else: i += 1
    return out


def segmenter_f1(post, meta, groups, fps: float):
    per = {t: {'p': [], 'r': []} for t in TIOUS}
    for gi, m in enumerate(meta):
        preds = proposals_from_posterior(post[groups == gi], fps)
        gts = [(s, e) for s, e, _ in m['cues']]
        for t in TIOUS:
            p, r = pr_at_tiou(preds, gts, t)
            if p is not None: per[t]['p'].append(p); per[t]['r'].append(r)
    out, ps, rs = {}, [], []
    for t in TIOUS:
        P = float(np.mean(per[t]['p'])) if per[t]['p'] else 0.0
        R = float(np.mean(per[t]['r'])) if per[t]['r'] else 0.0
        out[f'f1@{t}'] = 100 * (2 * P * R / (P + R) if P + R else 0.0)
        ps.append(P); rs.append(R)
    P, R = float(np.mean(ps)), float(np.mean(rs))
    out['f1_avg'] = 100 * (2 * P * R / (P + R) if P + R else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Synthesized benchmark root (contains poses/, vtt/, manifest.json)')
    ap.add_argument('--train_split', default='train')
    ap.add_argument('--test_split', default='test')
    ap.add_argument('--out', default=None, help='Output JSON (default <root>/seam_probe.json)')
    args = ap.parse_args()
    root = Path(args.root)
    fps = float(json.loads((root / 'manifest.json').read_text())['target_fps'])

    Xtr, Ytr, _, _, nkin = load_split(root, args.train_split, fps)
    Xte, Yte, Gte, Mte, _ = load_split(root, args.test_split, fps)
    report = {
        'root': str(root), 'fps': fps,
        'train_frames': int(len(Ytr)), 'test_frames': int(len(Yte)),
        'test_bridge_frac': float(Yte.mean()),
        'bridge_texture': json.loads((root / 'manifest.json').read_text()).get('bridge_texture', 'none (pre-fix)'),
    }

    # T0: one-line hard rules -- each is a value-reuse artifact detector needing no learning at all.
    def _rule(col_mask, name):
        tp = int((col_mask & (Yte == 1)).sum()); fp = int((col_mask & (Yte == 0)).sum()); fn = int((~col_mask & (Yte == 1)).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        return {'precision': p, 'recall': r, 'f1': 2 * p * r / max(1e-9, p + r)}
    FROZEN_COL, DUP_COL = nkin + 6, nkin + 7
    report['hard_rule_frozen_conf'] = {
        'bridge_frac': float(Xte[Yte == 1, FROZEN_COL].mean()), 'clip_frac': float(Xte[Yte == 0, FROZEN_COL].mean()),
        **_rule(Xte[:, FROZEN_COL] > 0.999, 'frozen'),
    }
    report['hard_rule_duplicate_conf_row'] = {
        'bridge_frac': float(Xte[Yte == 1, DUP_COL].mean()), 'clip_frac': float(Xte[Yte == 0, DUP_COL].mean()),
        **_rule(Xte[:, DUP_COL] > 0.5, 'dup'),
    }

    # T1 + T2 with feature ablations.
    ablations = {'all': slice(0, Xtr.shape[1]), 'kinematic_only': slice(0, nkin), 'confidence_only': slice(nkin, Xtr.shape[1])}
    report['probes'] = {}
    for name, sl in ablations.items():
        report['probes'][name] = {}
        for clf_name, clf in (('logreg', LogisticRegression(max_iter=2000)),
                              ('gbm', HistGradientBoostingClassifier(max_iter=200, random_state=0))):
            a, b = Xtr[:, sl], Xte[:, sl]
            if clf_name == 'logreg':
                mu, sd = a.mean(0), a.std(0) + 1e-8
                a, b = (a - mu) / sd, (b - mu) / sd
            clf.fit(a, Ytr)
            post = clf.predict_proba(b)[:, 1]
            row = {'auroc': float(roc_auc_score(Yte, post)), 'ap': float(average_precision_score(Yte, post))}
            row.update(segmenter_f1(post, Mte, Gte, fps))
            report['probes'][name][clf_name] = row
            print(f"[{name}/{clf_name}] AUROC={row['auroc']:.4f}  segmenter avg F1={row['f1_avg']:.2f}")

    out = Path(args.out) if args.out else root / 'seam_probe.json'
    out.write_text(json.dumps(report, indent=2))
    hf, hd = report['hard_rule_frozen_conf'], report['hard_rule_duplicate_conf_row']
    print(f"\nhard rules: frozen-conf P={hf['precision']:.3f} R={hf['recall']:.3f} | "
          f"dup-row P={hd['precision']:.3f} R={hd['recall']:.3f} (bridge frac {hd['bridge_frac']:.3f} vs clip {hd['clip_frac']:.3f})")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
