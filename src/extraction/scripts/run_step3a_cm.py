#!/usr/bin/env python3
"""Step 3a on the completion_mean direction.

Reworked from the hand-run run_step3a_from_X.py: takes the contrastive-matrix
dir and the direction dir as arguments instead of a hardcoded upload path, and
reads r_hat_L{L}_{pooling}.npz instead of the hardcoded last_prompt_token name.

    PYTHONPATH=src python scripts/run_step3a_cm.py \\
        --contrastive outputs_cm/contrastive \\
        --directions outputs/directions_cm \\
        --pooling completion_mean
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

def fam(r): return f"{r['regime']}|{r['template_family_eval']}"
def eff(p):
    s = p.std(ddof=1); return float(p.mean()/s) if s > 0 else 0.0
def unit(v): return v/np.linalg.norm(v)

def null_eff_p95(X, n=400, seed=0):
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        v = unit(rng.standard_normal(X.shape[1])); ds.append(abs(eff(X@v)))
    return float(np.percentile(ds, 95))

def boot_eff_ci(X, r, qids, n_boot=1000, seed=0):
    p = X@r; clusters = sorted(set(qids))
    idx = {c: [] for c in clusters}
    for i, q in enumerate(qids): idx[q].append(i)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        rr = []
        for k in rng.choice(len(clusters), len(clusters), replace=True):
            rr.extend(idx[clusters[k]])
        vals.append(eff(p[np.asarray(rr)]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrastive", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--pooling", default="completion_mean")
    ap.add_argument("--split", default="extract")
    a = ap.parse_args()

    CM = Path(a.contrastive); DIR = Path(a.directions)
    layers = sorted(int(p.name.split('_L')[1].split('_')[0])
                    for p in CM.glob(f'X_L*_{a.pooling}_{a.split}.npz'))
    if not layers:
        print(f"no X_L*_{a.pooling}_{a.split}.npz in {CM}"); return 1

    print(f"{'L':>4} | {'in-samp d':>9} {'null p95':>8} {'ratio':>6} | "
          f"{'LFO d':>7} {'LFO ratio':>9} {'inv':>4} | "
          f"{'X-reg d':>8} {'X-reg ratio':>11} | {'cos(e,w)':>9}")
    print("-"*92)

    res = {}
    for L in layers:
        Xf = CM/f'X_L{L}_{a.pooling}_{a.split}.npz'
        rows_f = CM/f'X_L{L}_{a.pooling}_{a.split}.rows.jsonl'
        with np.load(Xf) as f:
            X = f['X'].astype(np.float64)
        rows = [json.loads(l) for l in rows_f.read_text().splitlines() if l.strip()]
        qids = [r['question_id'] for r in rows]
        regimes = np.array([r['regime'] for r in rows])
        fams = np.array([fam(r) for r in rows])

        dpath = DIR/f'r_hat_L{L}_{a.pooling}.npz'
        with np.load(dpath) as f:
            r = f['r_hat'].astype(np.float64)
            re_ = f['r_hat_explicit'].astype(np.float64) if 'r_hat_explicit' in f else None
            rw = f['r_hat_woodlabs'].astype(np.float64) if 'r_hat_woodlabs' in f else None

        # in-sample separation vs random-direction null
        d_in = abs(eff(X@r)); p95 = null_eff_p95(X); ratio = d_in/p95 if p95 else 0.0

        # leave-family-out: fit on all-but-one family, test on held-out
        lfo_ds = []
        for held in sorted(set(fams)):
            tr = fams != held; te = fams == held
            if te.sum() < 2 or tr.sum() < 2: continue
            Xtr = X[tr]
            _, _, vt = np.linalg.svd(Xtr, full_matrices=False)
            v = vt[0]
            if v @ Xtr.mean(0) < 0: v = -v
            lfo_ds.append(abs(eff(X[te]@unit(v))))
        lfo_d = float(np.mean(lfo_ds)) if lfo_ds else 0.0
        lfo_ratio = lfo_d/p95 if p95 else 0.0
        lfo_inv = "y" if lfo_ratio >= 1.0 else "n"

        # cross-regime transfer: direction from explicit tested on woodlabs
        xreg_d = 0.0; xreg_ratio = 0.0; cos_ew = float('nan')
        if re_ is not None and rw is not None:
            ex_mask = regimes == "explicit"; wl_mask = regimes == "woodlabs"
            if wl_mask.sum() >= 2:
                xreg_d = abs(eff(X[wl_mask] @ re_))
                p95_wl = null_eff_p95(X[wl_mask])
                xreg_ratio = xreg_d/p95_wl if p95_wl else 0.0
            cos_ew = abs(float(re_ @ rw))

        print(f"{L:>4} | {d_in:>9.3f} {p95:>8.3f} {ratio:>6.2f} | "
              f"{lfo_d:>7.3f} {lfo_ratio:>9.2f} {lfo_inv:>4} | "
              f"{xreg_d:>8.3f} {xreg_ratio:>11.2f} | {cos_ew:>9.3f}")
        res[str(L)] = dict(d_in=d_in, null_p95=p95, ratio=ratio,
                           lfo_d=lfo_d, lfo_ratio=lfo_ratio,
                           xreg_d=xreg_d, xreg_ratio=xreg_ratio, cos_ew=cos_ew)

    out = DIR/f"step3a_{a.pooling}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")

    # band summary (36-76), the layers the edit targets
    band = [str(l) for l in layers if 36 <= l <= 76]
    if band:
        br = np.mean([res[l]["ratio"] for l in band])
        bl = np.mean([res[l]["lfo_ratio"] for l in band])
        bx = np.mean([res[l]["xreg_ratio"] for l in band])
        bc = np.nanmean([res[l]["cos_ew"] for l in band])
        print(f"\nBAND 36-76 means:  in-samp ratio {br:.2f}  LFO ratio {bl:.2f}  "
              f"X-reg ratio {bx:.2f}  cos(e,w) {bc:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
