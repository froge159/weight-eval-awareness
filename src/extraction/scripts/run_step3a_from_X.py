import json, sys, numpy as np
from pathlib import Path
UP=Path('/sessions/relaxed-sleepy-shannon/mnt/uploads')
LAYERS=sorted(int(p.name.split('_L')[1].split('_')[0]) for p in UP.glob('X_L*_extract.npz'))

def fam(r): return f"{r['regime']}|{r['template_family_eval']}"
def eff(p): 
    s=p.std(ddof=1); return float(p.mean()/s) if s>0 else 0.0
def unit(v): return v/np.linalg.norm(v)

def null_eff_p95(X, n=400, seed=0):
    rng=np.random.default_rng(seed); ds=[]
    for _ in range(n):
        v=unit(rng.standard_normal(X.shape[1])); ds.append(abs(eff(X@v)))
    return float(np.percentile(ds,95))

def boot_eff_ci(X, r, qids, n_boot=1000, seed=0):
    p=X@r; clusters=sorted(set(qids))
    idx={c:[] for c in clusters}
    for i,q in enumerate(qids): idx[q].append(i)
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        rr=[]
        for k in rng.choice(len(clusters),len(clusters),replace=True): rr.extend(idx[clusters[k]])
        vals.append(eff(p[np.asarray(rr)]))
    return float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

res={}
print(f"{'L':>4} | {'in-samp d':>9} {'null p95':>8} {'ratio':>6} | {'LFO d':>7} {'LFO ratio':>9} {'inv':>4} | "
      f"{'X-reg d':>8} {'X-reg ratio':>11} | {'cos(e,w)':>9}")
print("-"*104)
for L in LAYERS:
    with np.load(UP/f'X_L{L}_last_prompt_token_extract.npz') as f: X=f['X'].astype(np.float64)
    rows=[json.loads(l) for l in open(UP/f'X_L{L}_last_prompt_token_extract.rows.jsonl')]
    with np.load(UP/f'r_hat_L{L}_last_prompt_token.npz') as f:
        r=f['r_hat'].astype(np.float64); re_=f['r_hat_explicit'].astype(np.float64); rw=f['r_hat_woodlabs'].astype(np.float64)
    qids=[x['question_id'] for x in rows]; fams=[fam(x) for x in rows]; regs=[x['regime'] for x in rows]

    nullp = null_eff_p95(X)
    d_in  = eff(X@r); ci_in = boot_eff_ci(X,r,qids)

    # leave-family-out
    lfo=[]; inverted=0
    for held in sorted(set(fams)):
        te=np.array([f==held for f in fams]); tr=~te
        rt=unit(X[tr].mean(0)); d=eff(X[te]@rt); lfo.append(d)
        if d<0: inverted+=1
    lfo_mean=float(np.mean(lfo)); lfo_min=float(np.min(lfo))

    # cross-regime
    xr=[]
    for train in sorted(set(regs)):
        trm=np.array([g==train for g in regs]); tem=~trm
        rt=unit(X[trm].mean(0)); xr.append(eff(X[tem]@rt))
    xr_min=float(np.min(xr))

    res[L]=dict(layer=L, d_in_sample=d_in, d_in_ci=list(ci_in), null_p95=nullp,
                ratio_in=d_in/nullp, lfo_mean=lfo_mean, lfo_min=lfo_min,
                lfo_ratio=lfo_mean/nullp, lfo_inverted=inverted,
                xreg_min=xr_min, xreg_ratio=xr_min/nullp,
                cos_explicit_woodlabs=float(re_@rw))
    print(f"{L:>4} | {d_in:>9.2f} {nullp:>8.2f} {d_in/nullp:>5.1f}x | "
          f"{lfo_mean:>7.2f} {lfo_mean/nullp:>8.1f}x {inverted:>4} | "
          f"{xr_min:>8.2f} {xr_min/nullp:>10.1f}x | {float(re_@rw):>+9.3f}")

json.dump(res, open('/tmp/step3a/final.json','w'), indent=2)
print()
good=[L for L in LAYERS if res[L]['xreg_ratio']>=1.5 and res[L]['lfo_inverted']==0 and res[L]['cos_explicit_woodlabs']>=0.5]
print("Layers passing ALL of: cross-regime >=1.5x null, no inverted folds, cos(e,w) >= 0.5")
print(" ", good)
