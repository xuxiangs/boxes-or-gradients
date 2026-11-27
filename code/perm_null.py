import numpy as np, json
src=open('hgdp_china_analysis.py').read().split('# ---------- Test 1')[0]; exec(src)
china=groups_of(CHINA); idx_c=np.concatenate(china); sizes=[len(g) for g in china]; cuts=np.cumsum(sizes)[:-1]
rng=np.random.default_rng(7); null=[gst(np.split(rng.permutation(idx_c),cuts)) for _ in range(1000)]
obs=gst(china); json.dump(dict(null=[float(x) for x in null],obs=float(obs)),open('perm_null.json','w')); print(obs,np.mean(null),np.max(null),np.percentile(null,[2.5,97.5]))
