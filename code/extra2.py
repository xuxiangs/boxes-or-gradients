import numpy as np, json, math
src=open('hgdp_china_analysis.py').read().split('# ---------- Test 1')[0]
exec(src)
china=groups_of(CHINA); idx_c=np.concatenate(china); labs_c=pop[idx_c]
def km(a, b):
    p1, p2 = map(math.radians, (lat[a], lat[b])); dl = math.radians(lon[b] - lon[a])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))
labs=np.unique(labs_c); Xz=np.nan_to_num(X[idx_c]); K=np.array([len(c) for c in cols_of])[loc_of]
C=np.array([Xz[labs_c==l].sum(axis=0) for l in labs])
def logp(counts):
    m=np.zeros(NL); np.add.at(m,loc_of,counts); return np.log((counts+1.0/K)/(m[loc_of]+1.0))
LP=np.array([logp(c) for c in C]); S=Xz@LP.T
obs=[]; rnd=[]
for i in range(len(idx_c)):
    g=np.where(labs==labs_c[i])[0][0]; s=S[i].copy(); s[g]=Xz[i]@logp(C[g]-Xz[i]); a=np.argmax(s)
    if a!=g:
        obs.append(km(labs[a],labs_c[i])); rnd.append(np.mean([km(l,labs_c[i]) for l in labs if l!=labs_c[i]]))
obs=np.array(obs); rnd=np.array(rnd)
# permutation-style check: how often is observed mean below mean of random wrong groups
rng2=np.random.default_rng(1); sims=[]
for _ in range(10000):
    tot=0
    for i in range(len(idx_c)):
        pass
    break
print(len(obs), obs.mean(), np.median(obs), rnd.mean())
json.dump(dict(n_wrong=len(obs),mean_km_wrong=float(obs.mean()),median_km_wrong=float(np.median(obs)),mean_km_random_wrong=float(rnd.mean())),open('extra2.json','w'))
