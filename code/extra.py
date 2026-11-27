import numpy as np, json, math, importlib.util, sys, io, contextlib
# reuse functions from the main script without re-running the slow parts
src=open('hgdp_china_analysis.py').read().split('# ---------- Test 1')[0]
exec(src)
r=json.load(open('hgdp_china_results.json'))
china=groups_of(CHINA); idx_c=np.concatenate(china); labs_c=pop[idx_c]
def km(a, b):
    p1, p2 = map(math.radians, (lat[a], lat[b])); dl = math.radians(lon[b] - lon[a])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))
L3=[l for l in CHINA if name[l]!='Lahu']
ps=r['test3']['pairs']; nm={v:k for k,v in name.items()}
def mantel_from_pairs(labels,nperm=10000):
    n=len(labels); G=np.zeros((n,n)); K=np.zeros((n,n))
    d={(p['a'],p['b']):p for p in ps}
    for i in range(n):
        for j in range(i+1,n):
            a,b=name[labels[i]],name[labels[j]]; p=d.get((a,b)) or d.get((b,a)); G[i,j]=G[j,i]=p['gst']; K[i,j]=K[j,i]=p['km']
    iu=np.triu_indices(n,1); rr=np.corrcoef(G[iu],K[iu])[0,1]; c=0
    for _ in range(nperm):
        q=rng.permutation(n)
        if np.corrcoef(G[q][:,q][iu],K[iu])[0,1]>=rr: c+=1
    return float(rr), (1+c)/(1+nperm)
print('mantel no Lahu',mantel_from_pairs(L3))
print('gst china no Lahu',gst(groups_of(L3),500))
# assignment with all loci: where do people go?
labs=np.unique(labs_c); Xz=np.nan_to_num(X[idx_c]); K=np.array([len(c) for c in cols_of])[loc_of]
C=np.array([Xz[labs_c==l].sum(axis=0) for l in labs])
def logp(counts):
    m=np.zeros(NL); np.add.at(m,loc_of,counts); return np.log((counts+1.0/K)/(m[loc_of]+1.0))
LP=np.array([logp(c) for c in C]); S=Xz@LP.T; own=0; samehalf=0; near=[]
for i in range(len(idx_c)):
    g=np.where(labs==labs_c[i])[0][0]; s=S[i].copy(); s[g]=Xz[i]@logp(C[g]-Xz[i]); a=np.argmax(s)
    own+= a==g; samehalf+= (lat[labs[a]]>=34)==(lat[labs_c[i]]>=34)
    near.append(km(labs[a],labs_c[i]))
exp=np.mean([km(a,b) for a in CHINA for b in CHINA])
print('own',own/len(idx_c),'same N/S half',samehalf/len(idx_c),'median km to assigned group',np.median(near),'mean',np.mean(near),'mean km between random groups',exp)
json.dump(dict(mantel_no_lahu=mantel_from_pairs(L3),own=own/len(idx_c),same_half=samehalf/len(idx_c),median_km=float(np.median(near)),mean_km=float(np.mean(near)),mean_km_random=float(exp)),open('extra.json','w'))
