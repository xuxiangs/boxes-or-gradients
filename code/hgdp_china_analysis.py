"""
GEB2207 Topic 6 - Boxes or gradients?  Our own test on public genotype data.

Data : extended HGDP-CEPH panel, 1,350 people, 79 populations, 678 microsatellite loci
       (Rosenberg et al. 2002; Ramachandran et al. 2005; Wang et al. 2007), as distributed
       in the R package adegenet (data set `eHGDP`, extracted to ehgdp.pkl).
Focus: the 16 samples collected inside China (15 officially recognised ethnic groups; Han sampled twice).

Tests
 1. Variance partition  - Nei's G_ST with sample-size correction; bootstrap over loci (95% CI);
                          permutation of group labels (one-sided p).
 2. PCA                 - principal components of individual allele frequencies; correlation of
                          population mean scores with latitude / longitude (permutation p).
 3. Isolation by distance - pairwise G_ST against great-circle distance; Mantel test (10,000 permutations).
 4. Individuals         - (a) dissimilarity fraction omega (Witherspoon et al. 2007): how often a pair from
                          different groups is more similar than a pair from the same group;
                          (b) leave-one-out assignment accuracy as a function of the number of loci.
"""
import numpy as np, pickle, json, math
rng = np.random.default_rng(2207)
D = pickle.load(open('../data/ehgdp.pkl', 'rb'))
X, NA = D['X'].astype(np.float64), D['NA']
X[D['X'] == NA] = np.nan
info = D['info']; pop = D['pop'].astype(int)                       # population label 1..79 per individual
loc_of = np.array([D['loclev'].index(l) for l in D['locfac']])     # locus index per allele column
NL = len(D['loclev'])
cols_of = [np.where(loc_of == l)[0] for l in range(NL)]
name = {int(r.Label): r.Population for r in info.itertuples()}
region = {int(r.Label): r.Region for r in info.itertuples()}
lat = {int(r.Label): float(r.Latitude) for r in info.itertuples()}
lon = {int(r.Label): float(r.Longitude) for r in info.itertuples()}
CHINA = list(range(35, 51))
out = {}

def freqs(idx):
    """allele frequencies (per allele column) and chromosome counts (per locus) for a set of individuals"""
    c = np.nansum(X[idx], axis=0)
    m = np.array([c[cols].sum() for cols in cols_of])
    return c / np.maximum(m[loc_of], 1), m

def hs_ht(groups):
    """per-locus H_S and H_T (Nei & Chesser 1983) for a list of index arrays"""
    P, M = zip(*[freqs(g) for g in groups]); P, M = np.array(P), np.array(M, float)
    s = len(groups)
    ok = (M >= 2).all(axis=0)
    hom = np.array([[ (P[g, cols] ** 2).sum() for cols in cols_of] for g in range(s)])
    h = (M / np.maximum(M - 1, 1)) * (1 - hom)                       # unbiased gene diversity per group
    HS = h.mean(axis=0)
    pbar = P.mean(axis=0)
    mh = s / (1 / np.maximum(M, 1)).sum(axis=0)                      # harmonic mean of chromosome counts
    HT = 1 - np.array([(pbar[cols] ** 2).sum() for cols in cols_of]) + HS / (mh * s)
    return HS[ok], HT[ok]

def gst(groups, boot=0):
    HS, HT = hs_ht(groups)
    est = 1 - HS.sum() / HT.sum()
    if not boot: return est
    b = []
    for _ in range(boot):
        i = rng.integers(0, len(HS), len(HS)); b.append(1 - HS[i].sum() / HT[i].sum())
    return est, np.percentile(b, [2.5, 97.5])

def groups_of(labels): return [np.where(pop == l)[0] for l in labels]

# ---------- Test 1: where does the variation sit? ----------
world_pops = groups_of(range(1, 80))
regs = sorted(set(region.values()))
world_regs = [np.where(np.isin(pop, [l for l in range(1, 80) if region[l] == r]))[0] for r in regs]
g_world_pop, ci1 = gst(world_pops, 1000)
g_world_reg, ci2 = gst(world_regs, 1000)
china = groups_of(CHINA)
g_china, ci3 = gst(china, 1000)
china_nouy = groups_of([l for l in CHINA if name[l] != 'Uygur'])
g_china_nouy, ci4 = gst(china_nouy, 1000)
han2 = groups_of([35, 36]); g_han, ci5 = gst(han2, 1000)
# permutation test: shuffle ethnic labels among the people sampled in China
idx_c = np.concatenate(china); sizes = [len(g) for g in china]
null = []
for _ in range(1000):
    p = rng.permutation(idx_c); cuts = np.cumsum(sizes)[:-1]
    null.append(gst(np.split(p, cuts)))
null = np.array(null)
out['test1'] = dict(world_between_populations=g_world_pop, ci_world_pop=list(ci1),
                    world_between_regions=g_world_reg, ci_world_reg=list(ci2),
                    china_between_groups=g_china, ci_china=list(ci3),
                    china_without_uygur=g_china_nouy, ci_china_nouy=list(ci4),
                    han_north_vs_south=g_han, ci_han=list(ci5),
                    perm_null_mean=float(null.mean()), perm_null_max=float(null.max()),
                    perm_p=float((1 + (null >= g_china).sum()) / (1 + len(null))),
                    n_china=int(len(idx_c)), n_groups=len(china), n_loci=NL)
print('TEST1', json.dumps(out['test1'], indent=1))

# ---------- Test 2: PCA of the people sampled in China ----------
def pca(idx):
    A = X[idx] / 2.0
    keep = np.nanstd(A, axis=0) > 0
    A = A[:, keep]; mu = np.nanmean(A, axis=0)
    A = np.where(np.isnan(A), mu, A) - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U * S, (S ** 2) / (S ** 2).sum()
pcs, var = pca(idx_c)
labs_c = pop[idx_c]
def pop_corr(score, coord, nperm=10000):
    m = np.array([score[labs_c == l].mean() for l in CHINA]); c = np.array([coord[l] for l in CHINA])
    r = np.corrcoef(m, c)[0, 1]
    pr = np.array([abs(np.corrcoef(rng.permutation(m), c)[0, 1]) for _ in range(nperm)])
    return float(r), float((1 + (pr >= abs(r)).sum()) / (1 + nperm))
out['test2'] = dict(var_pc1=float(var[0]), var_pc2=float(var[1]), var_pc3=float(var[2]),
                    pc1_lon=pop_corr(pcs[:, 0], lon), pc1_lat=pop_corr(pcs[:, 0], lat),
                    pc2_lon=pop_corr(pcs[:, 1], lon), pc2_lat=pop_corr(pcs[:, 1], lat))
# the same without the Uygur sample (the most admixed, far-western sample)
keepi = np.array([name[l] != 'Uygur' for l in labs_c]); idx_n = idx_c[keepi]; labs_n = labs_c[keepi]
pcs_n, var_n = pca(idx_n)
def pop_corr_n(score, coord, nperm=10000):
    L = [l for l in CHINA if name[l] != 'Uygur']
    m = np.array([score[labs_n == l].mean() for l in L]); c = np.array([coord[l] for l in L])
    r = np.corrcoef(m, c)[0, 1]
    pr = np.array([abs(np.corrcoef(rng.permutation(m), c)[0, 1]) for _ in range(nperm)])
    return float(r), float((1 + (pr >= abs(r)).sum()) / (1 + nperm))
out['test2_no_uygur'] = dict(var_pc1=float(var_n[0]), var_pc2=float(var_n[1]),
                    pc1_lat=pop_corr_n(pcs_n[:, 0], lat), pc1_lon=pop_corr_n(pcs_n[:, 0], lon),
                    pc2_lat=pop_corr_n(pcs_n[:, 1], lat), pc2_lon=pop_corr_n(pcs_n[:, 1], lon))
print('TEST2', json.dumps(out['test2'], indent=1), json.dumps(out['test2_no_uygur'], indent=1))
np.save('pcs_china.npy', pcs[:, :4]); np.save('pcs_china_nouy.npy', pcs_n[:, :4])
np.save('labs_china.npy', labs_c); np.save('labs_china_nouy.npy', labs_n)

# ---------- Test 3: isolation by distance among the 16 samples ----------
def km(a, b):
    p1, p2 = map(math.radians, (lat[a], lat[b])); dl = math.radians(lon[b] - lon[a])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))
def mantel(labels, nperm=10000):
    n = len(labels); G = np.zeros((n, n)); K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            G[i, j] = G[j, i] = gst(groups_of([labels[i], labels[j]])); K[i, j] = K[j, i] = km(labels[i], labels[j])
    iu = np.triu_indices(n, 1); r = np.corrcoef(G[iu], K[iu])[0, 1]; cnt = 0
    for _ in range(nperm):
        p = rng.permutation(n)
        if np.corrcoef(G[p][:, p][iu], K[iu])[0, 1] >= r: cnt += 1
    return G, K, float(r), float((1 + cnt) / (1 + nperm))
G, K, r_all, p_all = mantel(CHINA)
L2 = [l for l in CHINA if name[l] != 'Uygur']
G2, K2, r_nouy, p_nouy = mantel(L2)
out['test3'] = dict(mantel_r=r_all, mantel_p=p_all, mantel_r_no_uygur=r_nouy, mantel_p_no_uygur=p_nouy,
                    pairs=[dict(a=name[CHINA[i]], b=name[CHINA[j]], km=round(K[i, j]), gst=round(float(G[i, j]), 5))
                           for i in range(16) for j in range(i + 1, 16)])
print('TEST3', r_all, p_all, r_nouy, p_nouy)

# ---------- Test 4a: are two people from one group more alike than two from different groups? ----------
def pair_dist(idx):
    n = len(idx); Dm = np.zeros((n, n)); Xi = X[idx]
    miss = np.array([np.isnan(Xi[:, cols[0]]) for cols in cols_of]).T          # n x loci
    Xz = np.nan_to_num(Xi)
    for i in range(n):
        shared = np.minimum(Xz[i], Xz).reshape(n, -1)                            # alleles shared, per column
        sl = np.array([shared[:, cols].sum(axis=1) for cols in cols_of]).T      # n x loci
        ok = ~(miss[i] | miss)
        Dm[i] = ((2 - sl) / 2 * ok).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
    return Dm
def omega(Dm, labels):
    iu = np.triu_indices(len(labels), 1); same = (labels[:, None] == labels[None, :])[iu]; d = Dm[iu]
    w, b = d[same], d[~same]
    order = np.argsort(np.concatenate([w, b])); ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    U = ranks[:len(w)].sum() - len(w) * (len(w) + 1) / 2          # number of (within, between) pairs with within > between
    return float(U / (len(w) * len(b))), float(w.mean()), float(b.mean())
Dc = pair_dist(idx_c)
om_china = omega(Dc, labs_c)
om_china_nouy = omega(Dc[np.ix_(keepi, keepi)], labs_n)
sub = np.concatenate([rng.choice(g, min(25, len(g)), replace=False) for g in world_regs])   # balanced world sample
reg_lab = np.array([regs.index(region[l]) for l in pop[sub]])
Dw = pair_dist(sub); om_world = omega(Dw, reg_lab)
out['test4a'] = dict(omega_china_groups=om_china[0], mean_within=om_china[1], mean_between=om_china[2],
                     omega_china_no_uygur=om_china_nouy[0], omega_world_regions=om_world[0],
                     world_within=om_world[1], world_between=om_world[2], n_world=int(len(sub)))
print('TEST4a', json.dumps(out['test4a'], indent=1))

# ---------- Test 4b: leave-one-out assignment, as a function of the number of loci ----------
def assign_curve(idx, labels, loci_grid, reps=20):
    labs = np.unique(labels); Xz = np.nan_to_num(X[idx]); K = np.array([len(c) for c in cols_of])[loc_of]
    C = np.array([Xz[labels == l].sum(axis=0) for l in labs])                    # allele counts per group
    res = {}
    for L in loci_grid:
        acc = []
        for _ in range(reps if L < NL else 1):
            ls = rng.choice(NL, L, replace=False); cm = np.isin(loc_of, ls)
            Cs, Xs, Ks = C[:, cm], Xz[:, cm], K[cm]; lo = loc_of[cm]
            uniq, inv = np.unique(lo, return_inverse=True)
            def logp(counts):
                m = np.zeros(len(uniq)); np.add.at(m, inv, counts)
                return np.log((counts + 1.0 / Ks) / (m[inv] + 1.0))
            LP = np.array([logp(c) for c in Cs]); S = Xs @ LP.T; correct = 0
            for i in range(len(idx)):
                g = np.where(labs == labels[i])[0][0]; s = S[i].copy()
                s[g] = Xs[i] @ logp(Cs[g] - Xs[i])                                # leave the person out of their own group
                correct += (np.argmax(s) == g)
            acc.append(correct / len(idx))
        res[L] = float(np.mean(acc))
    return res
grid = [1, 2, 5, 10, 20, 50, 100, 200, 400, 678]
acc_world = assign_curve(sub, reg_lab, grid)
acc_china = assign_curve(idx_c, labs_c, grid)
ns = np.array([0 if lat[l] >= 34 else 1 for l in labs_c])       # north vs south of about 34 N (Qinling-Huaihe line)
acc_ns = assign_curve(idx_c, ns, grid)
out['test4b'] = dict(world_regions=acc_world, china_groups=acc_china, china_north_south=acc_ns,
                     chance_world=1 / len(regs), chance_china=1 / 16, n_north=int((ns == 0).sum()), n_south=int((ns == 1).sum()))
print('TEST4b', json.dumps(out['test4b'], indent=1))
out['meta'] = dict(china_samples={name[l]: dict(n=int((pop == l).sum()), lat=lat[l], lon=lon[l]) for l in CHINA},
                   n_total=int(len(pop)), n_pops=79, n_loci=NL, regions=regs)
json.dump(out, open('hgdp_china_results.json', 'w'), indent=1)
