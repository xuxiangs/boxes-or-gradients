"""
GEB2207 Topic 6 - "Distinguishable is not the same as different"
A minimal simulation of the Lewontin (1972) / Edwards (2003) paradox.

Model (Balding-Nichols):
  * two populations that split from one ancestral population
  * L independent bi-allelic SNPs; ancestral allele frequency p ~ Uniform(0.05, 0.95)
  * each population's frequency ~ Beta(p(1-F)/F, (1-p)(1-F)/F), where F = FST
  * a diploid test individual is drawn from one population and assigned to the
    population under which its genotype is more likely (best case: true
    frequencies are known)
Output: share of individuals assigned to the correct population, for each
FST and each number of SNPs. FST values are the published genome-wide values
quoted on the slide (Bhatia 2013; Yi 2010; Xu 2009).
"""
import numpy as np, json

rng = np.random.default_rng(2207)
FST = {"Europe vs East Asia (0.106)": 0.106,
       "Han vs Tibetan (0.026)": 0.026,
       "Northern vs southern Han (0.0012)": 0.00116}
LOCI = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
N_IND = 400          # test individuals per replicate (half from each population)
N_REP = 25           # independent replicates (new allele frequencies each time)

def accuracy(F, L):
    acc = []
    for _ in range(N_REP):
        p = rng.uniform(0.05, 0.95, L)
        a, b = p * (1 - F) / F, (1 - p) * (1 - F) / F
        pA = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
        pB = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
        # per-allele log-likelihood ratios (A vs B)
        l1, l0 = np.log(pA / pB), np.log((1 - pA) / (1 - pB))
        half = N_IND // 2
        gA = rng.binomial(2, pA, (half, L))      # individuals truly from A
        gB = rng.binomial(2, pB, (half, L))      # individuals truly from B
        sA = gA @ l1 + (2 - gA) @ l0
        sB = gB @ l1 + (2 - gB) @ l0
        correct = (sA > 0).sum() + (sB < 0).sum() + 0.5 * ((sA == 0).sum() + (sB == 0).sum())
        acc.append(correct / N_IND)
    return float(np.mean(acc))

out = {}
for name, F in FST.items():
    out[name] = {"FST": F, "within_group_share": round(1 - F, 4),
                 "accuracy": {L: round(accuracy(F, L), 4) for L in LOCI}}
    print(name, out[name]["accuracy"])
json.dump(out, open("assignment_simulation_results.json", "w"), indent=1)
