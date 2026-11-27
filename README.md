# GEB2207 Topic 6 · Boxes or gradients? Analysis bundle

Our own test of two hypotheses on public genotype data:
H1 (boxes) each ethnic label marks a separate gene pool; H2 (gradient) one gene pool that changes gradually with distance.

## Data
`data/eHGDP.rda` is the `eHGDP` data set shipped with the R package adegenet (Jombart 2008;
github.com/thibautjombart/adegenet). It holds the extended HGDP-CEPH panel: 1,350 people, 79 population
samples, 678 autosomal microsatellites (Cann et al. 2002; Rosenberg et al. 2002; Ramachandran et al. 2005;
Wang et al. 2007). `data/ehgdp.pkl` is the same table converted for Python (read with the `rdata` package).
We use the 16 samples collected in China: 178 people from 15 officially recognised ethnic groups.
`ehgdp.pkl` was saved with a recent pandas and does not load under pandas 2.2; upgrade pandas if `pickle.load`
raises a `StringDtype` error.

## Code (Python 3, NumPy, pandas, matplotlib). Run from the `code` folder.
| File | What it does |
|---|---|
| `hgdp_china_analysis.py` | The four tests: GST with bootstrap and label permutation, PCA with permutation-tested correlation, Mantel test, overlap omega and leave-one-out assignment. Writes `hgdp_china_results.json`. |
| `perm_null.py` | The 1,000 label shuffles behind the histogram on slide 17. |
| `extra.py`, `extra2.py` | Robustness checks: tests without the Lahu, where misassigned people land. |
| `make_figures.py` | Draws the six figures used on slides 15 and 17-20 and on backup slide 36. Run it after `hgdp_china_analysis.py` and `perm_null.py`. |
| `make_fig_pca_labels.py` | Animated version of the PCA figure for slide 18: the same 178 dots with one sample lit at a time, from south to north. Reuses the axes of `figures/fig_pca.png` and writes `figures/fig_pca_labels.gif`. |
| `assignment_simulation.py` | Balding-Nichols simulation behind slide 13. |
| `make_maps.py` | Draws the maps on the cover and the data slide. It also needs two outline files that are not in this repository: `china.json` from Apache ECharts 4.9 and `land110.geojson` from world-atlas (Natural Earth). |

The scripts expect the data at `../data/ehgdp.pkl` and write their result files next to themselves
(copies of those files are in `results/`); the figure scripts write to `../figures/`.

## Main results (also in `results/`)
| Test | Statistic | Result |
|---|---|---|
| 1 Size | GST between the 16 samples | 0.96% [0.87-1.06]; label shuffle max 0.24%, p < 0.001 |
| 2 Shape | PC1 group means against latitude | r = 0.92, p = 0.0001 |
| 3 Distance | Mantel test, 120 pairs | r = 0.35, p = 0.006 |
| 4 Individuals | omega; leave-one-out assignment | omega = 36%; own group 21% (chance 6%); north/south 90% (chance about 50%) |

Check of the code: world GST is 6.8% between populations and 3.7% between seven regions,
in line with the published 5-7% and 3-5% (Rosenberg et al. 2002).
