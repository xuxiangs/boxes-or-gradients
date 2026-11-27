"""Animated figure for slide 14 (Test 2): the PCA cloud of the 178 people with one ethnic label lit at a time,
from the southernmost sample to the northernmost. The frames reuse the axes of figures/fig_pca.png, so every
dot sits on exactly the same pixel as in that picture and the two can be faded into each other.

Run from the code folder:   python make_fig_pca_labels.py
Optional arguments, in order: data pickle, base figure, output gif, a TrueType file of the Inter typeface
(the other figures use Inter; without it matplotlib falls back to its default sans)."""
import sys, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image

BG = '#faf9f5'; INK = '#141413'; TXT = '#3d3d3a'; MUT = '#6c6a64'; ACC = '#c2643f'; REST = '#cdc6b9'
NICE = {'Han': 'Han (pooled)', 'Han-NChina': 'Han (north)', 'Mongola': 'Mongol', 'Uygur': 'Uyghur', 'Xibo': 'Xibe'}
CHINA = list(range(35, 51))
DPI = 200


def china_pca(data_path):
    """PC scores of the people sampled in China, computed as in hgdp_china_analysis.py (Test 2)."""
    D = pickle.load(open(data_path, 'rb'))
    X = D['X'].astype(np.float64); X[D['X'] == D['NA']] = np.nan
    pop = D['pop'].astype(int); info = D['info']
    name = {int(r.Label): r.Population for r in info.itertuples()}
    lat = {int(r.Label): float(r.Latitude) for r in info.itertuples()}
    idx = np.concatenate([np.where(pop == l)[0] for l in CHINA]); labs = pop[idx]
    A = X[idx] / 2.0
    A = A[:, np.nanstd(A, axis=0) > 0]; mu = np.nanmean(A, axis=0)
    A = np.where(np.isnan(A), mu, A) - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    pcs = (U * S)[:, :2]
    # the sign of a principal component is arbitrary; use the orientation of fig_pca.png
    north = np.array([lat[l] >= 34 for l in labs])
    if pcs[north, 0].mean() < pcs[~north, 0].mean(): pcs[:, 0] *= -1          # northern samples on the right
    lahu = [l for l in CHINA if name[l] == 'Lahu'][0]
    if pcs[labs == lahu, 1].mean() > 0: pcs[:, 1] *= -1                       # the Lahu at the bottom
    return pcs, labs, name, lat


def axes_of(base_path):
    """Read the axes of an existing figure: returns the image and the pixel position of data point (x, y)."""
    im = np.asarray(Image.open(base_path).convert('RGB')).astype(int)
    H, W = im.shape[:2]
    ink = np.abs(im - np.array([0x3d, 0x3d, 0x3a])).sum(axis=2) < 60          # axis lines
    tick = np.abs(im - np.array([0x6c, 0x6a, 0x64])).sum(axis=2) < 40         # tick marks (muted grey)
    cols = np.where(ink.sum(axis=0) > 0.5 * H)[0]; rows = np.where(ink.sum(axis=1) > 0.5 * W)[0]
    xl0, xl1, yb0, yb1 = cols.min(), cols.max(), rows.min(), rows.max()       # left spine, bottom spine
    def runs(mask):
        i = np.where(mask)[0]; groups = np.split(i, np.where(np.diff(i) > 1)[0] + 1)
        return np.array([g.mean() + 0.5 for g in groups if len(g) >= 2])
    xt = runs(tick[yb1 + 4:yb1 + 14, :].all(axis=0))                          # tick marks under the x axis
    yt = runs(tick[:, xl0 - 13:xl0 - 3].all(axis=1))                          # tick marks left of the y axis
    assert len(xt) == 4 and len(yt) == 3, (xt, yt)
    ax_, bx_ = np.polyfit([-2, 0, 2, 4], xt, 1); ay_, by_ = np.polyfit([0, -4, -8], yt, 1)
    assert np.abs(np.polyval([ax_, bx_], [-2, 0, 2, 4]) - xt).max() < 0.75
    assert np.abs(np.polyval([ay_, by_], [0, -4, -8]) - yt).max() < 0.75
    xr = np.where(ink[yb0:yb1 + 1, :].all(axis=0))[0].max()                   # right end of the x axis
    geo = dict(W=W, H=H, ax=ax_, bx=bx_, ay=ay_, by=by_, xl1=xl1, yb0=yb0, xr=xr)
    return im.astype(np.uint8), geo


def blank_axes(im, geo):
    """The base figure with everything inside the axes removed (dots, legend, annotation)."""
    out = im.copy()
    out[:geo['yb0'] - 1, geo['xl1'] + 2:, :] = np.array([0xfa, 0xf9, 0xf5], dtype=np.uint8)
    return Image.fromarray(out).convert('RGBA')


def overlay(geo, draw):
    """Draw on a transparent canvas whose data coordinates match the base figure pixel for pixel."""
    W, H = geo['W'], geo['H']
    f = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); ax = f.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim((0 - geo['bx']) / geo['ax'], (W - geo['bx']) / geo['ax'])
    ax.set_ylim((H - geo['by']) / geo['ay'], (0 - geo['by']) / geo['ay'])
    draw(f, ax)
    f.canvas.draw(); buf = np.asarray(f.canvas.buffer_rgba()).copy(); plt.close(f)
    assert buf.shape[:2] == (H, W), buf.shape
    return Image.fromarray(buf, 'RGBA')


def to_palette(frame, cols):
    """Map every pixel to its nearest palette colour (exact search, no dithering)."""
    a = np.asarray(frame); u, inv = np.unique(a.reshape(-1, 3), axis=0, return_inverse=True)
    near = np.concatenate([((u[i:i + 4000, None, :].astype(np.int32) - cols[None].astype(np.int32)) ** 2).sum(axis=2).argmin(axis=1)
                           for i in range(0, len(u), 4000)]).astype(np.uint8)
    im = Image.fromarray(near[inv.reshape(-1)].reshape(a.shape[:2]), 'P'); im.putpalette([int(v) for v in cols.flatten()])
    return im


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else '../data/ehgdp.pkl'
    base = sys.argv[2] if len(sys.argv) > 2 else '../figures/fig_pca.png'
    out = sys.argv[3] if len(sys.argv) > 3 else '../figures/fig_pca_labels.gif'
    if len(sys.argv) > 4: fm.fontManager.addfont(sys.argv[4]); plt.rcParams['font.family'] = 'Inter'
    plt.rcParams.update({'font.size': 17.5, 'figure.facecolor': 'none', 'savefig.facecolor': 'none'})

    pcs, labs, name, lat = china_pca(data)
    im, geo = axes_of(base); clean = blank_axes(im, geo)
    W, H = geo['W'], geo['H']
    order = sorted(CHINA, key=lambda l: (lat[l], name[l]))                   # south to north
    x0 = 0.545; y0 = 1 - (geo['yb0'] - 262) / H; step = 56 / H               # caption block, figure fractions

    frames = []
    for k, l in enumerate(order, 1):
        m = labs == l
        def draw(f, ax):
            ax.scatter(pcs[~m, 0], pcs[~m, 1], s=95, color=REST, edgecolor=BG, lw=1.5, zorder=2)
            ax.scatter(pcs[m, 0], pcs[m, 1], s=95, color=ACC, edgecolor=BG, lw=1.5, zorder=3)
            la = ('%.1f' % lat[l]).rstrip('0').rstrip('.')
            f.text(x0, y0, '●', color=ACC, fontsize=15, va='center', ha='center')
            f.text(x0 + 0.022, y0, NICE.get(name[l], name[l]), color=INK, fontsize=20, va='center')
            f.text(x0 + 0.022, y0 - step, '%s°N · %d of the 178 people' % (la, m.sum()), color=MUT, va='center')
            f.text(x0, y0 - 2 * step, '●', color=REST, fontsize=15, va='center', ha='center')
            f.text(x0 + 0.022, y0 - 2 * step, 'everyone else', color=MUT, va='center')
            f.text(x0 + 0.022, y0 - 3 * step, 'sample %d of 16, south to north' % k, color=MUT, fontsize=15.5, va='center')
        frames.append(Image.alpha_composite(clean, overlay(geo, draw)).convert('RGB'))

    # one shared palette, no dithering, so flat areas stay flat and the background stays exactly #faf9f5
    strip = Image.new('RGB', (W * 2, H)); strip.paste(frames[0], (0, 0)); strip.paste(frames[len(frames) // 2], (W, 0))
    cols = np.array(strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).getpalette()).reshape(-1, 3)
    cols[np.abs(cols - np.array([0xfa, 0xf9, 0xf5])).sum(axis=1).argmin()] = (0xfa, 0xf9, 0xf5)
    pframes = [to_palette(fr, cols) for fr in frames]
    assert pframes[0].convert('RGB').getpixel((2, 2)) == (0xfa, 0xf9, 0xf5)
    pframes[0].save(out, save_all=True, append_images=pframes[1:], duration=700, loop=0, optimize=False)
    print('wrote', out, len(pframes), 'frames')
    return frames, geo, pcs, labs, lat


if __name__ == '__main__':
    main()
