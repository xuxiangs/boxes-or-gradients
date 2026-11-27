"""Maps for the GEB2207 Topic 6 deck. Base outlines: Apache ECharts 4.9 china.json (province outlines, Apache-2.0)
and world-atlas land-110m (Natural Earth, public domain). Points: HGDP-CEPH sampling places; colours from our own analysis."""
import json, numpy as np, pickle, math, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
BG='#faf9f5'; INK='#141413'; TXT='#3d3d3a'; MUT='#6c6a64'; GRID='#e6dfd8'; N='#1f7566'; S='#e8a55a'; SS='#a8661c'; LAND='#efe9de'; EDGE='#cfc7ba'
plt.rcParams.update({'font.family':'Inter','font.size':17.5,'text.color':TXT,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG})
OUT='../figures/'; W=''
D=pickle.load(open('../data/ehgdp.pkl','rb')); info=D['info']
name={int(x.Label):x.Population for x in info.itertuples()}; lat={int(x.Label):float(x.Latitude) for x in info.itertuples()}; lon={int(x.Label):float(x.Longitude) for x in info.itertuples()}
nice={'Han':'Han (pooled)','Han-NChina':'Han (north)','Mongola':'Mongol','Uygur':'Uyghur','Xibo':'Xibe'}
CH=list(range(35,51)); pcs=np.load(W+'pcs_china.npy'); labs=np.load(W+'labs_china.npy')

def decode(s, off, scale=1024.0):
    out=[]; px,py=off
    for i in range(0,len(s),2):
        x=ord(s[i])-64; y=ord(s[i+1])-64
        x=(x>>1)^(-(x&1)); y=(y>>1)^(-(y&1)); x+=px; y+=py; px,py=x,y
        out.append((x/scale,y/scale))
    return np.array(out)
cj=json.load(open('node_modules/echarts/map/json/china.json')); rings=[]
for f in cj['features']:
    g=f['geometry']; co=g['coordinates']; eo=g['encodeOffsets']
    if g['type']=='Polygon':
        for s,o in zip(co,eo): rings.append(decode(s,o))
    else:
        for poly,offs in zip(co,eo):
            for s,o in zip(poly,offs): rings.append(decode(s,o))

def albers(lo,la,p1=25,p2=47,l0=105,p0=36):
    lo,la=np.radians(lo),np.radians(la); p1,p2,l0,p0=map(math.radians,(p1,p2,l0,p0))
    n=(math.sin(p1)+math.sin(p2))/2; C=math.cos(p1)**2+2*n*math.sin(p1)
    rho=np.sqrt(C-2*n*np.sin(la))/n; rho0=math.sqrt(C-2*n*math.sin(p0))/n; th=n*(lo-l0)
    return rho*np.sin(th), rho0-rho*np.cos(th)
def base(w,h,line=True):
    f=plt.figure(figsize=(w/100,h/100),dpi=200); ax=f.add_axes([0,0,1,1]); ax.set_axis_off(); ax.set_aspect('equal')
    polys=[np.column_stack(albers(r[:,0],r[:,1])) for r in rings if len(r)>2]
    ax.add_collection(PolyCollection(polys,facecolor=LAND,edgecolor=EDGE,linewidths=1.2,zorder=1))
    if line:
        lo=np.linspace(74,134,200); x,y=albers(lo,np.full_like(lo,34.0)); ax.plot(x,y,color='#b5ad9f',lw=2,ls=(0,(6,5)),zorder=2)
        x,y=albers(np.array([75.0]),np.array([34.0])); ax.text(x[0]-0.012,y[0]+0.012,'34°N',color=MUT,ha='right',va='bottom',fontsize=16)
    ax.set_xlim(-0.66,0.60); ax.set_ylim(-0.345,0.375)
    # South China Sea islands, drawn as the customary inset (outline after ECharts)
    pts=[[[0,3.5],[7,11.2],[15,11.9],[30,7],[42,0.7],[52,0.7],[56,7.7],[59,0.7],[64,0.7],[64,0],[5,0],[0,3.5]],[[13,16.1],[19,14.7],[16,21.7],[11,23.1],[13,16.1]],[[12,32.2],[14,38.5],[15,38.5],[13,32.2],[12,32.2]],[[16,47.6],[12,53.2],[13,53.2],[18,47.6],[16,47.6]],[[6,64.4],[8,70],[9,70],[8,64.4],[6,64.4]],[[23,82.6],[29,79.8],[30,79.8],[25,82.6],[23,82.6]],[[37,70.7],[43,62.3],[44,62.3],[39,70.7],[37,70.7]],[[48,51.1],[51,45.5],[53,45.5],[50,51.1],[48,51.1]],[[51,35],[51,28.7],[53,28.7],[53,35],[51,35]],[[52,22.4],[55,17.5],[56,17.5],[53,22.4],[52,22.4]],[[58,12.6],[62,7],[63,7],[60,12.6],[58,12.6]]]
    ia=f.add_axes([0.905,0.035,0.075,0.075*93.1/64*w/h]); ia.set_xlim(0,64); ia.set_ylim(93.1,0); ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values(): sp.set_color(EDGE); sp.set_linewidth(1.5)
    ia.add_collection(PolyCollection([np.array(p) for p in pts],facecolor=LAND,edgecolor='#b5ad9f',linewidths=1.2))
    return f,ax
off={'Han':(15,-3,'left'),'Han-NChina':(12,6,'left'),'Dai':(-8,-17,'right'),'Lahu':(-13,8,'right'),'Naxi':(-12,8,'right'),'Yi':(-4,19,'center'),'Miao':(4,-19,'center'),'Tujia':(12,10,'left'),'She':(13,0,'left'),
     'Tu':(-12,4,'right'),'Xibo':(4,-19,'center'),'Uygur':(4,19,'center'),'Mongola':(-12,-2,'right'),'Daur':(2,-19,'center'),'Oroqen':(-2,19,'center'),'Hezhen':(4,19,'center')}
def points(ax,colour,edge,size):
    for l in CH:
        x,y=albers(np.array([lon[l]]),np.array([lat[l]])); ax.scatter(x,y,s=size(l),color=colour(l),edgecolor=edge(l),lw=2,zorder=4)
        dx,dy,ha=off[name[l]]; ax.annotate(nice.get(name[l],name[l]),(x[0],y[0]),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',color=INK,fontsize=17,zorder=5)

# 1. where the 16 samples were collected
f,ax=base(860,602)
points(ax,lambda l: N if lat[l]>=34 else S,lambda l: BG if lat[l]>=34 else SS,lambda l: 70+int((labs==l).sum())*9)
f.savefig(OUT+'map_samples.png'); plt.close(f)

# 2. the same places coloured by their mean position on PC1
m={l:pcs[labs==l,0].mean() for l in CH}; cmap=LinearSegmentedColormap.from_list('g',['#c2643f','#e8a55a','#efe3c8','#5db8a6','#1f7566']); vmax=3.0
f,ax=base(816,510)
points(ax,lambda l: cmap((m[l]+vmax)/(2*vmax)),lambda l: INK,lambda l: 230)
ca=f.add_axes([0.06,0.085,0.27,0.03]); g=np.linspace(0,1,256)[None,:]; ca.imshow(g,aspect='auto',cmap=cmap); ca.set_yticks([]); ca.set_xticks([28,227]); ca.set_xticklabels(['south','north'],fontsize=16,color=MUT); ca.tick_params(length=0,pad=5)
for sp in ca.spines.values(): sp.set_visible(False)
f.text(0.06,0.14,'Group mean on PC1',fontsize=16,color=INK,fontweight='semibold')
f.savefig(OUT+'map_pc1.png'); plt.close(f)

# 2b. cover variant without labels
f,ax=base(800,500,line=False)
for l in CH:
    x,y=albers(np.array([lon[l]]),np.array([lat[l]])); ax.scatter(x,y,s=330,color=cmap((m[l]+vmax)/(2*vmax)),edgecolor=INK,lw=2,zorder=4)
f.savefig(OUT+'map_cover.png'); plt.close(f)

# 3. the worldwide panel, with the China samples highlighted
wj=json.load(open('land110.geojson')); wr=[]
def addring(r):
    a=np.array(r)
    if len(a)>2 and a[:,1].max()>-60:
        a=a[:,:2].copy()
        if np.ptp(a[:,0])>300: a[a[:,0]<-100,0]+=360      # Chukotka: unwrap across the antimeridian
        wr.append(a)
gm=wj['geometry'] if 'geometry' in wj else None
feats=wj['features'] if 'features' in wj else [wj]
for ft in feats:
    g=ft['geometry']
    if g['type']=='Polygon': addring(g['coordinates'][0])
    else:
        for poly in g['coordinates']: addring(poly[0])
f=plt.figure(figsize=(7.0,3.3),dpi=200); ax=f.add_axes([0,0,1,1]); ax.set_axis_off()
ax.add_collection(PolyCollection(wr,facecolor=LAND,edgecolor='none',zorder=1)); ax.set_xlim(-168,192); ax.set_ylim(-58,82); ax.set_aspect(1.25)
allp=[int(x.Label) for x in info.itertuples()]
o=[l for l in allp if l not in CH]; ax.scatter([lon[l] for l in o],[lat[l] for l in o],s=60,color='#8e8b82',edgecolor=BG,lw=1,zorder=3)
ax.scatter([lon[l] for l in CH],[lat[l] for l in CH],s=75,color='#c2643f',edgecolor=BG,lw=1,zorder=4)
f.savefig(OUT+'map_world.png'); plt.close(f)
print('ok', min(m.values()), max(m.values()))
