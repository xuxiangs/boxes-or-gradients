"""Figures for the GEB2207 Topic 6 deck, drawn from hgdp_china_results.json (our own analysis)."""
import json, numpy as np, pickle, math, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BG='#faf9f5'; INK='#141413'; TXT='#3d3d3a'; MUT='#6c6a64'; GRID='#e6dfd8'; N='#1f7566'; S='#e8a55a'; SS='#a8661c'; ACC='#c2643f'
plt.rcParams.update({'font.family':'Inter','font.size':17.5,'text.color':TXT,'axes.labelcolor':MUT,'axes.edgecolor':TXT,
 'xtick.color':MUT,'ytick.color':MUT,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':1.6,'xtick.major.width':1.6,'ytick.major.width':1.6,'xtick.major.size':7,'ytick.major.size':7})
OUT='../figures/'
r=json.load(open('hgdp_china_results.json')); D=pickle.load(open('../data/ehgdp.pkl','rb')); info=D['info']
name={int(x.Label):x.Population for x in info.itertuples()}; lat={int(x.Label):float(x.Latitude) for x in info.itertuples()}; lon={int(x.Label):float(x.Longitude) for x in info.itertuples()}
nice={'Han':'Han (pooled)','Han-NChina':'Han (north)','Mongola':'Mongol','Uygur':'Uyghur','Xibo':'Xibe'}
CH=list(range(35,51)); pcs=np.load('pcs_china.npy'); labs=np.load('labs_china.npy')
col=lambda l: N if lat[l]>=34 else S; edge=lambda l: BG if lat[l]>=34 else SS
def fig(w,h): return plt.subplots(figsize=(w/100,h/100),dpi=200)
def lab(ax,x,y,t,dx,dy,ha='left',**k): ax.annotate(t,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',color=INK,fontsize=17.5,**k)

# A. where the 16 samples were collected
f,ax=fig(800,560)
ax.axhline(34,color=GRID,lw=2.5,ls=(0,(6,5)),zorder=0); ax.text(137,34.6,'34°N',ha='right',color=MUT)
off={'Han':(14,-2,'left'),'Han-NChina':(12,4,'left'),'Dai':(-10,-12,'right'),'Lahu':(-10,6,'right'),'Naxi':(-10,6,'right'),'Yi':(-6,16,'right'),'Miao':(6,-18,'left'),'Tujia':(10,10,'left'),'She':(12,0,'left'),
     'Tu':(-10,4,'right'),'Xibo':(10,-12,'left'),'Uygur':(10,12,'left'),'Mongola':(-10,-4,'right'),'Daur':(-4,-18,'center'),'Oroqen':(-4,17,'center'),'Hezhen':(10,-6,'left')}
for l in CH:
    n=int((labs==l).sum()); ax.scatter(lon[l],lat[l],s=70+n*9,color=col(l),edgecolor=edge(l),lw=2,zorder=3)
    dx,dy,ha=off[name[l]]; lab(ax,lon[l],lat[l],nice.get(name[l],name[l]),dx,dy,ha)
ax.set_xlim(74,138); ax.set_ylim(17,54); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)'); ax.set_xticks([80,100,120]); ax.set_yticks([20,30,40,50])
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_map.png'); plt.close(f)

# B. PCA cloud of the 178 people
f,ax=fig(800,500)
for north in (False,True):
    m=np.array([(lat[l]>=34)==north for l in labs]); ax.scatter(pcs[m,0],pcs[m,1],s=95,color=N if north else S,edgecolor=BG if north else SS,lw=1.5,alpha=.92,zorder=3 if north else 2)
lh=pcs[labs==40]; ax.annotate('Lahu, 8 people',(lh[:,0].mean()+0.5,lh[:,1].mean()),xytext=(60,0),textcoords='offset points',va='center',color=INK,arrowprops=dict(arrowstyle='-',color=MUT,lw=1.5))
ax.scatter([],[],s=95,color=N,label='sampled north of 34°N'); ax.scatter([],[],s=95,color=S,edgecolor=SS,lw=1.5,label='sampled south of 34°N')
ax.legend(frameon=False,loc='lower right',handletextpad=.3,borderaxespad=.2)
ax.set_xlabel('PC1 · 1.2% of all variation'); ax.set_ylabel('PC2 · 1.0%'); ax.set_xticks([-2,0,2,4]); ax.set_yticks([-8,-4,0])
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_pca.png'); plt.close(f)

# C. group mean PC1 against latitude
f,ax=fig(800,500)
m={l:pcs[labs==l,0].mean() for l in CH}; la=np.array([lat[l] for l in CH]); mm=np.array([m[l] for l in CH]); sl,ic=np.polyfit(la,mm,1)
xs=np.array([19,52]); ax.plot(xs,sl*xs+ic,color=TXT,lw=2.5,zorder=1)
off={'Han':(12,-10,'left'),'Han-NChina':(12,-6,'left'),'Dai':(12,-6,'left'),'Lahu':(10,12,'left'),'Naxi':(-12,6,'right'),'Yi':(-4,17,'center'),'Miao':(-12,4,'right'),'Tujia':(2,-19,'center'),'She':(12,-4,'left'),
     'Tu':(-12,6,'right'),'Xibo':(-12,-10,'right'),'Uygur':(-12,6,'right'),'Mongola':(12,-8,'left'),'Daur':(12,-10,'left'),'Oroqen':(-12,4,'right'),'Hezhen':(-10,14,'right')}
for l in CH:
    ax.scatter(lat[l],m[l],s=170,color=col(l),edgecolor=edge(l),lw=2,zorder=3); dx,dy,ha=off[name[l]]; lab(ax,lat[l],m[l],nice.get(name[l],name[l]),dx,dy,ha)
ax.text(19.5,3.05,'r = 0.92,  p = 0.0001',color=INK,fontweight='semibold',fontsize=19)
ax.set_xlim(18,55); ax.set_ylim(-3.2,3.6); ax.set_xlabel('Latitude where the group was sampled (°N)'); ax.set_ylabel('Group mean on PC1'); ax.set_xticks([20,30,40,50]); ax.set_yticks([-2,0,2])
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_pc1lat.png'); plt.close(f)

# D. isolation by distance
f,ax=fig(1000,560)
ps=r['test3']['pairs']; km=np.array([p['km'] for p in ps]); g=np.array([p['gst']*100 for p in ps]); lh=np.array(['Lahu' in (p['a'],p['b']) for p in ps])
s2,i2=np.polyfit(km,g,1); xs=np.array([0,4300]); ax.plot(xs,s2*xs+i2,color=TXT,lw=2.5,zorder=1)
ax.scatter(km[~lh],g[~lh],s=95,color=N,edgecolor=BG,lw=1.5,zorder=3,label='pair of ethnic-group samples'); ax.scatter(km[lh],g[lh],s=95,color=S,edgecolor=SS,lw=1.5,zorder=2,label='pair that includes the Lahu')
ax.legend(frameon=False,loc='upper left',handletextpad=.3,borderaxespad=.1)
ax.set_xlim(-80,4400); ax.set_ylim(-0.12,1.85); ax.set_xticks([0,1000,2000,3000,4000]); ax.set_xticklabels(['0','1,000','2,000','3,000','4,000']); ax.set_yticks([0,0.5,1.0,1.5]); ax.set_yticklabels(['0','0.5%','1.0%','1.5%'])
ax.set_xlabel('Distance between the two sampling places (km)'); ax.set_ylabel('Variation between the two groups')
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_ibd.png'); plt.close(f)

# E. assignment accuracy against number of markers
f,ax=fig(960,560); t=r['test4b']
for key,c,text,yo in (('world_regions',INK,'7 world regions',0),('china_north_south',N,'China: north or south',0),('china_groups',ACC,'China: 16 ethnic-group samples',0)):
    L=np.array([int(k) for k in t[key]]); a=np.array(list(t[key].values()))*100
    ax.plot(L,a,color=c,lw=3.5,marker='o',ms=10,mec=BG,mew=1.8,zorder=3)
ax.text(38,82,'7 world regions',color=INK,fontweight='semibold',ha='right'); ax.text(125,66,'China: north or south',color=N,fontweight='semibold',ha='left',va='top')
ax.text(660,30,'China: 16 ethnic-group samples',color='#a04e30',fontweight='semibold',ha='right')
for y,txt in ((50,'chance for north or south'),(14.3,'chance for regions'),(6.25,'chance for 16 samples')):
    ax.axhline(y,color=GRID,lw=2,ls=(0,(5,4)),zorder=0)
ax.text(860,45,'chance, north or south',color=MUT,fontsize=15.5,ha='right'); ax.text(1.05,2.2,'chance levels: 14% (regions), 6% (16 samples)',color=MUT,fontsize=15.5)
ax.set_xscale('log'); ax.set_xlim(0.85,900); ax.set_ylim(0,102); ax.set_xticks([1,10,100,678]); ax.set_xticklabels(['1','10','100','678']); ax.minorticks_off(); ax.set_yticks([0,25,50,75,100]); ax.set_yticklabels(['0','25%','50%','75%','100%'])
ax.set_xlabel('Number of markers used (log scale)'); ax.set_ylabel('People assigned to their own group')
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_assign.png'); plt.close(f)

# F. label-shuffle test
f,ax=fig(820,330); pn=json.load(open('perm_null.json')); nul=np.array(pn['null'])*100; ob=pn['obs']*100; ci=np.array(r['test1']['ci_china'])*100
ax.hist(nul,bins=np.arange(0,0.30,0.012),color='#b5ad9f',edgecolor=BG,lw=1,zorder=2); ax.axvspan(ci[0],ci[1],color=ACC,alpha=.22,lw=0,zorder=1); ax.axvline(ob,color=ACC,lw=4,zorder=3)
ymax=ax.get_ylim()[1]; ax.text(0.25,ymax*0.45,'1,000 shuffles of the labels',color=MUT,ha='left'); ax.text(ob-0.12,ymax*0.85,'observed 0.96%',color='#a04e30',fontweight='semibold',ha='right')
ax.set_xlim(0,1.12); ax.set_xticks([0,0.25,0.5,0.75,1.0]); ax.set_xticklabels(['0','0.25%','0.5%','0.75%','1.0%']); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Share of variation between the 16 groups')
f.tight_layout(pad=0.6); f.savefig(OUT+'fig_perm.png'); plt.close(f)
print('ok')
