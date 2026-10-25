#!/usr/bin/env python3
"""Plot two-carrier avalanche-feedback screening contour and regression.

Important: this is NOT the fully self-consistent avalanche-corrected VBD surface.
It uses converged Poisson-DD e+h carriers WITHOUT G_II feedback for the base field,
then computes the small-signal two-carrier G_II feedback gain and finds gain=1.
The separately successful self-consistent II source solves are overlaid as samples.
"""
import os,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from jfet_Ld10_avalanche_regression import fit_kfold, polynomial_design

import argparse
HERE=os.path.dirname(os.path.abspath(__file__))
_ap=argparse.ArgumentParser()
_ap.add_argument('--input',default=os.path.join(HERE,'jfet_vbd_Ld10_fullavalanche_results.jsonl'))
_ap.add_argument('--prefix',default=os.path.join(HERE,'jfet_Ld10_eh_feedback_screen'))
_args=_ap.parse_args()
PFX=_args.prefix
rows=[json.loads(z) for z in open(_args.input) if z.strip()]
aa=np.array(sorted({r['a'] for r in rows}))
na=np.array(sorted({r['NA'] for r in rows}))
A,NA=np.meshgrid(aa,na)
base=np.full(A.shape,np.nan)
full=np.full(A.shape,np.nan)
mask=np.zeros(A.shape,dtype=int) # 0 missing,1 not-off,2 usable full,3 only-screen,4 outside coeff validity,5 failed
raw={(r['a'],r['NA']):r for r in rows}
for j,N in enumerate(na):
    for i,a in enumerate(aa):
        r=raw.get((a,N))
        if r is None: continue
        off=r.get('nchan',np.inf)<0.5e16
        if not off:
            mask[j,i]=1;continue
        V0=r.get('Vbd_no_impact',np.nan)
        if not np.isfinite(V0):
            mask[j,i]=5;continue
        base[j,i]=V0
        if r.get('stage')=='ok' and np.isfinite(r.get('Vbd',np.nan)):
            full[j,i]=r['Vbd'];mask[j,i]=2
        else: mask[j,i]=3
        if N>1e18:mask[j,i]=4
# screening fit only points with OFF state and within nondegenerate P+ approx
valid=np.isfinite(base)&(NA<=1e18)
aobs=A[valid]; nobs=NA[valid];y=base[valid]
if len(y)<12:raise ValueError(f'Insufficient OFF samples: {len(y)}')
u=(aobs-.5)/.3;v=np.log10(nobs)-17.5
fit=fit_kfold(u,v,y,orders=(1,2,3,4))
# 2D continuous surrogate on convex hull of numerical samples
ag=np.linspace(aa[0],aa[-1],181); ng=np.geomspace(na[0],na[-1],201)
a2,n2=np.meshgrid(ag,ng)
X,_=polynomial_design(((a2-.5)/.3).ravel(),(np.log10(n2)-17.5).ravel(),fit['order'])
pred=(X@fit['coeff']).reshape(a2.shape)
tri=Delaunay(np.column_stack([aobs,np.log10(nobs)]))
inside=(tri.find_simplex(np.column_stack([a2.ravel(),np.log10(n2).ravel()]))>=0).reshape(a2.shape)
pred=np.ma.array(pred,mask=~inside)
ae=np.empty(len(aa)+1);ae[1:-1]=.5*(aa[1:]+aa[:-1]);ae[0]=aa[0]-.5*(aa[1]-aa[0]);ae[-1]=aa[-1]+.5*(aa[-1]-aa[-2])
ne=np.empty(len(na)+1);ne[1:-1]=np.sqrt(na[1:]*na[:-1]);ne[0]=na[0]/np.sqrt(na[1]/na[0]);ne[-1]=na[-1]*np.sqrt(na[-1]/na[-2])
limmin=max(0,float(np.nanmin(base)));limmax=float(np.nanmax(base))
levels=np.arange(np.ceil(limmin/10)*10,np.floor(limmax/10)*10+1,10)
fig,ax=plt.subplots(1,2,figsize=(13,5.6),layout='constrained',sharey=True,sharex=True)
p0=ax[0].pcolormesh(ae,ne,np.ma.masked_invalid(base),vmin=limmin,vmax=limmax,cmap='viridis')
p1=ax[1].contourf(a2,n2,pred,levels=np.linspace(limmin,limmax,24),cmap='viridis',vmin=limmin,vmax=limmax,extend='both')
if len(levels)>1:
    for aa0, xx, yy, zz in [(ax[0],A,NA,np.ma.masked_invalid(base)),(ax[1],a2,n2,pred)]:
        if np.ma.count(zz)>2:
            cs=aa0.contour(xx,yy,zz,levels=levels,colors='white',linewidths=.7)
            aa0.clabel(cs,inline=True,fontsize=8,fmt='%d V')
for k in ax:
    k.set_yscale('log');k.set_xlim(min(aa),max(aa));k.set_ylim(min(na),max(na));k.set_xlabel('P+ pillar thickness a (µm)')
    k.plot(.5,1e17,'*',ms=13,color='#D32F2F',markeredgecolor='white',mew=.8)
    k.grid(alpha=.12)
ax[0].set_ylabel('P+ acceptor doping $N_A$ (cm$^{-3}$)')
# markers: self-consistent impact calculation at nearby subthreshold biases converged
for j in range(len(na)):
    for i in range(len(aa)):
        if mask[j,i]==2:ax[0].plot(aa[i],na[j],'.',color='white',ms=5)
        elif mask[j,i]==1:ax[0].plot(aa[i],na[j],marker='x',color='black',ms=5)
        elif mask[j,i]==3:ax[0].plot(aa[i],na[j],marker='o',mfc='none',mec='white',ms=6)
        elif mask[j,i]==4:ax[0].plot(aa[i],na[j],marker='s',mfc='none',mec='orange',ms=6)
ax[0].set_title('Two-carrier ionization-feedback unity gain\n2D field without avalanche space-charge feedback')
ax[1].set_title(f'Polynomial surrogate (order {fit["order"]})\nheld-out $R^2$ = {fit["test_r2"]:.3f}')
fig.colorbar(p0,ax=ax,shrink=.85,pad=.017,label='Screening avalanche-onset estimate (V)')
fig.suptitle('$L_d=10$ µm, $H=2$ µm, $h=2-2a$, $V_G=-4$ V, $N_D=10^{16}$ cm$^{-3}$\n'
    'Open circles: nonlinear avalanche-source solve not certified; x: channel not OFF',fontsize=12)
fig.savefig(PFX+'_contour_regression.png',dpi=175,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.2,4.8),layout='constrained')
axs[0].scatter(fit['test_obs'],fit['test_pred'],s=55)
lo=min(fit['test_obs'].min(),fit['test_pred'].min());hi=max(fit['test_obs'].max(),fit['test_pred'].max());axs[0].plot([lo,hi],[lo,hi],':',color='gray')
axs[0].set_xlabel('Screening DD+e/h gain onset (V)');axs[0].set_ylabel('Polynomial prediction (V)')
axs[0].set_title(f'Independent holdout • $R^2$={fit["test_r2"]:.4f}\nRMSE={fit["test_rmse"]:.2f} V')
# show all successful avalanche-backreaction full estimates vs base field estimated onset
q=np.isfinite(full)&np.isfinite(base)&(NA<=1e18)
axs[1].scatter(base[q],full[q],s=55)
if np.sum(q)>0:
    lo2=min(base[q].min(),full[q].min());hi2=max(base[q].max(),full[q].max());axs[1].plot([lo2,hi2],[lo2,hi2],':',color='gray')
axs[1].set_xlabel('Feedback screening onset (V)');axs[1].set_ylabel('Self-consistent source: extrapolated onset (V)')
axs[1].set_title(f'Avalanche-source correction • {np.sum(q)} calculated points')
for a0 in axs:a0.grid(alpha=.15)
fig.savefig(PFX+'_validation.png',dpi=175,bbox_inches='tight');plt.close(fig)
co=[{'i':int(i),'j':int(j),'coefficient':float(cc)} for (i,j),cc in zip(fit['terms'],fit['coeff'])]
report=dict(method='two-carrier avalanche feedback=1 on zero-impact-ionization selfconsistent 2D Poisson drift diffusion field; additional fully coupled avalanche subthreshold solves for a subset',
    **{'correction_full_ionization': 'Not complete over the grid: do not call plotted screening values fully self-consistent breakdown voltages'},
    fit_degree=int(fit['order']),fit_R2=float(fit['r2']),test_R2=float(fit['test_r2']),test_RMSE_V=float(fit['test_rmse']),
    points_fit=len(y),points_source_converged=int(np.sum(q)),model_range={'a_um':[float(aobs.min()),float(aobs.max())],'NA_cm3':[float(nobs.min()),float(nobs.max())]},
    coefficients=co,kfold=[{'degree':int(k),'rmse':float(v),'r2':float(r2)} for k,v,r2 in fit['scores']],
    physical_limitations=['No Fermi-Dirac, velocity saturation, SRH recombination or bandgap narrowing',
    'Ideal abrupt rectangular gate tips, no rounding, mesh 0.1um',
    'The ionization feedback operator uses |Jn| and |Jp| and a local field-dependent Van Overstraeten-de Man model',
    'For E outside 1.75e5 to 6.4e5 V/cm, coefficients are extrapolated, calibration is limited',
    'N+ source/drain lateral depth 0.2um each assumed; no empirical validation',
    'Strongly depleted channel screen n_mid/N_D<0.5 at Vg=-4, Vd=0.'])
with open(PFX+'_fit.json','w') as f:json.dump(report,f,indent=2)
with open(PFX+'_data.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['a_um','NA_cm3','h_um','Vbd_feedback_screen_V','Vbd_source_coupled_extrapolated_V','n_mid_gate_off_cm3','stage','validity_code'])
    for (a,na0),r in sorted(raw.items()):
        ix=np.where(aa==a)[0][0];iy=np.where(na==na0)[0][0]
        w.writerow([a,na0,2-2*a,r.get('Vbd_no_impact'),r.get('Vbd'),r.get('nchan'),r['stage'],int(mask[iy,ix])])
model=f'''#!/usr/bin/env python3
"""Polynomial screening surrogate; NOT full self-consistent Vbd.

Trained from two-carrier ionization feedback on zero-II Poisson-DD 2D fields.
Training domain should be restricted to convex hull of reliable OFF samples.
"""
import numpy as np
COEFFICIENTS={repr([float(c) for c in fit['coeff']])}
TERMS={repr(fit['terms'])}

def predict_vbd_screen(a_um, NA_cm3):
    a=np.asarray(a_um,dtype=float)
    NA=np.asarray(NA_cm3,dtype=float)
    if np.any((a<{float(aobs.min())})|(a>{float(aobs.max())})|(NA<{float(nobs.min())})|(NA>{float(nobs.max())})):
        raise ValueError('Outside training bounding range')
    u=(a-.5)/.3
    v=np.log10(NA)-17.5
    return sum(c*u**i*v**j for c,(i,j) in zip(COEFFICIENTS,TERMS))

if __name__=='__main__':
    print(predict_vbd_screen(0.5,1e17))
'''
with open(PFX+'_predict.py','w') as f:f.write(model)
print('RESULTS',len(rows),'eligible screening',len(y),'full-coupled converged',np.sum(q))
print('fit',fit['order'],'R2',fit['r2'],'holdout',fit['test_r2'],'RMSE',fit['test_rmse'])
print('plot',PFX+'_contour_regression.png')
