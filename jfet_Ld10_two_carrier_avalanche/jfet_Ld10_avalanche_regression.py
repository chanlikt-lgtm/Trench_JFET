#!/usr/bin/env python3
"""Fit and plot preliminary 2D JFET electron+hole avalanche-feedback onset vs a, NA.

Input JSONL produced by jfet_vbd_sweep_full.py. Uses only points with
self-consistent subthreshold avalanche results, a depleted channel at VG=-4,
and a reasonably short gain-to-unity extrapolation. Outputs:
- observed estimated Vbd contour and regression contour
- scatter validation and parity plot
- model coefficients and predict.py
- full data CSV and JSON report
"""
import os,json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

ROOT=os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.join(ROOT,'jfet_vbd_Ld10_fullavalanche_results.jsonl')

def polynomial_design(u,v,order):
    terms=[(i,d-i) for d in range(order+1) for i in range(d,-1,-1)]
    X=np.column_stack([u**i*v**j for i,j in terms])
    return X,terms


def fit_kfold(u,v,y,orders=(1,2,3,4),random_seed=42):
    rng=np.random.default_rng(random_seed)
    N=len(y); idx=np.arange(N);rng.shuffle(idx)
    folds=np.array_split(idx,5)
    scores=[]
    for degree in orders:
        A,terms=polynomial_design(u,v,degree)
        if N < (len(terms)+5):continue
        predicted=np.full(N,np.nan)
        for test in folds:
            train=np.setdiff1d(idx,test)
            if len(train)<=len(terms):break
            coeff=np.linalg.lstsq(A[train],y[train],rcond=None)[0]
            predicted[test]=A[test]@coeff
        if np.all(np.isfinite(predicted)):
            rmse=np.sqrt(np.mean((y-predicted)**2))
            r2=1.0-np.sum((y-predicted)**2)/np.sum((y-y.mean())**2)
            scores.append((degree,rmse,r2))
    if not scores: raise ValueError('Too few physically eligible samples to fit a polynomial')
    degree=min(scores,key=lambda r:r[1])[0]
    A,terms=polynomial_design(u,v,degree)
    coeff=np.linalg.lstsq(A,y,rcond=None)[0]
    pred=A@coeff
    # Independent 25% holdout (not selected based on its test performance).
    count=max(3,int(round(N*.25)))
    test=idx[:count];train=idx[count:]
    ct=np.linalg.lstsq(A[train],y[train],rcond=None)[0]
    yhat=A[test]@ct
    r2hold=1-np.sum((y[test]-yhat)**2)/np.sum((y[test]-y[test].mean())**2) if count>1 and np.sum((y[test]-y[test].mean())**2)>0 else np.nan
    rms= np.sqrt(np.mean((y-pred)**2))
    return dict(order=degree,coeff=coeff,terms=terms,pred=pred,rmse=rms,
       train_n=len(train),test_n=len(test),r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2),
       test_r2=r2hold,test_rmse=np.sqrt(np.mean((y[test]-yhat)**2)),
       test_max_error=np.max(np.abs(y[test]-yhat)),scores=scores,test_obs=y[test],test_pred=yhat)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default=INPUT);p.add_argument('--prefix',default=os.path.join(ROOT,'jfet_Ld10_full_ionization'))
    args=p.parse_args()
    rows=[]
    with open(args.input) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if len(rows)<10:raise ValueError('Not enough parametric calculations available yet')
    aa=np.array(sorted({r['a'] for r in rows}),dtype=float)
    na=np.array(sorted({r['NA'] for r in rows}),dtype=float)
    lookup={(r['a'],r['NA']):r for r in rows}
    raw=np.full((len(na),len(aa)),np.nan)
    qual=np.zeros_like(raw,dtype=int)  # 0 missing, 1 not-off, 2 long extrap, 3 numerical fail, 4 reliable, 5 high-doping caveat
    for j,nb in enumerate(na):
        for i,a in enumerate(aa):
            r=lookup.get((a,nb))
            if r is None: continue
            if r.get('stage')!='ok':
                qual[j,i]=3;continue
            V=r.get('Vbd',np.nan)
            if not np.isfinite(V): qual[j,i]=3;continue
            ncenter=r.get('nchan',np.inf)
            if ncenter>0.5e16:qual[j,i]=1;continue
            raw[j,i]=V
            probes=r.get('gain_probes',[])
            if len(probes)<2:
                qual[j,i]=2;continue
            gap=V-probes[-1][0]
            E=probes[-1][2]
            # Only well-depleted, grid-screened, within data calibration ranges
            if gap>15.0 or gap< -3.0 or E>6.4e5:
                qual[j,i]=2
            elif nb>1.0e18:
                qual[j,i]=5 # P+ degenerate - classical Boltzmann questionable
            else:
                qual[j,i]=4
    valid=qual==4
    xi,xj=np.meshgrid(aa,na)
    avals=xi[valid];nvals=xj[valid];V=raw[valid]
    # If only few reliable, broaden fit to all OFF examples with short extrap and high-doping caveat.
    fallback=False
    if len(V)<14:
        valid=(qual==4)|(qual==5)
        avals=xi[valid];nvals=xj[valid];V=raw[valid]
        fallback=True
    if len(V)<12:
        raise ValueError(f'Only {len(V)} valid off-state points, regression would be underdetermined')
    u=(avals-0.5)/0.3
    vv=np.log10(nvals)-17.5
    fit=fit_kfold(u,vv,V)
    order=fit['order']
    terms=fit['terms'];c=fit['coeff']
    # Predict only within swept domain; separate from physically calculated data.
    ag=np.linspace(aa[0],aa[-1],181)
    nag=np.geomspace(na[0],na[-1],201)
    ggA,ggN=np.meshgrid(ag,nag)
    xx,_=polynomial_design(((ggA-.5)/.3).ravel(),(np.log10(ggN)-17.5).ravel(),order)
    predicted=(xx@c).reshape(ggA.shape)
    # Mask outside convex hull of reliable (a, logNA) samples
    from scipy.spatial import Delaunay
    pts=np.column_stack([avals,np.log10(nvals)])
    tri=Delaunay(pts)
    inside=tri.find_simplex(np.column_stack([ggA.ravel(),np.log10(ggN).ravel()]))>=0
    predict_masked=np.ma.array(predicted,mask=~inside.reshape(ggA.shape))
    pmin=max(0,float(np.nanmin(raw[np.isfinite(raw)])))
    pmax=float(np.nanmax(raw[np.isfinite(raw)]))
    # plottable raw values off state; flags annotate confidence
    masked=np.ma.array(raw,mask=~np.isfinite(raw))
    ae=np.empty(len(aa)+1);ae[1:-1]=(aa[:-1]+aa[1:])/2;ae[0]=aa[0]-(aa[1]-aa[0])/2;ae[-1]=aa[-1]+(aa[-1]-aa[-2])/2
    ne=np.empty(len(na)+1);ne[1:-1]=np.sqrt(na[:-1]*na[1:]);ne[0]=na[0]/np.sqrt(na[1]/na[0]);ne[-1]=na[-1]*np.sqrt(na[-1]/na[-2])
    fig,axes=plt.subplots(1,2,figsize=(13.6,5.3),sharex=True,sharey=True,layout='constrained')
    norm=Normalize(pmin,pmax)
    cb=axes[0].pcolormesh(ae,ne,masked,cmap='viridis',norm=norm,shading='flat')
    # reliability overlay: little open circles for extrapolation/degenerate data
    for j in range(len(na)):
        for i in range(len(aa)):
            q=qual[j,i]
            if q==1:axes[0].plot(aa[i],na[j],marker='x',ms=5,color='black')
            elif q==2:axes[0].plot(aa[i],na[j],marker='o',ms=6,mfc='none',mec='#C73938',mew=1.2)
            elif q==3:axes[0].plot(aa[i],na[j],marker='s',ms=5,mfc='none',mec='#8B22A4',mew=1.)
            elif q==5:axes[0].plot(aa[i],na[j],marker='D',ms=5,mfc='none',mec='#E18E00',mew=1.)
    levels=np.arange(np.ceil(pmin/10)*10,np.floor(pmax/10)*10+1,10)
    if len(levels)>1:
        try:
            lines=axes[0].contour(xi,xj,masked,levels=levels,colors='white',linewidths=.65)
            axes[0].clabel(lines,fontsize=7,inline=True,fmt='%d V')
        except ValueError:pass
    cf=axes[1].contourf(ggA,ggN,predict_masked,levels=np.linspace(pmin,pmax,26),cmap='viridis',norm=norm,extend='both')
    if len(levels)>1:
        lines=axes[1].contour(ggA,ggN,predict_masked,levels=levels,colors='white',linewidths=.75)
        axes[1].clabel(lines,fontsize=7,inline=True,fmt='%d V')
    for ax in axes:
        ax.set_yscale('log');ax.set_xlim(min(aa),max(aa));ax.set_ylim(min(na),max(na))
        ax.set_xlabel('P+ pillar thickness a (µm)')
        ax.plot(.5,1e17,marker='*',ms=12,color='red',markeredgecolor='white',markeredgewidth=.7)
        ax.grid(alpha=.13)
    axes[0].set_ylabel('P+ doping $N_A$ (cm$^{-3}$)')
    axes[0].set_title('Computed subthreshold ionization-feedback onset\n(missing/invalid results not filled)')
    axes[1].set_title(f'Polynomial regression, degree {order}\nheld-out R² = {fit["test_r2"]:.3f}')
    fig.colorbar(cb,ax=axes,pad=.015,shrink=.93,label='Estimated $V_{BD}$ (V)')
    fig.suptitle('$L_d=10$ µm · $H=2$ µm · $h=2-2a$ · $V_G=-4$ V · 300 K\n'
                 '2D DD + e⁻/h⁺ impact source at subthreshold biases; feedback unity extrapolated',fontsize=12)
    plot_path=args.prefix+'_contour_vs_regression.png'
    fig.savefig(plot_path,dpi=170,bbox_inches='tight');plt.close(fig)
    # validation parity plot
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    ax.scatter(fit['test_obs'],fit['test_pred'],s=40,alpha=.85)
    lo=min(fit['test_obs'].min(),fit['test_pred'].min());hi=max(fit['test_obs'].max(),fit['test_pred'].max())
    ax.plot([lo,hi],[lo,hi],'--',color='gray',linewidth=1)
    ax.set_xlabel('Calculated avalanche-onset estimate (V)');ax.set_ylabel('Polynomial held-out prediction (V)')
    ax.set_title(f'Holdout validation • {fit["test_n"]} points\nR²={fit["test_r2"]:.4f}  RMSE={fit["test_rmse"]:.2f} V')
    ax.grid(alpha=.25)
    fig.tight_layout()
    validation_path=args.prefix+'_validation.png'
    fig.savefig(validation_path,dpi=170);plt.close(fig)
    # prediction artifact
    def coeff_json():
        return [{"i":i,"j":j,"coefficient":float(ci)} for (i,j),ci in zip(terms,c)]
    results=dict(description='exploratory 2D DD impact-ionization-feedback onset, NOT certified physical breakdown',
        model='selfconsistent electron+hole ionization source in converged subthreshold states; gain to unity extrapolated',
        n_inputs=len(rows),n_valid_fit=int(len(V)),count_by_status={str(k):int((qual==k).sum()) for k in range(6)},
        normalizations={'u':'(a_um-0.5)/0.3','v':'log10(NA_cm3)-17.5'},
        degree=int(order),coefficients=coeff_json(),r2_fit=float(fit['r2']),
        rmse_fit_V=float(fit['rmse']),r2_holdout=float(fit['test_r2']),
        rmse_holdout_V=float(fit['test_rmse']),max_holdout_error_V=float(fit['test_max_error']),
        kfold_scores=[{'degree':int(deg),'rmse_V':float(e),'r2':float(r)} for deg,e,r in fit['scores']],
        notes=['Fixed 2um total silicon height. N-channel h=2-2a.',
         'Constant low-field mobility; no SRH, velocity saturation, bandgap narrowing, quantum effects or avalanche transient.',
         'Finite rectangular corners unresolved; mesh is 0.1um x 0.1um.',
         'N+ source/drain widths 0.2um each, assumed from prior illustrative model.',
         'Screened reliable points: n_center/N_D<=0.5, extrapolation length<=15V, E_last<=640kV/cm, NA<=1e18cm^-3.',
         'No experimental data; R2 quantifies only agreement with the numerical model.',
         f'Fit domain convex hull of {len(V)} filtered sample locations.',
         f'Screen fallback to high-doping samples used: {fallback}.'])
    with open(args.prefix+'_fit.json','w') as f:json.dump(results,f,indent=2)
    with open(args.prefix+'_grid.csv','w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['a_um','NA_cm3','h_um','Vbd_est_V','Vbd_noII_V','off_center_n_cm3','gain_probe_last','extrapolation_V','E_at_last_probe_V_cm','status','quality_code'])
        for (a,na_val),r in sorted(lookup.items()):
            probes=r.get('gain_probes',[]);vp=probes[-1][0] if probes else np.nan
            wr.writerow([a,na_val,2-2*a,r.get('Vbd'),r.get('Vbd_no_impact'),r.get('nchan'),
                 probes[-1][1] if probes else '',r.get('Vbd',np.nan)-vp if probes else '',probes[-1][2] if probes else '',r['stage'],int(qual[np.where(na==na_val)[0][0],np.where(aa==a)[0][0]])])
    predictor=f'''#!/usr/bin/env python3
"""Surrogate regression for exploratory JFET Ld=10um e/h avalanche-onset model.
Validity restricted to computed reliable region. NOT measured VBD.
"""
import numpy as np
COEFF={repr([float(x) for x in c])}
TERMS={repr(terms)}
A_MIN={float(avals.min())}
A_MAX={float(avals.max())}
NA_MIN={float(nvals.min())}
NA_MAX={float(nvals.max())}

def predict_vbd(a_um, NA_cm3):
    a_um=np.asarray(a_um, dtype=float)
    NA_cm3=np.asarray(NA_cm3, dtype=float)
    if np.any((a_um<A_MIN)|(a_um>A_MAX)|(NA_cm3<NA_MIN)|(NA_cm3>NA_MAX)):
        raise ValueError('Outside reliable fitted bounding range')
    u=(a_um-0.5)/0.3
    v=np.log10(NA_cm3)-17.5
    return sum(cc*u**ii*v**jj for cc,(ii,jj) in zip(COEFF,TERMS))

if __name__=='__main__':
    print('Example:',predict_vbd(0.5,1e17))
'''
    with open(args.prefix+'_predict.py','w') as f:f.write(predictor)
    print('PLOT',plot_path,flush=True)
    print('VALIDATION',validation_path,flush=True)
    print('R2',fit['r2'],'holdout R2',fit['test_r2'],'nvalid',len(V),'degree',order,flush=True)
    print('statuses',results['count_by_status'],flush=True)
    return results

if __name__=='__main__':main()
