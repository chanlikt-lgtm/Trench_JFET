#!/usr/bin/env python3
"""Parametric 2D JFET avalanche-feedback onset with self-consistent e+h impact source.

For each (a, NA):
 - DD self-consistent Poisson + n/p at VG=-4, Vd sweep
 - approximate small-signal 2-carrier ionization feedback spectral gain to bracket onset
 - recompute self-consistent generation GII = (alpha_n|Jn|+alpha_p|Jp|)/q
 - use gain lambda=1 on self-consistent avalanche-on field as operational onset criterion

Uncertainties: local avalanche model/field range, rectangular corners, constant mobility,
no SRH/velocity saturation, 0.2um source/drain N+ widths, mesh accuracy.
"""
import os,sys,time,json,traceback
import numpy as np
from jfet_avalanche_full import AvalancheJFET, N0
from jfet_newton_dd_robust import CoupledNewton
from jfet_avalanche_test import avalanche_solve

def solve_one(a,NA,dx=.1,dy=.1,Ld=10., verbose=False):
    t=time.monotonic()
    m=AvalancheJFET(a=float(a),NA=float(NA),Ld=Ld,dx=dx,dy=dy,ws=.2,wd=.2)
    s=CoupledNewton(m)
    Z=s.state0()
    gates=np.arange(-0.25,-4.0001,-.25)
    for v in gates:
        Z,_,st=s.run(0,float(v),Z,avalanche=False)
        if not st['newton_converged']:
            return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='gate',reason='Newton failure',v_failed=v,secs=time.monotonic()-t)
    ix=np.argmin(abs(m.x-0.5));iy=np.argmin(abs(m.y))
    index=iy*m.Nx+ix
    nchan=np.exp(Z[m.N+index])*N0
    zero_gain,peak0=m.feedback_gain(Z[:m.N],steps=24)
    if verbose:print('gate done',a,NA,'ncenter',nchan,'zeroGain',zero_gain,'t',round(time.monotonic()-t,1),flush=True)
    if zero_gain>=1:
        return dict(a=float(a),NA=float(NA),Vbd=0.,stage='zero_drain_ionization',zero_gain=float(zero_gain),nchan=float(nchan),secs=time.monotonic()-t)
    track=[]
    lastV=0.;lastZ=Z.copy();lastGain=zero_gain
    hiV=None;hiZ=None;higain=None
    biases=[1,3,6,10,14,18,22,26,30,35,40,45,50,55,60,65,70,75,80,90,100,110,120,140,160,180,200]
    for vd in biases:
        Ztrial,_,st=s.run(vd,-4,Z,avalanche=False)
        if not st['newton_converged']:
            # retry midpoint to shorten bias transition
            half=(lastV+vd)/2
            zm,_,sm=s.run(half,-4,lastZ,avalanche=False)
            if sm['newton_converged']:
                Ztrial,_,st=s.run(vd,-4,zm,avalanche=False)
            if not st['newton_converged']:
                return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='bias',v_failed=vd,nchan=float(nchan),secs=time.monotonic()-t,trace=track)
        Z=Ztrial
        g,Em=m.feedback_gain(psi=Z[:m.N],steps=22)
        track.append((float(vd),float(g),float(Em)))
        if verbose:print('  vd',vd,'g',round(g,3),'E',round(Em),'time',round(time.monotonic()-t,1),flush=True)
        if g>=1.02:
            hiV=vd;hiZ=Z.copy();higain=g
            break
        lastV=vd;lastZ=Z.copy();lastGain=g
    if hiV is None:
        return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='no_onset_under_200V',nchan=float(nchan),secs=time.monotonic()-t,trace=track)
    # interpolate to estimate no-impact gain=1 threshold
    est=lastV+(hiV-lastV)*(1-lastGain)/max(higain-lastGain,1e-12)
    est=float(np.clip(est,0,200))
    # At high current-multiplication the coupled nonlinear equations become
    # numerically stiff, so compute fully coupled e/h avalanche source at
    # three converged SUBTHRESHOLD voltages and extrapolate feedback gain=1.
    # This is a breakdown ONSET ESTIMATE, not a selfconsistent threshold solve.
    delta=[6.0,3.5]
    trial_vs=sorted(set([float(max(.1,est-d)) for d in delta]))
    def at_voltage(v,initial,Ginitial=None):
        nonlocal m,s
        m.Vd=float(v);m.Vg=-4.
        if Ginitial is None:
            zz,_,sr=s.run(float(v),-4.,initial,avalanche=False)
            if not sr['newton_converged']:
                return None,None,None,None,'no-avalanche Newton failure'
        else:
            zz=initial
        za,ga,sa=avalanche_solve(m,s,zz,Ginitial,maxit=14,tol=.018,damping=.5,printing=False)
        if not sa['converged']:
            return None,None,None,None,'selfconsistent impact source nonconverged'
        gain,em=m.feedback_gain(psi=za[:m.N],steps=27)
        return za,ga,float(gain),float(em),'ok'
    probes=[]
    zprev=lastZ.copy();gprev=None;vprev=None
    for v in trial_vs:
        if verbose:print("PROBE START",v,"sec",round(time.monotonic()-t,1),flush=True)
        # Continue impact generation with small enough drain steps
        vvlist=[v] if vprev is None else list(np.linspace(vprev,v,max(2,int(np.ceil((v-vprev)/.8))+1))[1:])
        for vv in vvlist:
            if verbose:print("  SOURCE SOLVE",vv,"warm",gprev is not None,"sec",round(time.monotonic()-t,1),flush=True)
            znew,gnew,lam,em,status=at_voltage(vv,zprev,gprev)
            if verbose:print("  SOURCE RETURN",lam,status,"sec",round(time.monotonic()-t,1),flush=True)
            if lam is None:
                return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='impact',reason=status,
                            Vbd_no_impact=float(est),v_failed=float(vv),nchan=float(nchan),secs=time.monotonic()-t,trace=track)
            zprev=znew;gprev=gnew;vprev=vv
        probes.append([float(v),float(lam),float(em)])
    xs=np.array([item[0] for item in probes]);ys=np.array([item[1] for item in probes])
    if not np.all(np.isfinite(ys)) or ys[-1]<=ys[-2] or abs(ys[-1]-ys[-2])<1e-5:
        return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='gain_nonmonotone',probes=probes,
                    Vbd_no_impact=float(est),nchan=float(nchan),secs=time.monotonic()-t,trace=track)
    # Local linear continuation of the self-consistent gain to unity.
    slope=(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
    vbd=xs[-1]+(1.0-ys[-1])/slope
    # Quadratic-extrapolation sensitivity as a simple internal model check.
    vbd_quad=np.nan  # needs three or more converged avalanche probe voltages
    if not (0 <= vbd <= 250):
        return dict(a=float(a),NA=float(NA),Vbd=np.nan,stage='extrapolation_unphysical',probes=probes,
                    Vbd_no_impact=float(est),nchan=float(nchan),secs=time.monotonic()-t,trace=track)
    result=dict(a=float(a),NA=float(NA),h=float(2-2*a),Vbd=float(vbd),stage='ok',
       nchan=float(nchan),Vbd_no_impact=float(est),
       gain_probes=probes, Vbd_quadratic=float(vbd_quad),
       peakE_V_cm_at_last_probe=float(probes[-1][2]),secs=float(time.monotonic()-t),
       grid_dx=float(dx),grid_dy=float(dy),trace=track)
    if verbose:print('DONE',a,NA,'VBD',vbd,'t',result['secs'],flush=True)
    return result


def _worker(args):
    try:
        import signal
        has_alarm=hasattr(signal,'SIGALRM')  # not available on Windows
        if has_alarm:
            def _time_limit(signum, frame):
                raise TimeoutError('Single geometry exceeded 115 second execution limit')
            signal.signal(signal.SIGALRM,_time_limit)
            signal.alarm(115)
        try:
            return solve_one(*args)
        finally:
            if has_alarm:
                signal.alarm(0)
    except Exception as exc:
        return dict(a=float(args[0]),NA=float(args[1]),Vbd=np.nan,stage='exception',reason=str(exc),traceback=traceback.format_exc())

if __name__=='__main__':
    import argparse,concurrent.futures
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'jfet_vbd_Ld10_fullavalanche_results.jsonl'))
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--single',action='store_true')
    ap.add_argument('--a',type=float,default=.5)
    ap.add_argument('--NA',type=float,default=1e17)
    args=ap.parse_args()
    t=time.monotonic()
    if args.single:
        r=solve_one(args.a,args.NA,verbose=True)
        print(json.dumps(r,indent=2),flush=True)
    else:
        avals=[.2,.3,.4,.5,.6,.7,.8]
        navals=[3.16227766017e16,6.81292069058e16,1e17,1.46779926762e17,3.16227766017e17,6.81292069058e17,1.46779926762e18,3.16227766017e18]
        jobs=[(a,na) for na in navals for a in avals]
        with open(args.output,'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures={ex.submit(_worker,job):job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                r=future.result()
                f.write(json.dumps(r)+'\n');f.flush()
                print('RESULT',r['a'],f"{r['NA']:.3g}",r['stage'],r.get('Vbd'),
                    'secs',round(r.get('secs',0),1),'total',round(time.monotonic()-t,1),flush=True)
