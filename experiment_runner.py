#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reading Experiment — v5e (semi-bold strong with logging)
- <strong>…</strong> rendu en "demi-gras" si une variante Medium/DemiBold est installée.
- Journalise la famille sélectionnée dans la console et dans data/<session>/font_report.txt.
"""

import os, sys, csv, json, time, math, glob, threading, subprocess, platform, wave
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.font as tkfont

# Optional deps
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
try:
    from PIL import ImageGrab, ImageStat
    HAS_PIL = True
except Exception:
    HAS_PIL = False
try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

def now_iso(): return datetime.now().isoformat(timespec='seconds')
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _detect_openface_root():
    candidates=[os.getenv("OPENFACE_ROOT",""),"./openface",
                os.path.expanduser("~/Desktop/external_libs/openFace"),
                os.path.expanduser("~/Desktop/external_libs/OpenFace")]
    for c in candidates:
        if not c: continue
        binp=os.path.join(os.path.abspath(c),"build","bin","FeatureExtraction")
        if os.path.isfile(binp): return os.path.abspath(c)
    return os.path.abspath(os.getenv("OPENFACE_ROOT","./openface"))

OPENFACE_ROOT=_detect_openface_root()
OPENFACE_BIN=os.path.join(OPENFACE_ROOT,"build","bin","FeatureExtraction")
OF_DEVICE=str(os.getenv("OPENFACE_DEVICE","0"))
DATA_ROOT=os.path.abspath(os.getenv("DATA_ROOT","./data"))
MATERIALS_DIR=os.path.abspath(os.getenv("MATERIALS_DIR","./materials"))
PLAN_CSV=os.path.abspath(os.getenv("PLAN_CSV","./plan.csv"))

TEXT_W_PCT=float(os.getenv("TEXT_W_PCT","0.70"))
TEXT_H_PCT=float(os.getenv("TEXT_H_PCT","0.70"))
BG_COLOR="#FFFFFF"; FG_COLOR="#111111"
FONT_FAMILY=os.getenv("FONT_FAMILY","Avenir Next"); BASE_FONT_SIZE=int(os.getenv("BASE_FONT_SIZE","24"))
FULLSCREEN=True; BEEP_LAST_SECONDS=5
TELEMETRY_HZ=float(os.getenv("TELEMETRY_HZ","1.0"))
AMBIENT_CAMERA_INDEX=int(os.getenv("AMBIENT_CAMERA_INDEX","0"))
AMBIENT_PIPE=os.getenv("AMBIENT_PIPE","").strip()

SMOOTH_WINDOW=5
SACCADE_VEL_MIN=35.0
FIX_MIN_MS=100

# ---------- Demi-gras helper ----------
SEMIBOLD_CANDIDATES = [
    "Avenir Next Medium",
    "Avenir Next Demi Bold",
    "Avenir Next DemiBold",
    "AvenirNext-Medium",
    "AvenirNext-DemiBold",
    "Avenir Next-Medium",
    # alternatives fréquentes macOS
    "Avenir Medium",
    "Avenir Demi Bold",
    "Avenir-DemiBold",
    # fallbacks génériques si Avenir Next indisponible
    "Helvetica Neue Medium",
    "SF Pro Text Semibold",
    "Arial Rounded MT Bold",  # visuellement un intermédiaire parfois acceptable
]

def probe_semibold(size, fallback_family="Avenir Next"):
    env_name = os.getenv("SEMIBOLD_FAMILY", "").strip()
    try:
        families = set(tkfont.families())
    except Exception:
        families = set()
    picked_name = None
    if env_name and env_name in families:
        picked_name = env_name
        f = tkfont.Font(family=env_name, size=size)
        return f, picked_name
    for name in SEMIBOLD_CANDIDATES:
        if name in families:
            picked_name = name
            f = tkfont.Font(family=name, size=size)
            return f, picked_name
    # fallback : gras adouci
    try:
        f = tkfont.Font(family=fallback_family, size=max(8, size-1), weight="bold")
        picked_name = f"{fallback_family} (bold -1pt fallback)"
        return f, picked_name
    except tk.TclError:
        f = tkfont.Font(size=size, weight="bold")
        picked_name = "system-bold (fallback)"
        return f, picked_name

# ------------------ Data structures ------------------
from dataclasses import dataclass
@dataclass
class TrialDef:
    trial_id: str
    condition: str
    text_path: str
    qcm_path: str
    likert_path: str

def read_plan(path:str):
    trials=[]
    if not os.path.isfile(path): raise FileNotFoundError(f"Plan file not found: {path}")
    with open(path,newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        for row in rdr:
            trials.append(TrialDef(
                trial_id=row['trial_id'].strip(),
                condition=row['condition'].strip().lower(),
                text_path=os.path.abspath(row['text_path'].strip()),
                qcm_path=os.path.abspath(row['qcm_path'].strip()),
                likert_path=os.path.abspath(row.get('likert_path','').strip() or os.path.join(MATERIALS_DIR,'likert.json')),
            ))
    return trials

class MarkerLogger:
    def __init__(self, sess_dir:str):
        self.path=os.path.join(sess_dir,"markers.csv")
        with open(self.path,'w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow(["ts_iso","unix","of_t","label","details_json"])
        self.of_start_unix=None
    def set_of_start(self, unix:float): self.of_start_unix=float(unix)
    def mark(self,label:str,details:dict|None=None):
        unix=time.time(); of_t=(unix-self.of_start_unix) if self.of_start_unix is not None else ""
        with open(self.path,'a',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow([now_iso(), f"{unix:.3f}", f"{of_t:.3f}" if of_t!="" else "", label, json.dumps(details or {},ensure_ascii=False)])

class OpenFaceController:
    def __init__(self,out_dir:str):
        self.out_dir=out_dir; self.proc=None; self.of_start_unix=None
    def start_once(self):
        ensure_dir(self.out_dir)
        if not os.path.isfile(OPENFACE_BIN): raise FileNotFoundError(f"OpenFace binary not found: {OPENFACE_BIN}")
        args=[OPENFACE_BIN,'-device',str(OF_DEVICE),'-out_dir',self.out_dir,'-q','-pose','-gaze','-aus']
        self.of_start_unix=time.time()
        self.proc=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        print(f"[OpenFace] session started (PID={self.proc.pid})")
    def stop(self):
        if self.proc:
            try: self.proc.terminate(); self.proc.wait(timeout=5)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            print("[OpenFace] session stopped")
        self.proc=None
    def is_running(self): return self.proc is not None and (self.proc.poll() is None)
    def latest_csv(self):
        files=sorted(glob.glob(os.path.join(self.out_dir,'*.csv')))
        return files[-1] if files else None

class TelemetrySampler:
    def __init__(self,app,sess_dir:str):
        self.app=app; self.sess_dir=sess_dir
        self.stop_evt=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.csv_path=os.path.join(sess_dir,'telemetry.csv')
        self.ambient_cap=None; self._init_ambient()
    def _init_ambient(self):
        if not HAS_CV2: return
        try:
            cap=cv2.VideoCapture(AMBIENT_PIPE if AMBIENT_PIPE else AMBIENT_CAMERA_INDEX)
            if cap is not None and cap.isOpened():
                self.ambient_cap=cap
                print(f"[Ambient] Using {'pipe' if AMBIENT_PIPE else 'camera'}: {AMBIENT_PIPE or AMBIENT_CAMERA_INDEX}")
            else:
                if cap is not None: cap.release()
        except Exception as e:
            print("[Ambient] init failed:", e)
    def start(self):
        with open(self.csv_path,'w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow(['ts_iso','screen_lum','ambient_lum','pose_Rx','pose_Ry','pose_Rz'])
        self.thread.start()
    def stop(self):
        self.stop_evt.set(); self.thread.join(timeout=3)
        if self.ambient_cap:
            try: self.ambient_cap.release()
            except Exception: pass
    def _grab_screen_lum(self):
        if not HAS_PIL: return None
        try:
            bbox=self.app.window_bbox()
            if not bbox: return None
            x1,y1,x2,y2=bbox; cx=(x1+x2)//2; cy=(y1+y2)//2; r=100
            img=ImageGrab.grab(bbox=(cx-r,cy-r,cx+r,cy+r))
            return float(ImageStat.Stat(img.convert('L')).mean[0])
        except Exception: return None
    def _ambient_lum(self):
        if not HAS_CV2 or self.ambient_cap is None: return None
        ret,frame=self.ambient_cap.read()
        if not ret: return None
        import cv2 as _cv2
        gray=_cv2.cvtColor(frame,_cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    def _read_pose(self):
        csvp=self.app.ofc.latest_csv() if self.app.ofc else None
        if not csvp: return (None,None,None)
        try:
            with open(csvp,'r',encoding='utf-8') as f:
                header=f.readline().strip().split(','); col={n:i for i,n in enumerate(header)}
                if not {'pose_Rx','pose_Ry','pose_Rz'}.issubset(col): return (None,None,None)
                last=None
                for line in f: last=line
                if not last: return (None,None,None)
                p=last.strip().split(',')
                return (float(p[col['pose_Rx']]),float(p[col['pose_Ry']]),float(p[col['pose_Rz']]))
        except Exception: return (None,None,None)
    def _run(self):
        period=1.0/max(1e-6,TELEMETRY_HZ)
        while not self.stop_evt.is_set():
            ts=now_iso(); screen=self._grab_screen_lum(); amb=self._ambient_lum(); Rx,Ry,Rz=self._read_pose()
            with open(self.csv_path,'a',newline='',encoding='utf-8') as f:
                w=csv.writer(f)
                w.writerow([ts, f"{screen:.2f}" if screen is not None else '',
                               f"{amb:.2f}" if amb is not None else '',
                               f"{Rx:.6f}" if Rx is not None else '',
                               f"{Ry:.6f}" if Ry is not None else '',
                               f"{Rz:.6f}" if Rz is not None else ''])
            time.sleep(period)

def moving_average(xs,w:int):
    if w<=1: return xs[:]
    out=[]; s=0.0; q=[]
    for x in xs:
        q.append(x); s+=x
        if len(q)>w: s-=q.pop(0)
        out.append(s/len(q))
    return out

def invert_3x3(m):
    a=[row[:] for row in m]; inv=[[1,0,0],[0,1,0],[0,0,1]]
    for i in range(3):
        piv=a[i][i]
        if abs(piv)<1e-12: return None
        f=1.0/piv
        for j in range(3):
            a[i][j]*=f; inv[i][j]*=f
        for k in range(3):
            if k==i: continue
            g=a[k][i]
            for j in range(3):
                a[k][j]-=g*a[i][j]
                inv[k][j]-=g*inv[i][j]
    return inv

def lstsq_2d(gx,gy,sx,sy):
    n=len(gx)
    if n<3: return None
    Sxx=sum(g*g for g in gx); Syy=sum(g*g for g in gy); Sxy=sum(gx[i]*gy[i] for i in range(n))
    Sx=sum(gx); Sy=sum(gy); N=n
    A=[[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,N]]
    bx=[sum(gx[i]*sx[i] for i in range(n)),
        sum(gy[i]*sx[i] for i in range(n)),
        sum(sx)]
    by=[sum(gx[i]*sy[i] for i in range(n)),
        sum(gy[i]*sy[i] for i in range(n)),
        sum(sy)]
    invA=invert_3x3(A)
    if invA is None: return None
    cx=[sum(invA[i][j]*bx[j] for j in range(3)) for i in range(3)]
    cy=[sum(invA[i][j]*by[j] for j in range(3)) for i in range(3)]
    def r2(coeffs,t,u,y):
        yhat=[coeffs[0]*t[i]+coeffs[1]*u[i]+coeffs[2] for i in range(len(y))]
        ym=sum(y)/len(y) if y else 0.0
        ss_tot=sum((yy-ym)**2 for yy in y)
        ss_res=sum((y[i]-yhat[i])**2 for i in range(len(y)))
        return 1.0-(ss_res/(ss_tot if ss_tot>1e-12 else 1.0)), yhat
    R2x,_=r2(cx,gx,gy,sx); R2y,_=r2(cy,gx,gy,sy)
    return {'cx':cx,'cy':cy,'R2x':R2x,'R2y':R2y}

def parse_openface_csvs(sess_dir:str):
    return sorted(glob.glob(os.path.join(sess_dir,'of_session','*.csv')))

def analyze_csv(of_csv:str)->dict:
    with open(of_csv,'r',encoding='utf-8') as f:
        header=f.readline().strip().split(','); col={n:i for i,n in enumerate(header)}
        need=['timestamp','confidence','success','gaze_angle_x','gaze_angle_y']
        for n in need:
            if n not in col: raise ValueError(f"Missing column in {os.path.basename(of_csv)}: {n}")
        ts=[]; gax=[]; gay=[]; Rx=[]; Ry=[]; Rz=[]; au45=[]
        for line in f:
            p=line.rstrip('\n').split(',')
            try:
                t=float(p[col['timestamp']]); c=float(p[col['confidence']]); s=int(p[col['success']])
                if s==1 and c>=0.8:
                    ts.append(t); gax.append(float(p[col['gaze_angle_x']])); gay.append(float(p[col['gaze_angle_y']]))
                    if 'pose_Rx' in col: Rx.append(float(p[col['pose_Rx']])); Ry.append(float(p[col['pose_Ry']])); Rz.append(float(p[col['pose_Rz']]))
                    if 'AU45_c' in col: au45.append(int(p[col['AU45_c']]))
            except Exception: pass
    dur=ts[-1]-ts[0] if ts else 0.0
    gax_s=moving_average(gax,SMOOTH_WINDOW); gay_s=moving_average(gay,SMOOTH_WINDOW)
    v=[0.0]
    for i in range(1,len(ts)):
        dt=max(1e-6,ts[i]-ts[i-1]); dx=gax_s[i]-gax_s[i-1]; dy=gay_s[i]-gay_s[i-1]
        speed=math.degrees(math.atan2(math.hypot(dx,dy),1.0))/dt
        v.append(speed)
    is_fix=[sv<SACCADE_VEL_MIN for sv in v]
    fix=[]; i=0
    while i<len(ts):
        if is_fix[i]:
            j=i
            while j<len(ts) and is_fix[j]: j+=1
            if (ts[j-1]-ts[i])*1000.0>=FIX_MIN_MS: fix.append((i,j-1))
            i=j
        else: i+=1
    n_fix=len(fix)
    mean_fix_ms= sum((ts[j]-ts[i])*1000.0 for i,j in fix)/n_fix if n_fix else 0.0
    reg=0; last=None
    for gx in gax_s:
        if last is not None and gx<last: reg+=1
        last=gx
    blink=''
    if au45:
        bl=0; prev=0
        for a in au45:
            if prev==0 and a==1: bl+=1
            prev=a
        blink=bl/max(1e-6,dur/60.0)
    return {'file':os.path.basename(of_csv),'frames_kept':len(ts),'duration_s':dur,'n_fix':n_fix,'mean_fix_ms':mean_fix_ms,'regressions_proxy':reg,
            'blink_rate_per_min':blink, 'pose_Rx_mean':(sum(Rx)/len(Rx)) if Rx else '',
            'pose_Ry_mean':(sum(Ry)/len(Ry)) if Ry else '', 'pose_Rz_mean':(sum(Rz)/len(Rz)) if Rz else ''}

def compute_calibration_qc(sess_dir:str, of_start_unix:float|None):
    if of_start_unix is None: return None
    events=[]
    try:
        with open(os.path.join(sess_dir,'markers.csv'),'r',encoding='utf-8') as f:
            rdr=csv.reader(f); next(rdr,None)
            for row in rdr:
                try:
                    events.append({'unix':float(row[1]), 'label':row[3], 'details':json.loads(row[4]) if row[4] else {}})
                except Exception: pass
    except Exception:
        return None
    rows=[]
    for p in parse_openface_csvs(sess_dir):
        with open(p,'r',encoding='utf-8') as f:
            head=f.readline().strip().split(','); col={n:i for i,n in enumerate(head)}
            need={'timestamp','gaze_angle_x','gaze_angle_y','success','confidence'}
            if not need.issubset(col): continue
            for line in f:
                pr=line.rstrip('\n').split(',')
                try:
                    t=float(pr[col['timestamp']]); s=int(pr[col['success']]); c=float(pr[col['confidence']])
                    if s==1 and c>=0.8:
                        rows.append((t,float(pr[col['gaze_angle_x']]),float(pr[col['gaze_angle_y']])))
                except Exception: pass
    if not rows: return None
    samples={}
    for e in events:
        if e['label']!='calib_point': continue
        det=e['details']; sx=det.get('x'); sy=det.get('y'); start=det.get('unix'); dur=float(det.get('dur_s',2.0))
        if sx is None or sy is None or start is None: continue
        gx=[]; gy=[]
        for (t,x,y) in rows:
            u=of_start_unix+t
            if u>=start and u<=start+dur:
                gx.append(x); gy.append(y)
        if gx:
            name=det.get('name','pt')
            samples.setdefault(name,{'gx':[],'gy':[],'sx':sx,'sy':sy})
            samples[name]['gx']+=gx; samples[name]['gy']+=gy
    report={'points':{},'mapping':None}
    pts_gx=[]; pts_gy=[]; pts_x=[]; pts_y=[]
    for name,v in samples.items():
        mgx=sum(v['gx'])/len(v['gx']); mgy=sum(v['gy'])/len(v['gy'])
        report['points'][name]={'screen_x':v['sx'],'screen_y':v['sy'],'mean_gaze_x':mgx,'mean_gaze_y':mgy,'n':len(v['gx'])}
        pts_gx.append(mgx); pts_gy.append(mgy); pts_x.append(v['sx']); pts_y.append(v['sy'])
    if len(pts_gx)>=3:
        fit=lstsq_2d(pts_gx,pts_gy,pts_x,pts_y)
        if fit:
            report['mapping']={'cx':fit['cx'],'cy':fit['cy'],'R2x':fit['R2x'],'R2y':fit['R2y']}
    with open(os.path.join(sess_dir,'calibration_report.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    return report

def export_normalized_gaze(sess_dir:str):
    cal_p=os.path.join(sess_dir,'calibration_report.json')
    if not os.path.isfile(cal_p): return
    cal=json.load(open(cal_p,'r',encoding='utf-8'))
    if not cal or not cal.get('mapping'): return
    cx=cal['mapping']['cx']; cy=cal['mapping']['cy']
    intervals={}
    with open(os.path.join(sess_dir,'markers.csv'),'r',encoding='utf-8') as f:
        rdr=csv.reader(f); header=next(rdr,None); cur=None
        for row in rdr:
            try:
                label=row[3]; det=json.loads(row[4]) if row[4] else {}; of_t=float(row[2]) if row[2] else None
            except Exception:
                continue
            if label=='reading_start' and of_t is not None:
                cur=det.get('trial'); intervals[cur]=[of_t,None]
            elif label=='reading_end' and of_t is not None and cur is not None:
                if det.get('trial')==cur and cur in intervals:
                    intervals[cur][1]=of_t; cur=None
    rows=[]
    for p in parse_openface_csvs(sess_dir):
        with open(p,'r',encoding='utf-8') as f:
            head=f.readline().strip().split(','); col={n:i for i,n in enumerate(head)}
            if not {'timestamp','gaze_angle_x','gaze_angle_y','success','confidence'}.issubset(col): continue
            for line in f:
                pr=line.rstrip('\n').split(',')
                try:
                    t=float(pr[col['timestamp']]); s=int(pr[col['success']]); c=float(pr[col['confidence']])
                    if s==1 and c>=0.8:
                        rows.append((t,float(pr[col['gaze_angle_x']]),float(pr[col['gaze_angle_y']])))
                except Exception: pass
    for trial_id,(t0,t1) in intervals.items():
        if t0 is None or t1 is None: continue
        lp=os.path.join(sess_dir,f'layout_{trial_id}.json')
        if not os.path.isfile(lp): continue
        lay=json.load(open(lp,'r',encoding='utf-8')); L=lay['text_area_abs']
        left=L['left']; top=L['top']; width=L['width']; height=L['height']
        out=os.path.join(sess_dir,f'gaze_norm_{trial_id}.csv')
        with open(out,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['of_t','screen_x','screen_y','norm_x','norm_y'])
            for (t,gx,gy) in rows:
                if t<t0 or t>t1: continue
                sx=cx[0]*gx + cx[1]*gy + cx[2]
                sy=cy[0]*gx + cy[1]*gy + cy[2]
                nx=(sx-left)/max(1e-6,width); ny=(sy-top)/max(1e-6,height)
                nx=max(0.0,min(1.0,nx)); ny=max(0.0,min(1.0,ny))
                w.writerow([f"{t:.3f}",f"{sx:.1f}",f"{sy:.1f}",f"{nx:.4f}",f"{ny:.4f}"])

def run_analysis(sess_dir:str, ofc:'OpenFaceController'):
    mets=[]; paths=parse_openface_csvs(sess_dir)
    for p in paths:
        try: mets.append(analyze_csv(p))
        except Exception as e: print("[Analysis] skip", p, "->", e)
    if mets:
        with open(os.path.join(sess_dir,'metrics.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(mets[0].keys())); w.writeheader(); [w.writerow(m) for m in mets]
    try:
        compute_calibration_qc(sess_dir, ofc.of_start_unix if ofc else None)
    except Exception as e:
        print("[Analysis] calib QC error:", e)
    try:
        export_normalized_gaze(sess_dir)
    except Exception as e:
        print("[Analysis] gaze_norm error:", e)
    with open(os.path.join(sess_dir,'analysis_done.txt'),'w') as f:
        f.write(datetime.now().isoformat(timespec='seconds')+"\n")

class ExperimentApp:
    def _normalize_segment(self, s: str) -> str:
        import re
        s = s.replace("\r", "\n")
        s = re.sub(r"\n{2,}", "\n\n", s)
        s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
        s = re.sub(r"[ \t]+", " ", s)
        return s

    def __init__(self, root: tk.Tk, trials):
        self.root=root
        if True: self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG_COLOR); self.root.update_idletasks()
        self.screen_w=self.root.winfo_screenwidth(); self.screen_h=self.root.winfo_screenheight()
        self.canvas=tk.Canvas(self.root,bg=BG_COLOR,highlightthickness=0); self.canvas.pack(fill="both",expand=True)
        self.root.update_idletasks()
        self.trials=trials; self.text_widget=None
        self.participant_id=""; self.group=""
        self.session_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        self._build_start_panel()

        self.enter_evt=threading.Event()
        self.root.bind_all("<Return>", self._on_enter)
        self.root.bind_all("<KP_Enter>", self._on_enter)
        self.root.bind("<Escape>", lambda e: self._quit())

        self._compute_text_area()

        self.ofc=None
        self.markers=None
        self.telemetry=None
        self.state="start_panel"
        self.current=-1; self.reading=False

    def _build_start_panel(self):
        self.panel = tk.Frame(self.canvas, bg=BG_COLOR, highlightthickness=0)
        self.canvas.create_window(self.root.winfo_width()//2, self.root.winfo_height()//2, window=self.panel, anchor="center")
        row=0
        tk.Label(self.panel, text="Numéro de simulation :", bg=BG_COLOR, fg=FG_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE)).grid(row=row, column=0, sticky="e", padx=10, pady=8)
        self.e_pid=tk.Entry(self.panel, font=(FONT_FAMILY, BASE_FONT_SIZE), width=20)
        self.e_pid.grid(row=row, column=1, sticky="w", padx=10, pady=8); row+=1
        tk.Label(self.panel, text="Type de profil :", bg=BG_COLOR, fg=FG_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE)).grid(row=row, column=0, sticky="e", padx=10, pady=8)
        self.cb_grp=ttk.Combobox(self.panel, values=["dys","control","autre"], state="readonly", width=18, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.cb_grp.set("dys")
        self.cb_grp.grid(row=row, column=1, sticky="w", padx=10, pady=8); row+=1

        self.btn_of = tk.Button(self.panel, text="Lancer OpenFace", font=(FONT_FAMILY, BASE_FONT_SIZE), command=self._start_openface, highlightthickness=0, borderwidth=0)
        self.btn_of.grid(row=row, column=0, padx=10, pady=12, sticky="ew")
        self.lbl_status = tk.Label(self.panel, text="OpenFace : non lancé", bg=BG_COLOR, fg="#aa0000", font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.lbl_status.grid(row=row, column=1, padx=10, pady=12, sticky="w"); row+=1

        self.btn_go = tk.Button(self.panel, text="Commencer la calibration", state=tk.DISABLED, font=(FONT_FAMILY, BASE_FONT_SIZE), command=self._start_protocol, highlightthickness=0, borderwidth=0)
        self.btn_go.grid(row=row, column=0, columnspan=2, padx=10, pady=8, sticky="ew")

    def _start_openface(self):
        pid=self.e_pid.get().strip() or "001"; grp=self.cb_grp.get().strip() or "unknown"
        self.participant_id=pid; self.group=grp
        self.sess_dir=ensure_dir(os.path.join(DATA_ROOT,f"P{pid}_{grp}_{self.session_id}"))
        self.events_path=os.path.join(self.sess_dir,'events.jsonl')
        with open(self.events_path,'a',encoding='utf-8') as f: f.write(json.dumps({"event":"app_start","ts":now_iso()},ensure_ascii=False)+"\n")
        # font report
        try:
            test_font, picked = probe_semibold(BASE_FONT_SIZE, fallback_family=FONT_FAMILY)
            rep=os.path.join(self.sess_dir,"font_report.txt")
            with open(rep,"w",encoding="utf-8") as rf:
                rf.write(f"Chosen semi-bold font: {picked}\n")
                rf.write("Tip: export SEMIBOLD_FAMILY=\"Avenir Next Medium\" (si présent dans tkfont.families())\n")
            print("[Font] Semi-bold chosen:", picked)
        except Exception as e:
            print("[Font] Probe error:", e)

        try:
            self.ofc=OpenFaceController(ensure_dir(os.path.join(self.sess_dir,"of_session"))); self.ofc.start_once()
        except Exception as e:
            messagebox.showerror("OpenFace", f"Impossible de lancer OpenFace:\n{e}")
            return
        self.markers=MarkerLogger(self.sess_dir); self.markers.set_of_start(self.ofc.of_start_unix); self.markers.mark("of_start",{})
        self.telemetry=TelemetrySampler(self,self.sess_dir); self.telemetry.start()
        self.lbl_status.config(text="OpenFace : lancé (OK)", fg="#007700")
        self.btn_go.config(state=tk.NORMAL)

    def _start_protocol(self):
        try: self.panel.destroy()
        except Exception: pass
        self.state="await_start"
        self._countdown(5,"Préparation...")
        self._calibration_block(label="Calibration (1/2)")
        self._calibration_block(label="Calibration (2/2)")
        self._countdown(3,"Lecture 1 dans...")
        self.current=0; self._start_trial(self.trials[self.current])

    def _log(self,obj):
        if not hasattr(self,'events_path'): return
        with open(self.events_path,'a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    def _quit(self):
        if messagebox.askokcancel("Quitter","Quitter l'expérience ?"):
            try:
                if self.telemetry: self.telemetry.stop()
                if self.ofc: self.ofc.stop()
            except Exception: pass
            self.root.destroy()
    def window_bbox(self):
        try:
            self.root.update_idletasks()
            x=self.root.winfo_rootx(); y=self.root.winfo_rooty()
            w=self.root.winfo_width(); h=self.root.winfo_height()
            return (x,y,x+w,y+h)
        except Exception: return None
    def _compute_text_area(self):
        self.root.update_idletasks()
        w=self.root.winfo_width() or self.root.winfo_screenwidth()
        h=self.root.winfo_height() or self.root.winfo_screenheight()
        W=int(w*TEXT_W_PCT); H=int(h*TEXT_H_PCT); cx=w//2; cy=h//2
        self.text_area={'w':W,'h':H,'cx':cx,'cy':cy}
    def _clear_canvas(self): self.canvas.delete("all"); self.root.update_idletasks()
    def _clear_enter(self): self.enter_evt.clear()
    def _beep(self):
        try: self.root.bell()
        except Exception: sys.stdout.write('\a'); sys.stdout.flush()
    def _on_enter(self, e=None):
        self.enter_evt.set()
        if self.state=="reading" and self.reading:
            self.reading=False
            tr=self.trials[self.current]
            if self.markers: self.markers.mark("reading_end",{"trial":tr.trial_id})
            self._log({"event":"reading_end","trial":tr.trial_id})
            self._pause(15,True,"Pause avant QCM", then=self._start_qcm)
    def _title(self, text):
        self._clear_canvas()
        self.canvas.create_text(self.root.winfo_width()//2,self.root.winfo_height()//2,text=text,fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+6,"bold"))
        self.root.update()
    def _countdown(self, seconds:int, label:str=""):
        self.state="countdown"; self._clear_enter(); self._clear_canvas()
        y=self.root.winfo_height()//2
        if label: self.canvas.create_text(self.root.winfo_width()//2,y-80,text=label,fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
        for s in range(seconds,0,-1):
            self.canvas.delete("count")
            self.canvas.create_text(self.root.winfo_width()//2,y,text=str(s),tags="count",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+28,"bold"))
            if s<=BEEP_LAST_SECONDS: self._beep()
            self.root.update(); time.sleep(1)
        self.canvas.delete("count"); self.root.update()
    def _calibration_block(self,label="Calibration"):
        self.state="calibration"; self._clear_canvas()
        W=self.root.winfo_width(); H=self.root.winfo_height()
        points=[(W//2,H//2,'center'),
                (int(W*0.1),int(H*0.1),'top_left'),
                (int(W*0.9),int(H*0.1),'top_right'),
                (int(W*0.1),int(H*0.9),'bottom_left'),
                (int(W*0.9),int(H*0.9),'bottom_right')]
        rad=18
        for (x,y,name) in points:
            self._clear_enter(); self._clear_canvas()
            self.canvas.create_text(W//2,int(H*0.08),text=label+f" — {name}",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
            self.canvas.create_oval(x-rad,y-rad,x+rad,y+rad,fill="#D60000",outline="")
            self.canvas.create_line(x-20,y,x+20,y,fill="#111111",width=3)
            self.canvas.create_line(x,y-20,x,y+20,fill="#111111",width=3)
            self.root.update()
            start=time.time()
            if self.markers: self.markers.mark("calib_point",{"name":name,"x":x,"y":y,"unix":start,"dur_s":2.0})
            for _ in range(20):
                time.sleep(0.1); self.root.update()
        self._log({"event":"calibration_block_done","label":label})
    def _start_trial(self,tr:'TrialDef'):
        self.state="reading"; self._clear_canvas(); self._compute_text_area()
        text=open(tr.text_path,'r',encoding='utf-8').read()
        self._render_text(text)
        self._save_layout(tr.trial_id)
        self.reading=True
        if self.markers: self.markers.mark("reading_start",{"trial":tr.trial_id,"condition":tr.condition})
        self._log({"event":"reading_start","trial":tr.trial_id,"condition":tr.condition})
    def _save_layout(self, trial_id:str):
        self.root.update_idletasks()
        left=self.text_area['cx']-self.text_area['w']//2
        top =self.text_area['cy']-self.text_area['h']//2
        abs_left=self.root.winfo_rootx()+left; abs_top=self.root.winfo_rooty()+top
        layout={"trial_id":trial_id,"font_family":FONT_FAMILY,"base_font_size":BASE_FONT_SIZE,
                "line_spacing_px":int(max(2,BASE_FONT_SIZE*0.5)),
                "text_area_rel":{"left":left,"top":top,"width":self.text_area['w'],"height":self.text_area['h']},
                "text_area_abs":{"left":abs_left,"top":abs_top,"width":self.text_area['w'],"height":self.text_area['h']},
                "screen":{"w":self.root.winfo_width(),"h":self.root.winfo_height()}}
        with open(os.path.join(self.sess_dir,f"layout_{trial_id}.json"),'w',encoding='utf-8') as f: json.dump(layout,f,ensure_ascii=False,indent=2)
    def _render_text(self, text:str):
        txt=tk.Text(self.canvas,wrap="word",bg=BG_COLOR,fg=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE),borderwidth=0,highlightthickness=0)
        self.canvas.create_window(self.text_area['cx'],self.text_area['cy'],window=txt,anchor="center",width=self.text_area['w'],height=self.text_area['h'])
        sp=int(max(2,BASE_FONT_SIZE*0.5))
        txt.tag_configure("body", spacing1=0, spacing2=sp, spacing3=0, lmargin1=0, lmargin2=0, justify='left')
        semibold_font, picked = probe_semibold(BASE_FONT_SIZE, fallback_family=FONT_FAMILY)
        txt.tag_configure("bold", font=semibold_font)
        # Affiche dans la console la sélection réelle
        print("[Font] <strong> uses:", picked)
        # Parsing des balises <strong> / <b>
        i=0
        def norm(seg:str):
            import re
            seg = seg.replace("\r","\n")
            seg = re.sub(r"\n{2,}", "\n\n", seg)
            seg = re.sub(r"(?<!\n)\n(?!\n)", " ", seg)
            seg = re.sub(r"[ \t]+", " ", seg)
            return seg
        while i<len(text):
            b1=text.find("<strong>",i); b2=text.find("<b>",i)
            if b1==-1 and b2==-1:
                seg=norm(text[i:])
                if seg: txt.insert("end", seg, ("body",))
                break
            bstart=min([x for x in [b1,b2] if x!=-1])
            if bstart>i:
                seg=norm(text[i:bstart])
                if seg: txt.insert("end", seg, ("body",))
            if text.startswith("<strong>",bstart):
                tago,tagc="<strong>","</strong>"
            else:
                tago,tagc="<b>","</b>"
            bend=text.find(tagc,bstart+len(tago))
            if bend==-1:
                seg=norm(text[bstart:])
                if seg: txt.insert("end", seg, ("body",))
                break
            bold_text=norm(text[bstart+len(tago):bend])
            if bold_text: txt.insert("end", bold_text, ("body","bold"))
            i=bend+len(tagc)
        txt.configure(state="disabled")
        self.text_widget=txt; self.root.update()
    def _pause(self, seconds:int, beep_last:bool, label:str, then=None):
        self.state="pause"; self._clear_enter()
        for s in range(seconds,0,-1):
            self._clear_canvas()
            self.canvas.create_text(self.root.winfo_width()//2,self.root.winfo_height()//2-40,text=label,fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
            self.canvas.create_text(self.root.winfo_width()//2,self.root.winfo_height()//2+10,text=str(s),fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+18,'bold'))
            if beep_last and s<=BEEP_LAST_SECONDS: self._beep()
            self.root.update(); time.sleep(1)
        if then: then()
    def _start_qcm(self):
        tr=self.trials[self.current]; q= json.load(open(tr.qcm_path,'r',encoding='utf-8'))
        out_csv=os.path.join(self.sess_dir,f"qcm_{tr.trial_id}.csv")
        clicklog=os.path.join(self.sess_dir,f"qcm_clicklog_{tr.trial_id}.csv")
        with open(clicklog,'w',newline='',encoding='utf-8') as cf:
            csv.writer(cf).writerow(["trial_id","q_index","event_ts_iso","elapsed_ms","selected_idx"])
        with open(out_csv,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(["trial_id","q_index","type","q_text","choice","text_answer","correct_idx","n_changes","rt_ms","audio_path","choice_seq_json","choice_time_ms_json","clicklog_csv"])
            items=q.get("questions",[])
            for idx,item in enumerate(items):
                if idx<6:
                    choice, changes, rt, seq, times = self._q_mcq(item["text"], item.get("options",["A","B","C","D"]), lambda sel, t0: self._append_click(clicklog,tr.trial_id,idx,sel,t0))
                    w.writerow([tr.trial_id,idx,"mcq",item["text"],choice,"",item.get("correct",""),changes,rt,"",json.dumps(seq),json.dumps(times),os.path.basename(clicklog)])
                else:
                    txt, rt, aud = self._q_open(item["text"])
                    w.writerow([tr.trial_id,idx,"open",item["text"],"",txt,"",0,rt,aud or "","","",""])
        self._pause(15,True,"Pause avant échelle", then=self._start_likert)
    def _append_click(self, path, trial_id, q_idx, sel, t0):
        with open(path,'a',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow([trial_id,q_idx,now_iso(), int((time.time()-t0)*1000),sel])
    def _q_mcq(self, qtext, options, on_click):
        self._clear_enter(); self._clear_canvas(); margin=100
        self.canvas.create_text(self.root.winfo_width()//2,margin,text="Question (QCM)",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+6,'bold'))
        self.canvas.create_text(self.root.winfo_width()//2,margin+60,text=qtext,width=int(self.root.winfo_width()*0.8),fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
        var=tk.IntVar(value=-1); changes=0; t0=time.time(); y0=margin+140; radios=[]; items=[]; seq=[]; times=[]
        def on_change(*_):
            nonlocal changes
            sel=var.get()
            if sel!=-1:
                changes+=1; seq.append(sel); times.append(int((time.time()-t0)*1000)); on_click(sel,t0)
        for i,opt in enumerate(options):
            rb=tk.Radiobutton(self.canvas,text=opt,variable=var,value=i,bg=BG_COLOR,fg=FG_COLOR,
                              font=(FONT_FAMILY,BASE_FONT_SIZE),selectcolor=BG_COLOR,activebackground=BG_COLOR,activeforeground=FG_COLOR,
                              command=on_change,anchor='w',justify='left',highlightthickness=0)
            it=self.canvas.create_window(self.root.winfo_width()//2-300,y0+i*50,window=rb,anchor='w'); radios.append(rb); items.append(it)
        while not self.enter_evt.is_set(): self.root.update(); time.sleep(0.01)
        for rb in radios:
            try: rb.destroy()
            except Exception: pass
        for it in items:
            try: self.canvas.delete(it)
            except Exception: pass
        rt=int((time.time()-t0)*1000); self._clear_enter()
        return var.get(), changes, rt, seq, times
    def _q_open(self, qtext):
        self._clear_enter(); self._clear_canvas(); margin=100
        self.canvas.create_text(self.root.winfo_width()//2,margin,text="Réponse libre (Entrée = valider)",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+6,'bold'))
        self.canvas.create_text(self.root.winfo_width()//2,margin+60,text=qtext,width=int(self.root.winfo_width()*0.8),fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
        entry=tk.Text(self.canvas,width=80,height=5,bg="#f7f7f7",fg="#111",font=(FONT_FAMILY,BASE_FONT_SIZE),highlightthickness=0,borderwidth=0)
        e_item=self.canvas.create_window(self.root.winfo_width()//2,margin+140,window=entry,anchor='n')
        aud_path=None; recording={"on":False,"buf":[],"stream":None,"sr":16000}
        def toggle():
            if not HAS_AUDIO: return
            if recording["on"]:
                try: recording["stream"].stop(); recording["stream"].close()
                except Exception: pass
                recording["on"]=False; mic.config(text="🎙️ Démarrer dictée")
            else:
                recording["on"]=True; recording["buf"]=[]
                def cb(indata,frames,time_info,status):
                    try: recording["buf"].append(indata.tobytes())
                    except Exception: pass
                import sounddevice as _sd
                recording["stream"]=_sd.InputStream(samplerate=recording["sr"],channels=1,dtype='int16',callback=cb); recording["stream"].start()
                mic.config(text="⏹️ Arrêter dictée")
        mic=tk.Button(self.canvas,text=("🎙️ Démarrer dictée" if HAS_AUDIO else "🎙️ Dictée indisponible"),
                      state=(tk.NORMAL if HAS_AUDIO else tk.DISABLED),command=toggle,font=(FONT_FAMILY,BASE_FONT_SIZE),highlightthickness=0,borderwidth=0)
        m_item=self.canvas.create_window(self.root.winfo_width()//2,margin+300,window=mic,anchor='n')
        t0=time.time()
        while not self.enter_evt.is_set(): self.root.update(); time.sleep(0.01)
        if recording["on"]:
            try: recording["stream"].stop(); recording["stream"].close()
            except Exception: pass
            recording["on"]=False
        if HAS_AUDIO and recording["buf"]:
            try:
                aud_dir=ensure_dir(os.path.join(self.sess_dir,"audio"))
                aud_path=os.path.join(aud_dir,f"q_open_{int(time.time())}.wav")
                with wave.open(aud_path,'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(recording["sr"]); wf.writeframes(b"".join(recording["buf"]))
            except Exception: aud_path=None
        answer=entry.get("1.0","end").strip()
        try: entry.destroy(); mic.destroy()
        except Exception: pass
        try: self.canvas.delete(e_item); self.canvas.delete(m_item)
        except Exception: pass
        rt=int((time.time()-t0)*1000); self._clear_enter()
        return answer, rt, aud_path
    def _start_likert(self):
        tr=self.trials[self.current]; lik=json.load(open(tr.likert_path,'r',encoding='utf-8'))
        out_csv=os.path.join(self.sess_dir,f"likert_{tr.trial_id}.csv")
        with open(out_csv,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(["trial_id","item_idx","item_text","response","rt_ms"])
            for i,item in enumerate(lik.get("items",[])):
                resp,rt=self._likert_item(item); w.writerow([tr.trial_id,i,item,resp,rt])
        if self.current==0:
            self._long_pause_then_next()
        else:
            self._finish()
    def _likert_item(self,item_text:str):
        self._clear_enter(); self._clear_canvas(); margin=100
        self.canvas.create_text(self.root.winfo_width()//2,margin,text="Échelle (1–7)",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+6,'bold'))
        self.canvas.create_text(self.root.winfo_width()//2,margin+60,text=item_text,width=int(self.root.winfo_width()*0.8),fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE))
        var=tk.IntVar(value=4); t0=time.time(); y=margin+140; x0=self.root.winfo_width()//2-240; btns=[]; items=[]
        for v in range(1,8):
            b=tk.Radiobutton(self.canvas, text=str(v),variable=var,value=v,bg=BG_COLOR,fg=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE),
                             selectcolor=BG_COLOR,activebackground=BG_COLOR,activeforeground=FG_COLOR)
            it=self.canvas.create_window(x0+(v-1)*80,y,window=b,anchor='w'); btns.append(b); items.append(it)
        while not self.enter_evt.is_set(): self.root.update(); time.sleep(0.01)
        for b in btns:
            try: b.destroy()
            except Exception: pass
        for it in items:
            try: self.canvas.delete(it)
            except Exception: pass
        rt=int((time.time()-t0)*1000); self._clear_enter()
        return var.get(), rt
    def _long_pause_then_next(self):
        self._pause(60, True, "Grande pause")
        self._calibration_block(label="Calibration (1/2)")
        self._calibration_block(label="Calibration (2/2)")
        self._countdown(3,"Lecture 2 dans...")
        self.current=1; self._start_trial(self.trials[1])
    def _finish(self):
        self._title("Analyse en cours..."); self.root.update(); time.sleep(0.5)
        try: run_analysis(self.sess_dir, self.ofc)
        except Exception as e: print("[Analysis] error:", e)
        self._clear_canvas()
        self.canvas.create_text(self.root.winfo_width()//2,self.root.winfo_height()//2-40,text="Session terminée — merci !",fill=FG_COLOR,font=(FONT_FAMILY,BASE_FONT_SIZE+6,'bold'))
        btn=tk.Button(self.canvas,text="Ouvrir le dossier des résultats et quitter",font=(FONT_FAMILY,BASE_FONT_SIZE),command=self._open_results,highlightthickness=0,borderwidth=0)
        self.canvas.create_window(self.root.winfo_width()//2,self.root.winfo_height()//2+10,window=btn)
    def _open_results(self):
        print("[RESULTS] Dossier :", self.sess_dir)
        try:
            sysname=platform.system()
            if sysname=="Darwin": subprocess.Popen(["open", self.sess_dir])
            elif sysname=="Windows": os.startfile(self.sess_dir)  # type: ignore
            else: subprocess.Popen(["xdg-open", self.sess_dir])
        except Exception as e: print("[RESULTS] open error:", e)
        try:
            if self.telemetry: self.telemetry.stop()
            if self.ofc: self.ofc.stop()
        except Exception: pass
        self.root.destroy()

def main():
    try:
        trials=read_plan(PLAN_CSV)
    except Exception as e:
        print("[FATAL] Cannot read plan:", e); print("Columns must be: trial_id,condition,text_path,qcm_path,likert_path"); return
    root=tk.Tk(); app=ExperimentApp(root,trials)
    def on_close():
        try:
            if app.telemetry: app.telemetry.stop()
            if app.ofc: app.ofc.stop()
        except Exception: pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__=="__main__": main()
