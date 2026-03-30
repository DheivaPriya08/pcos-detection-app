import os, re, warnings
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

st.set_page_config(page_title="PCOS Detection & Risk Assessment",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:#f0f4ff;}
div[data-testid="stSidebar"]{background:#fff;box-shadow:2px 0 12px rgba(0,0,0,.07);}
.hero{background:linear-gradient(135deg,#4f46e5,#7c3aed);border-radius:18px;padding:30px 34px;color:#fff;margin-bottom:22px;}
.hero h1{font-size:1.85rem;font-weight:700;margin:0 0 5px;}
.hero p{font-size:.92rem;opacity:.88;margin:0;}
.card{background:#fff;border-radius:14px;padding:20px 22px;box-shadow:0 2px 10px rgba(0,0,0,.06);margin-bottom:12px;}
.cl{border-left:4px solid #4f46e5;}.cg{border-left:4px solid #10b981;}
.cr{border-left:4px solid #ef4444;}.cy{border-left:4px solid #f59e0b;}
.mlabel{font-size:.7rem;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.06em;}
.mval{font-size:1.6rem;font-weight:700;color:#111827;line-height:1.2;}
.msub{font-size:.78rem;color:#6b7280;margin-top:2px;}
.risk-lo{background:#ecfdf5;border:1.5px solid #10b981;border-radius:12px;padding:18px;}
.risk-md{background:#fffbeb;border:1.5px solid #f59e0b;border-radius:12px;padding:18px;}
.risk-hi{background:#fef2f2;border:1.5px solid #ef4444;border-radius:12px;padding:18px;}
.pill{display:inline-block;border-radius:999px;padding:3px 11px;font-size:.73rem;font-weight:600;}
.pg{background:#d1fae5;color:#065f46;}.pr{background:#fee2e2;color:#991b1b;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;
    border:none;border-radius:10px;padding:11px 24px;font-weight:600;font-size:.93rem;width:100%;}
.shdr{font-size:.98rem;font-weight:700;color:#374151;border-bottom:2px solid #e5e7eb;padding-bottom:7px;margin:18px 0 12px;}
.info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:13px 17px;font-size:.84rem;color:#1e40af;margin-bottom:14px;}
</style>""", unsafe_allow_html=True)

DOMAIN_WEIGHTS = {"METABOLIC":0.35,"REPRODUCTIVE":0.40,"HYPERANDROGENIC":0.20,"GENERAL":0.05}

def clean_name(n): return re.sub(r"[^0-9a-zA-Z_]","_",n.strip())

@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    needed=["ensemble_meta","base_models","domain_models","train_stats","mice_imputer","scaler","feature_columns","best_threshold"]
    missing=[f for f in needed if not os.path.exists(f"models/{f}.pkl")]
    if missing: return None, f"Missing pkl files: {missing}"
    try:
        return {"meta":joblib.load("models/ensemble_meta.pkl"),"base":joblib.load("models/base_models.pkl"),
                "domain":joblib.load("models/domain_models.pkl"),"stats":joblib.load("models/train_stats.pkl"),
                "mice":joblib.load("models/mice_imputer.pkl"),"scaler":joblib.load("models/scaler.pkl"),
                "features":joblib.load("models/feature_columns.pkl"),"threshold":joblib.load("models/best_threshold.pkl")}, None
    except Exception as e: return None, str(e)

def apply_severity(domain_models, stats, X_sc):
    df=pd.DataFrame(index=X_sc.index)
    for d,(model,fc) in domain_models.items():
        avail=[f for f in fc if f in X_sc.columns]
        if not avail: df[f"{d}_SEV"]=0.0; continue
        p=model.predict_proba(X_sc[avail])[:,1]
        lo,hi=stats[d]["min"],stats[d]["max"]
        df[f"{d}_SEV"]=np.clip((p-lo)/(hi-lo+1e-6),0,1)
    df=df.fillna(0)
    df["OVERALL_SEV"]=sum(df[f"{d}_SEV"]*DOMAIN_WEIGHTS[d] for d in DOMAIN_WEIGHTS if f"{d}_SEV" in df)
    return df

def predict(inp, mdl):
    fc=mdl["features"]
    row=pd.DataFrame([{c:inp.get(c,np.nan) for c in fc}])
    row_imp=pd.DataFrame(mdl["mice"].transform(row),columns=fc)
    row_sc=pd.DataFrame(mdl["scaler"].transform(row_imp),columns=fc)
    sev_df=apply_severity(mdl["domain"],mdl["stats"],row_sc)
    X_final=pd.concat([row_sc,sev_df],axis=1)
    base_probs=pd.DataFrame({name:m.predict_proba(X_final)[:,1] for name,m in mdl["base"].items()})
    prob=mdl["meta"].predict_proba(base_probs)[:,1][0]
    pred=int(prob>=mdl["threshold"])
    sev={d:float(sev_df[f"{d}_SEV"].iloc[0]) for d in DOMAIN_WEIGHTS if f"{d}_SEV" in sev_df}
    sev["OVERALL"]=float(sev_df["OVERALL_SEV"].iloc[0])
    return pred,float(prob),sev

def infertility_risk(inp):
    parts=[]
    amh=inp.get("AMH_ng_mL_",np.nan); fsh=inp.get("FSH_mIU_mL_",np.nan)
    foll=(inp.get("Follicle_No___L_",0) or 0)+(inp.get("Follicle_No___R_",0) or 0)
    cyc=inp.get("Cycle_R_I_",4); bmi=inp.get("BMI",np.nan)
    if not np.isnan(amh): parts.append(("AMH Reserve",max(0,1-amh/5),0.35))
    if not np.isnan(fsh): parts.append(("FSH Level",min(1,fsh/20),0.25))
    if foll>0: parts.append(("Follicle Count",max(0,1-foll/20),0.20))
    parts.append(("Cycle",0.70 if cyc==2 else 0.20,0.15))
    if not np.isnan(bmi): parts.append(("BMI",0.6 if bmi<18.5 or bmi>30 else 0.3 if bmi>25 else 0.0,0.05))
    if not parts: return 0.5,"Moderate",parts
    tw=sum(w for _,_,w in parts); sc=sum(r*w for _,r,w in parts)/tw
    return sc,("Low" if sc<0.35 else "High" if sc>=0.60 else "Moderate"),parts

def radar_chart(sev):
    doms=["METABOLIC","REPRODUCTIVE","HYPERANDROGENIC","GENERAL"]
    vals=[sev.get(d,0) for d in doms]+[sev.get(doms[0],0)]
    ang=[n/len(doms)*2*np.pi for n in range(len(doms))]+[0]
    fig,ax=plt.subplots(figsize=(4,4),subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f0f4ff"); ax.set_facecolor("#f0f4ff")
    ax.plot(ang,vals,"o-",lw=2,color="#4f46e5"); ax.fill(ang,vals,alpha=0.22,color="#4f46e5")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([d.capitalize() for d in doms],fontsize=8)
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1]); ax.set_yticklabels(["25%","50%","75%","100%"],fontsize=6)
    ax.set_title("Domain Severity",fontsize=10,fontweight="bold",pad=14)
    plt.tight_layout(); return fig

def prob_gauge(prob,title):
    col="#ef4444" if prob>0.6 else "#f59e0b" if prob>0.4 else "#10b981"
    fig,ax=plt.subplots(figsize=(4,2)); fig.patch.set_facecolor("#f0f4ff"); ax.set_facecolor("#f0f4ff")
    ax.barh([0],[1],color="#e5e7eb",height=0.4); ax.barh([0],[prob],color=col,height=0.4)
    ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.set_yticks([])
    ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0%","25%","50%","75%","100%"],fontsize=8)
    ax.set_title(f"{title}\n{prob:.1%}",fontsize=10,fontweight="bold",color=col)
    ax.spines[["top","right","left"]].set_visible(False); plt.tight_layout(); return fig

def inf_bar(parts,score):
    if not parts: return None
    labels=[c[0] for c in parts]; vals=[c[1] for c in parts]
    cols=["#ef4444" if v>0.6 else "#f59e0b" if v>0.35 else "#10b981" for v in vals]
    fig,ax=plt.subplots(figsize=(5,2.8)); fig.patch.set_facecolor("#f0f4ff"); ax.set_facecolor("#f0f4ff")
    bars=ax.barh(labels,vals,color=cols,height=0.5,edgecolor="white")
    for b,v in zip(bars,vals): ax.text(v+.02,b.get_y()+b.get_height()/2,f"{v:.0%}",va="center",fontsize=8)
    ax.axvline(score,color="#4f46e5",lw=1.5,ls="--",label=f"Score:{score:.0%}")
    ax.set_xlim(0,1.15); ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8); ax.set_title("Infertility Risk Factors",fontsize=10,fontweight="bold")
    plt.tight_layout(); return fig

def sidebar_inputs():
    st.sidebar.markdown("## 📋 Clinical Inputs")
    inp={}
    with st.sidebar.expander("👤 General",expanded=True):
        inp["Age__yrs_"]=st.number_input("Age (yrs)",15,60,28)
        inp["BMI"]=st.number_input("BMI",10.0,60.0,22.0,step=0.1)
        inp["Weight__Kg_"]=st.number_input("Weight (kg)",30.0,150.0,60.0,step=0.5)
        inp["Hb__g_dl_"]=st.number_input("Hb (g/dL)",5.0,18.0,13.0,step=0.1)
        inp["Pulse_rate_bpm_"]=st.number_input("Pulse (bpm)",40,120,72)
        inp["RR__breaths_min_"]=st.number_input("Resp Rate",10,30,16)
    with st.sidebar.expander("🩸 Hormones & Labs"):
        inp["FSH_mIU_mL_"]=st.number_input("FSH (mIU/mL)",0.0,50.0,7.0,step=0.1)
        inp["LH_mIU_mL_"]=st.number_input("LH (mIU/mL)",0.0,50.0,5.0,step=0.1)
        inp["FSH_LH"]=round(inp["FSH_mIU_mL_"]/max(inp["LH_mIU_mL_"],0.01),3)
        st.caption(f"FSH/LH = {inp['FSH_LH']:.3f} (auto)")
        inp["AMH_ng_mL_"]=st.number_input("AMH (ng/mL)",0.0,20.0,3.5,step=0.1)
        inp["PRG_ng_mL_"]=st.number_input("Progesterone (ng/mL)",0.0,30.0,1.0,step=0.1)
        inp["PRL_ng_mL_"]=st.number_input("Prolactin (ng/mL)",0.0,100.0,15.0,step=0.5)
        inp["TSH__mIU_L_"]=st.number_input("TSH (mIU/L)",0.0,10.0,2.0,step=0.1)
        inp["Vit_D3__ng_mL_"]=st.number_input("Vit D3 (ng/mL)",0.0,100.0,25.0,step=0.5)
        inp["RBS_mg_dl_"]=st.number_input("RBS (mg/dL)",50.0,400.0,90.0,step=1.0)
    with st.sidebar.expander("📐 Measurements"):
        inp["Waist_inch_"]=st.number_input("Waist (inch)",20.0,60.0,32.0,step=0.5)
        inp["Hip_inch_"]=st.number_input("Hip (inch)",25.0,70.0,38.0,step=0.5)
        whr=round(inp["Waist_inch_"]/max(inp["Hip_inch_"],1),3)
        inp["Waist_Hip_Ratio"]=whr; st.caption(f"Waist:Hip = {whr:.3f} (auto)")
        inp["BP__Systolic__mmHg_"]=st.number_input("BP Systolic (mmHg)",80,200,120)
        inp["BP__Diastolic__mmHg_"]=st.number_input("BP Diastolic (mmHg)",50,130,80)
    with st.sidebar.expander("🫘 Follicles & Cycle"):
        inp["Follicle_No___L_"]=st.number_input("Follicles Left",0,30,6)
        inp["Follicle_No___R_"]=st.number_input("Follicles Right",0,30,6)
        inp["Avg_F_size__L__mm_"]=st.number_input("Avg Size L (mm)",0.0,30.0,8.0,step=0.5)
        inp["Avg_F_size__R__mm_"]=st.number_input("Avg Size R (mm)",0.0,30.0,8.0,step=0.5)
        inp["Endometrium__mm_"]=st.number_input("Endometrium (mm)",0.0,20.0,8.0,step=0.5)
        inp["Cycle_R_I_"]=st.selectbox("Cycle Type",[2,4],format_func=lambda x:"Irregular" if x==2 else "Regular")
        inp["Cycle_length_days_"]=st.number_input("Cycle Length (days)",20,60,28)
    with st.sidebar.expander("⚕️ Symptoms & Lifestyle"):
        inp["Weight_gain_Y_N_"]=int(st.checkbox("Weight Gain"))
        inp["hair_growth_Y_N_"]=int(st.checkbox("Excess Hair Growth"))
        inp["Skin_darkening__Y_N_"]=int(st.checkbox("Skin Darkening"))
        inp["Hair_loss_Y_N_"]=int(st.checkbox("Hair Loss"))
        inp["Pimples_Y_N_"]=int(st.checkbox("Pimples"))
        inp["Fast_food__Y_N_"]=int(st.checkbox("Fast Food Regularly"))
        inp["Reg_Exercise_Y_N_"]=int(st.checkbox("Regular Exercise"))
    # Engineered features matching pipeline exactly
    fsh=inp["FSH_mIU_mL_"]; lh=inp["LH_mIU_mL_"]; amh=inp["AMH_ng_mL_"]
    inp["FSH_LH_Ratio"]=round(fsh/max(lh,0.01),4)
    inp["Total_Follicles"]=inp["Follicle_No___L_"]+inp["Follicle_No___R_"]
    inp["AMH_FSH_Ratio"]=round(amh/max(fsh,0.01),4)
    inp["BMI_Waist_Ratio"]=round(inp["BMI"]/max(inp["Waist_inch_"],1),4)
    inp["Androgenic_Score"]=(inp["hair_growth_Y_N_"]+inp["Skin_darkening__Y_N_"]+inp["Hair_loss_Y_N_"]+inp["Pimples_Y_N_"])
    return inp

def main():
    st.markdown("""<div class="hero">
      <h1>🩺 PCOS Detection & Risk Assessment</h1>
      <p>Stacked Ensemble (LR · RF · GradientBoost · CatBoost) &nbsp;|&nbsp; Domain Severity &nbsp;|&nbsp; Infertility Risk</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        mdl, err = load_models()
    if err:
        st.error(f"❌ {err}")
        st.info("Upload all pkl files to the models/ folder on GitHub.")
        st.stop()

    st.markdown('<div class="info">ℹ️ Fill clinical inputs in the sidebar then click <b>▶ Run Analysis</b></div>',unsafe_allow_html=True)
    inp = sidebar_inputs()
    run = st.button("▶ Run Analysis", use_container_width=True)

    if not run:
        c1,c2,c3,c4=st.columns(4)
        for col,icon,t,d in [(c1,"🧬","Stacked Ensemble","AUC 0.953 · MCC 0.896"),
                              (c2,"📊","Domain Severity","4-domain PCOS profiling"),
                              (c3,"🔴","Infertility Risk","AMH · FSH · Follicles · Cycle"),
                              (c4,"📋","Clinical Recs","Personalised guidance")]:
            with col:
                st.markdown(f'<div class="card cl"><div style="font-size:1.5rem">{icon}</div><div class="mlabel">{t}</div><div class="msub">{d}</div></div>',unsafe_allow_html=True)
        return

    try:
        pred,prob,sev=predict(inp,mdl)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.markdown("---"); st.markdown("## 📊 Results")

    diag="PCOS Detected" if pred else "No PCOS Detected"
    vcol="#ef4444" if pred else "#10b981"
    ppill="pr" if pred else "pg"
    picon="⚠️ Positive" if pred else "✅ Negative"
    ov=sev.get("OVERALL",0)
    svlbl="Severe" if ov>0.7 else "Moderate" if ov>0.4 else "Mild"
    iscore,itier,_=infertility_risk(inp)
    tc={"Low":"cg","Moderate":"cy","High":"cr"}[itier]

    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown(f'<div class="card {"cr" if pred else "cg"}"><div class="mlabel">Diagnosis</div><div class="mval" style="color:{vcol};font-size:1.2rem">{diag}</div><div class="msub"><span class="pill {ppill}">{picon}</span></div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card cl"><div class="mlabel">PCOS Probability</div><div class="mval">{prob:.1%}</div><div class="msub">Threshold: {mdl["threshold"]:.3f}</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card cy"><div class="mlabel">Overall Severity</div><div class="mval">{ov:.1%}</div><div class="msub">{svlbl}</div></div>',unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card {tc}"><div class="mlabel">Infertility Risk</div><div class="mval">{iscore:.0%}</div><div class="msub">{itier} tier</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="shdr">📊 Domain Severity Analysis</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2)
    with d1: st.pyplot(radar_chart(sev),use_container_width=True)
    with d2:
        st.pyplot(prob_gauge(prob,"PCOS Probability"),use_container_width=True)
        st.markdown("<br>",unsafe_allow_html=True)
        for dom in ["METABOLIC","REPRODUCTIVE","HYPERANDROGENIC","GENERAL"]:
            v=sev.get(dom,0); bc="#ef4444" if v>0.7 else "#f59e0b" if v>0.4 else "#10b981"
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0"><span style="width:120px;font-size:.78rem;color:#374151">{dom.capitalize()}</span><div style="flex:1;background:#e5e7eb;border-radius:4px;height:9px"><div style="width:{v*100:.0f}%;background:{bc};height:9px;border-radius:4px"></div></div><span style="width:36px;text-align:right;font-size:.78rem;font-weight:600;color:{bc}">{v:.0%}</span></div>',unsafe_allow_html=True)

    st.markdown('<div class="shdr">🔴 Infertility Risk Assessment</div>',unsafe_allow_html=True)
    iscore,itier,iparts=infertility_risk(inp)
    rcls={"Low":"risk-lo","Moderate":"risk-md","High":"risk-hi"}[itier]
    ricon={"Low":"🟢","Moderate":"🟡","High":"🔴"}[itier]
    ir1,ir2=st.columns(2)
    with ir1:
        st.markdown(f'<div class="{rcls}"><div style="font-size:1.05rem;font-weight:700">{ricon} {itier} Infertility Risk</div><div style="font-size:1.8rem;font-weight:700;margin:8px 0">{iscore:.0%}</div><div style="font-size:.8rem;color:#374151">Based on AMH · FSH · Follicles · Cycle · BMI</div></div>',unsafe_allow_html=True)
        notes=[]
        amh=inp.get("AMH_ng_mL_"); fsh=inp.get("FSH_mIU_mL_"); tf=inp.get("Total_Follicles",0)
        if amh and amh<1.0: notes.append("⚠️ Very low AMH — reduced ovarian reserve")
        if amh and amh>6.0: notes.append("ℹ️ High AMH — polycystic morphology likely")
        if fsh and fsh>10:  notes.append("⚠️ Elevated FSH — diminished ovarian reserve")
        if tf>=20:          notes.append("ℹ️ High follicle count — consistent with PCOS")
        if inp.get("Cycle_R_I_")==2: notes.append("⚠️ Irregular cycle — affects conception")
        if notes:
            st.markdown("<br>**Clinical Notes:**",unsafe_allow_html=True)
            for n in notes: st.markdown(f"- {n}")
    with ir2:
        fig=inf_bar(iparts,iscore)
        if fig: st.pyplot(fig,use_container_width=True)

    st.markdown("---"); st.markdown('<div class="shdr">📋 Clinical Recommendations</div>',unsafe_allow_html=True)
    recs=[]
    if pred:
        recs+=["**Confirm PCOS** with transvaginal ultrasound if not done",
               "Full hormonal panel: FSH, LH, AMH, testosterone, DHEAS",
               "Metabolic screen: fasting glucose, insulin, HbA1c, lipid profile"]
        if inp.get("Cycle_R_I_")==2: recs.append("Discuss cycle regulation — OCP or progesterone cycling")
        if inp.get("BMI",22)>25: recs.append("Lifestyle intervention: 5–10% weight loss can restore ovulation")
        if inp.get("BMI",22)>30: recs.append("Consider metformin for insulin sensitisation")
    else:
        recs+=["PCOS unlikely — routine annual review recommended","Maintain healthy BMI and regular exercise"]
    if itier=="High": recs.append("**Fertility specialist referral recommended** — high infertility risk (>60%)")
    elif itier=="Moderate": recs.append("Consider fertility evaluation if conception not achieved in 6–12 months")
    for r in recs: st.markdown(f"• {r}")

    with st.expander("🔍 Base Model Probabilities"):
        fc=mdl["features"]
        row=pd.DataFrame([{c:inp.get(c,np.nan) for c in fc}])
        row_imp=pd.DataFrame(mdl["mice"].transform(row),columns=fc)
        row_sc=pd.DataFrame(mdl["scaler"].transform(row_imp),columns=fc)
        sev_df=apply_severity(mdl["domain"],mdl["stats"],row_sc)
        X_fin=pd.concat([row_sc,sev_df],axis=1)
        bm_cols=st.columns(len(mdl["base"]))
        for col,(name,m) in zip(bm_cols,mdl["base"].items()):
            p=m.predict_proba(X_fin)[:,1][0]; pill="pr" if p>=0.5 else "pg"
            with col:
                st.markdown(f'<div class="card cl" style="text-align:center"><div class="mlabel">{name}</div><div class="mval">{p:.1%}</div><div class="msub"><span class="pill {pill}">{"PCOS" if p>=0.5 else "No PCOS"}</span></div></div>',unsafe_allow_html=True)

    st.markdown('<div style="font-size:.73rem;color:#9ca3af;text-align:center;padding:6px 0">⚕️ For clinical decision support only. Results must be interpreted by a qualified healthcare provider.</div>',unsafe_allow_html=True)

if __name__=="__main__":
    main()
