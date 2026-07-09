@echo off
chcp 65001 >nul
cd /d "C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle"

echo ============================================================
echo  RESUME CHAIN: ADIM 2 (trendyol_ai) + ENSEMBLE
echo  Model: Trendyol domain CE  -  Veri: ai_expanded_terms.csv
echo ============================================================

REM ===== ADIM 2: trendyol_ai (domain + AI genisletme) =====
echo.
echo [%time%] ADIM 2: trendyol_ai egitimi basliyor (AI genisletilmis sorgular)...
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_ai --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file ai_expanded_terms.csv
echo [%time%] trendyol_ai BITTI!

echo [%time%] ai submission'lari uretiliyor...
python -X utf8 -c "import numpy as np,pandas as pd,os;DATA=r'C:\Users\ASUS\Desktop\trendyol';ART=os.path.join(DATA,'artifacts');score=np.load(os.path.join(ART,'ce_test_trendyol_ai.npy'));sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'));[pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_ai_p{int(ppr*100)}.csv'),index=False) or print(f'  ai PPR{ppr}: {(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 5: Optimal PPR analizi (tum modeller) =====
echo.
echo [%time%] ADIM 5: Tum modeller icin optimal PPR hesaplaniyor...
python -X utf8 find_optimal_ppr.py

REM ===== ADIM 6: BUYUK ENSEMBLE =====
echo.
echo ============================================================
echo [%time%] ADIM 6: Tum modellerin ensemble'i
echo ============================================================
python -X utf8 -c "
import numpy as np,pandas as pd,os
DATA=r'C:\Users\ASUS\Desktop\trendyol';ART=os.path.join(DATA,'artifacts')
def rn(x):
    o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(len(x));return r/(len(x)-1)
models={}
for n in ['trendyol_ce','trendyol_full','trendyol_qexp','trendyol_qaug','trendyol_ai']:
    p=os.path.join(ART,f'ce_test_{n}.npy')
    if os.path.exists(p): models[n]=rn(np.load(p)); print(f'  +{n}')
bp=os.path.join(ART,'_stale_easy','ce_test_berturk.npy')
if os.path.exists(bp): models['berturk']=rn(np.load(bp)); print('  +berturk')
sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'))
# --- Ensemble A: SADECE domain modelleri ---
dom={k:v for k,v in models.items() if 'trendyol' in k}
if len(dom)>=2:
    ens_d=sum(dom.values())/len(dom)
    for ppr in [0.30,0.33,0.36]:
        thr=np.quantile(ens_d,1-ppr);pred=(ens_d>=thr).astype(int)
        out=os.path.join(DATA,f'submission_domens{len(dom)}_p{int(ppr*100)}.csv')
        pd.DataFrame({'id':sub['id'],'prediction':pred}).to_csv(out,index=False)
        print(f'  domain-ens({len(dom)}) PPR{ppr}: {pred.mean():.4f} -> {os.path.basename(out)}')
# --- Ensemble B: TUM modeller ---
if len(models)>=2:
    ens_all=sum(models.values())/len(models)
    for ppr in [0.30,0.33,0.36]:
        thr=np.quantile(ens_all,1-ppr);pred=(ens_all>=thr).astype(int)
        out=os.path.join(DATA,f'submission_fullens{len(models)}_p{int(ppr*100)}.csv')
        pd.DataFrame({'id':sub['id'],'prediction':pred}).to_csv(out,index=False)
        print(f'  full-ens({len(models)}) PPR{ppr}: {pred.mean():.4f} -> {os.path.basename(out)}')
print(f'Modeller: {list(models.keys())}')
"

echo.
echo ============================================================
echo  HERSEY BITTI! %date% %time%
echo ============================================================
pause
