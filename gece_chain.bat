@echo off
chcp 65001 >nul
cd /d "C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle"

echo ============================================================
echo  GECE EGITIM CHAIN - %date% %time%
echo  Internet GEREKMEZ - tum modeller lokal cache'te
echo  Siralama: qaug bekle - trendyol_ai - xlm-r-large - seed2
echo ============================================================

REM ===== ADIM 1: qaug egitimi (sentetik augmentasyon ile) =====
echo.
echo [%time%] ADIM 1: qaug egitimi basliyor...
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_qaug --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file augmented_terms.csv --pairs_file train_pairs_augmented.parquet
echo [%time%] qaug egitimi BITTI.

:QAUG_SUB
echo [%time%] qaug submission'lari uretiliyor...
python -X utf8 -c "import numpy as np,pandas as pd,os;DATA=r'C:\Users\ASUS\Desktop\trendyol';ART=os.path.join(DATA,'artifacts');score=np.load(os.path.join(ART,'ce_test_trendyol_qaug.npy'));sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'));[pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_qaug_p{int(ppr*100)}.csv'),index=False) or print(f'  qaug PPR{ppr}: {(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 2: trendyol_ai (domain + AI genisletme) =====
echo.
echo ============================================================
echo [%time%] ADIM 2: trendyol_ai egitimi (AI genisletilmis sorgular)
echo   Model: Trendyol domain CE  |  Veri: ai_expanded_terms.csv
echo ============================================================
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_ai --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file ai_expanded_terms.csv
echo [%time%] trendyol_ai BITTI!
python -X utf8 -c "import numpy as np,pandas as pd,os;DATA=r'C:\Users\ASUS\Desktop\trendyol';ART=os.path.join(DATA,'artifacts');score=np.load(os.path.join(ART,'ce_test_trendyol_ai.npy'));sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'));[pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_ai_p{int(ppr*100)}.csv'),index=False) or print(f'  ai PPR{ppr}: {(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 3 ve 4 ATLANIYOR =====
echo [%time%] ADIM 3 ve 4 kullanici istegiyle atlaniyor...
echo.

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
for n in ['trendyol_ce','trendyol_full','trendyol_qexp','trendyol_qaug','trendyol_ai','trendyol_s43','xlmr_large']:
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
echo.
echo  SABAH GONDER (onerilen):
echo    1. submission_trendyol_ce_opt_p26.csv  (mevcut en iyi model, optimal PPR)
echo    2. submission_ai_p26.csv               (domain + AI genisletme, optimal PPR)
echo    3. submission_domens*_p26.csv           (domain ensemble, optimal PPR)
echo    4. submission_domens*_p33.csv           (domain ensemble, eski PPR)
echo    5. submission_fullens*_p27.csv          (full ensemble)
echo.
echo  EN IYI MEVCUT: submission_trendyol_p33.csv = LB 0.83
echo  DENEME: submission_trendyol_ce_opt_p26.csv (holdout optimali)
echo ============================================================
pause
