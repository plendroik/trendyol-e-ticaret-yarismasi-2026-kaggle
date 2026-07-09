@echo off
chcp 65001 >nul
cd /d "C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle"

echo ============================================================
echo  BEKLE-VE-BASLAT: qaug bitmesini + AI dosyasini bekliyor
echo  Baslama: %date% %time%
echo ============================================================

REM ===== ADIM 1: AI genisletme dosyasinin hazir olmasini bekle =====
:WAIT_AI
python -X utf8 -c "import pandas as pd,sys; d=pd.read_csv(r'C:\Users\ASUS\Desktop\trendyol\ai_expanded_terms.csv'); n=len(d); print(f'[AI] {n}/50153 ({n/50153*100:.1f}%%)'); sys.exit(0 if n>=49000 else 1)" 2>nul
if errorlevel 1 (
    echo   [%time%] AI henuz bitmedi, 3 dk bekliyorum...
    timeout /t 180 /nobreak >nul
    goto WAIT_AI
)
echo [%time%] AI genisletme TAMAM!

REM ===== ADIM 2: qaug GPU egitiminin bitmesini bekle =====
echo [%time%] qaug egitiminin bitmesini bekliyorum (ce_test_trendyol_qaug.npy)...
:WAIT_QAUG
if exist "C:\Users\ASUS\Desktop\trendyol\artifacts\ce_test_trendyol_qaug.npy" (
    echo [%time%] qaug BITTI! GPU bos.
    goto START_AI_TRAIN
)
echo   [%time%] qaug devam ediyor, 2 dk bekliyorum...
timeout /t 120 /nobreak >nul
goto WAIT_QAUG

:START_AI_TRAIN
REM ===== ADIM 3: trendyol_ai egitimi (INTERNET GEREKMEZ) =====
echo.
echo ============================================================
echo [%time%] trendyol_ai egitimi BASLIYOR (ai_expanded_terms.csv)
echo   Internet GEREKMEZ - model lokal, veri lokal.
echo ============================================================
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_ai --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file ai_expanded_terms.csv
echo [%time%] trendyol_ai egitimi BITTI!

REM ===== ADIM 4: submission uret =====
echo [%time%] Submission'lar uretiliyor...
python -X utf8 -c "import numpy as np,pandas as pd,os;DATA=r'C:\Users\ASUS\Desktop\trendyol';ART=os.path.join(DATA,'artifacts');score=np.load(os.path.join(ART,'ce_test_trendyol_ai.npy'));sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'));[pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_ai_p{int(ppr*100)}.csv'),index=False) or print(f'  ai PPR{ppr}: pos={((score>=np.quantile(score,1-ppr)).mean()):.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 5: buyuk ensemble =====
echo [%time%] Ensemble uretiliyor...
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
if len(models)<2: print('Yetersiz model'); exit()
ens=sum(models.values())/len(models)
sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv'))
for ppr in [0.30,0.33,0.36]:
    thr=np.quantile(ens,1-ppr);pred=(ens>=thr).astype(int)
    out=os.path.join(DATA,f'submission_ens{len(models)}_p{int(ppr*100)}.csv')
    pd.DataFrame({'id':sub['id'],'prediction':pred}).to_csv(out,index=False)
    print(f'  ens{len(models)} PPR{ppr}: {pred.mean():.4f} -> {os.path.basename(out)}')
"

echo.
echo ============================================================
echo  HERSEY BITTI! %date% %time%
echo  Sabah gonder:
echo    submission_ai_p33.csv
echo    submission_ens*_p33.csv
echo ============================================================
pause
