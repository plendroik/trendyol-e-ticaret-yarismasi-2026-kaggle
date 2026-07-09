@echo off
chcp 65001 >nul
cd /d "C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle"

echo ============================================================
echo  GECE CHAIN BASLADI: %date% %time%
echo  Plan: qaug egitimi -> AI bitmesini bekle -> trendyol_ai egitimi
echo ============================================================

REM ===== ADIM 1: qexp submission'larini uret (test skoru zaten hazir olmali) =====
echo.
echo [%time%] ADIM 1: qexp submission'lari uretiliyor...
python -X utf8 -c "import numpy as np, pandas as pd, os; DATA=r'C:\Users\ASUS\Desktop\trendyol'; ART=os.path.join(DATA,'artifacts'); f=os.path.join(ART,'ce_test_trendyol_qexp.npy'); assert os.path.exists(f), 'qexp test skoru yok!'; score=np.load(f); sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv')); [pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_qexp_p{int(ppr*100)}.csv'),index=False) or print(f'  qexp PPR{ppr}: pos_rate={(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"
echo [%time%] qexp submission'lar HAZIR.

REM ===== ADIM 2: qaug egitimi (sentetik augmentasyon) ~2.5-3 saat =====
echo.
echo [%time%] ADIM 2: qaug egitimi basliyor (augmented_terms + augmented_pairs)...
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_qaug --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file augmented_terms.csv --pairs_file train_pairs_augmented.parquet
echo [%time%] qaug egitimi BITTI.

REM qaug submission'lari uret
echo [%time%] qaug submission'lari uretiliyor...
python -X utf8 -c "import numpy as np, pandas as pd, os; DATA=r'C:\Users\ASUS\Desktop\trendyol'; ART=os.path.join(DATA,'artifacts'); score=np.load(os.path.join(ART,'ce_test_trendyol_qaug.npy')); sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv')); [pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_qaug_p{int(ppr*100)}.csv'),index=False) or print(f'  qaug PPR{ppr}: pos_rate={(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 3: AI genisletme bitmesini bekle =====
echo.
echo [%time%] ADIM 3: ai_expanded_terms.csv bitmesini bekliyorum...
:WAIT_AI
python -X utf8 -c "import pandas as pd, sys; d=pd.read_csv(r'C:\Users\ASUS\Desktop\trendyol\ai_expanded_terms.csv'); n=len(d); print(f'  AI expansion: {n}/50153 ({n/50153*100:.1f}%%)'); sys.exit(0 if n>=49000 else 1)"
if errorlevel 1 (
    echo   Henuz bitmedi, 3 dk bekliyorum...
    timeout /t 180 /nobreak >nul
    goto WAIT_AI
)
echo [%time%] AI genisletme TAMAMLANDI!

REM ===== ADIM 4: trendyol_ai egitimi (AI genisletilmis sorgularla) ~3.5 saat =====
echo.
echo [%time%] ADIM 4: trendyol_ai egitimi basliyor (ai_expanded_terms.csv)...
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_ai --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs --terms_file ai_expanded_terms.csv
echo [%time%] trendyol_ai egitimi BITTI.

REM trendyol_ai submission'lari uret
echo [%time%] trendyol_ai submission'lari uretiliyor...
python -X utf8 -c "import numpy as np, pandas as pd, os; DATA=r'C:\Users\ASUS\Desktop\trendyol'; ART=os.path.join(DATA,'artifacts'); score=np.load(os.path.join(ART,'ce_test_trendyol_ai.npy')); sub=pd.read_csv(os.path.join(DATA,'submission_pairs.csv')); [pd.DataFrame({'id':sub['id'],'prediction':(score>=np.quantile(score,1-ppr)).astype(int)}).to_csv(os.path.join(DATA,f'submission_ai_p{int(ppr*100)}.csv'),index=False) or print(f'  ai PPR{ppr}: pos_rate={(score>=np.quantile(score,1-ppr)).mean():.4f}') for ppr in [0.30,0.33,0.36]]"

REM ===== ADIM 5: 4'lu ensemble (trendyol_ce + full + qexp + ai + berturk) =====
echo.
echo [%time%] ADIM 5: Ensemble submission'lar uretiliyor...
python -X utf8 -c "
import numpy as np, pandas as pd, os
DATA=r'C:\Users\ASUS\Desktop\trendyol'; ART=os.path.join(DATA,'artifacts')
def rn(x):
    o=np.argsort(x,kind='stable'); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r/(len(x)-1)
models = {}
for name in ['trendyol_ce','trendyol_full','trendyol_qexp','trendyol_qaug','trendyol_ai']:
    p = os.path.join(ART, f'ce_test_{name}.npy')
    if os.path.exists(p):
        models[name] = rn(np.load(p))
        print(f'  loaded {name}')
berturk_p = os.path.join(ART,'_stale_easy','ce_test_berturk.npy')
if os.path.exists(berturk_p):
    models['berturk'] = rn(np.load(berturk_p))
    print(f'  loaded berturk')
if len(models) < 2:
    print('Yetersiz model, ensemble atlanıyor'); exit()
ens = sum(models.values()) / len(models)
sub = pd.read_csv(os.path.join(DATA,'submission_pairs.csv'))
for ppr in [0.30,0.33,0.36]:
    thr=np.quantile(ens,1-ppr); pred=(ens>=thr).astype(int)
    out=os.path.join(DATA,f'submission_ensemble{len(models)}_p{int(ppr*100)}.csv')
    pd.DataFrame({'id':sub['id'],'prediction':pred}).to_csv(out,index=False)
    print(f'  ensemble({len(models)}) PPR{ppr}: pos_rate={pred.mean():.4f} -> {os.path.basename(out)}')
print(f'Modeller: {list(models.keys())}')
"

echo.
echo ============================================================
echo  GECE CHAIN TAMAMLANDI: %date% %time%
echo  Sabah kontrol et:
echo    submission_qexp_p33.csv     (domain + TF-IDF genisletme)
echo    submission_qaug_p33.csv     (domain + augmentasyon)
echo    submission_ai_p33.csv       (domain + AI genisletme)
echo    submission_ensemble*_p33.csv (tum modellerin ensemble'i)
echo  Hepsi: C:\Users\ASUS\Desktop\trendyol\
echo ============================================================
pause
