@echo off
REM ===== Gece eğitimi: dolu domain CE + submission'lar (sabaha hazir) =====
cd /d "C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle"

echo [%date% %time%] Dolu domain CE egitimi basliyor (~5-6 saat)...
python -X utf8 train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_full --epochs 3 --max_len 256 --train_batch 24 --infer_batch 64 --max_train 0

echo [%date% %time%] Egitim bitti. Submission'lar uretiliyor...
python -X utf8 make_overnight_subs.py

echo.
echo ============================================================
echo  TAMAMLANDI. Submission'lar hazir:
echo    submission_trendyolfull_p33.csv   (tek domain model)
echo    submission_ensTF_p33.csv          (domain + BERTurk ensemble)
echo  Konum: C:\Users\ASUS\Desktop\trendyol\
echo ============================================================
pause
