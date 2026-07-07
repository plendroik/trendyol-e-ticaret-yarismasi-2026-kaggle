# Etiketli Veri (LLM-hakem etiketleri)

> ⚠️ **SADECE TAKIM İÇİ.** Bu dosyalar yarışma verisinden türetilmiştir; takım dışına
> paylaşmak kural ihlalidir (diskalifiye). Repo private kalmalı.

## Dosyalar

### `test_judge_labels.csv` — ANA DOSYA (id, label)
Test (submission_pairs) çiftlerinin gpt-4o-mini **tek-çift** hakem etiketleri.
- Kapsam: ana modelin (`ce_test_trendyol_ce.npy`) skor bantları **[0.03–0.97]**
  (0.92-0.97 kısmı eklenmeye devam ediyor), ~525k+ çift.
- `id` = submission_pairs.csv'deki id. `label` = 1 alakalı / 0 alakasız.
- Hakem kalitesi (bilinen pozitifler + rastgele negatiflerle ölçüldü):
  **RECALL 0.94 / FPR 0.06.** Prompt: `judge_test_band.py` içindeki `SYS`.
- ÖNEMLİ: tek-çift formatı şart — batch'li sorgu kaliteyi bozuyor (ölçüldü).

### `gpt4o_labels.csv` (id, label)
Mini-hakem ile ana modelin en sert çatıştığı 25k çiftin **gpt-4.1-mini**
(RECALL 0.96 / FPR 0.05) ikinci-görüş etiketleri. Çakışmada bunlar öncelikli
kullanılabilir; LB'de tek başına fark yaratmadı (v5=0.869 vs v2=0.869).

### `llm_labels.csv` (term_id, item_id, label)
Eski, EĞİTİM-sorgusu tarafı etiketler (embedding-ANN adayları, batch'li format).
GÜRÜLTÜLÜ — bununla eğitilen model 0.79 verdi. Sadece referans için duruyor;
kullanman önerilmez.

## Ne işe yaradı (LB kanıtı)
- Hibrit (model emin uçlar + bantta hakem): 0.834 → 0.869 → 0.873
- + damıtma (bu etiketlerle domain CE eğitip uçları temizleme): **0.896**
- Damıtma tarifi: `gen_distill_trainset.py` + `train_cross_encoder.py --pairs_file
  train_pairs_distill.parquet` (suffix `distill`).

Rescue etiketleri (derin-negatif kurtarma, ~68k) tamamlanınca buraya eklenecek.
