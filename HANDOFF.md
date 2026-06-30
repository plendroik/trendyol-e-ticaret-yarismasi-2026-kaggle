# DEVİR NOTU — Trendyol Datathon 2026 (Kaggle aşaması)

> Çalışmayı devralacak kişi için. Önce bu dosyayı ve `tarama.md`'yi oku. En altta yeni bir Claude Code oturumuna yapıştırılabilecek hazır prompt var.

---

## 1. ŞU AN NEREDEYİZ (29 Haziran gecesi sonu)

- **Görev:** (arama terimi, ürün) çifti için **binary alaka**. Metrik: **macro-F1**. Submission: `id,prediction` (0/1).
- **EN İYİ LB (public) = 0.83** → `submission_trendyol_p33.csv` (= **Trendyol e-ticaret DOMAIN modeli** cross-encoder olarak, kolay neg, %33 kalibre). 0.80'i kıran tek şey DOMAIN modeli oldu. Yedek final: `submission_easyCE_p33.csv` (0.80).
- **LB geçmişi (yön bunlardan çıktı):**
  | Submission | pos_rate | LB |
  |-----------|----------|-----|
  | Eski GBDT+CE kolay blend | 0.327 | 0.76 |
  | Zor-negatif blend | 0.14 / 0.47 | 0.64 / 0.68 |
  | prob (muhtemelen all-ZEROS → π≈%30) | — | 0.41 |
  | **Kolay CE TEK BAŞINA** | **0.33** | **0.80** ✅ |
  | rich+FGM ensemble (ens2) | 0.31 | 0.78 |
  | **BERTurk+mDeBERTa ensemble** | 0.33 | **0.80** (fark yok → veri darboğaz) |

## 2. 🔴 KRİTİK BULGULAR (LB ile kanıtlı)

1. **TEST PREVALANSI ~%30 POZİTİF** (önceki "%69.5" YANLIŞTI). O 0.41'lik prob muhtemelen **all-ZEROS**'tı (all-ones değil): %30 prevalansta all-zeros = (1−π)/(2−π) = 0.41. **Kesin kanıt:** kolay CE %33 tahminle 0.80 aldı; eğer test %69.5 pozitif olsaydı %33 tahminde mükemmel model bile en fazla **0.635** alabilirdi → imkânsız. Yani π≈%30. **Optimal tahmin oranı (PPR) ~%30-33** (0.33→0.80 kanıtlı; daha düşük/yüksek kötü: ens@0.31→0.78, hard@0.47→0.68).
2. **🎯 GENEL MODEL DARBOĞAZ, DOMAIN MODELİ KIRDI (30 Haz kanıtı).** GENEL modeller hep 0.80: BERTurk=0.80, mDeBERTa≈0.80, ensemble=0.80, rich+FGM=0.78, pseudo=0.79, per-query=0.57. AMA **Trendyol e-ticaret DOMAIN modeli (cross-encoder) = 0.83** (eksik eğitimli haliyle bile!). → Tavan genel-model kapasitesindendi; **domain ön-eğitimi aynı veriden daha çok sinyal çıkarıyor.** KALDIRAÇ = domain model, daha büyük/genel model DEĞİL. Trendyol modeli: `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`, `trust_remote_code=True` ile `AutoModelForSequenceClassification`.
3. **ZOR NEGATİFLER LB'DE İŞE YARAMADI.** Embedding-ANN hard + denoising → 0.64-0.68 < kolay 0.80. Test negatifleri embedding-en-yakın değil, leksikal/kategori-orta-ilişkili (rastgele+TF-IDF onları daha iyi yakalıyor).
4. **CROSS-ENCODER YILDIZ, GBDT SEYRELTİYOR.** Kolay CE tek (0.80) > GBDT+CE blend (0.76). GBDT KATMA.
5. **OFFLINE HOLDOUT YANILTIYOR (+0.10 iyimser, reweight edilse bile).** Model kıyaslamak için bile güvenilmez. **SADECE LB'ye güven.**

➡️ **DOĞRU YÖN (29 Haz sonu): model değiştirmeyi BIRAK. VERİYE yüklen** → pseudo-labeling (test pseudo-etiketleri ile retrain, cold-start'a birebir), TF-IDF/LLM sorgu genişletme, daha temsili negatif. Kalibrasyon ~%33 sabit, CE-only.

## 3. KRİTİK VERİ GERÇEKLERİ

- **TAM COLD-START:** Eğitim (17.968) ve test (32.185) terimleri **HİÇ örtüşmüyor.** Semantik (embedding/cross-encoder) ezberden çok daha önemli. `GroupKFold(term_id)` şart (kullanılıyor; holdout = fold 4).
- **Bütçe:** günde 5 submission, finalde 2 seçim. LB'de hill-climbing yapma; lokal CV'ye güven.
- **Kolonlar:** items=`item_id,title,category,brand,gender,age_group,attributes` · terms=`term_id,query` · training_pairs=`id,term_id,item_id,label`(hepsi 1) · submission_pairs=`id,term_id,item_id`.

## 4. DOSYALAR

| Dosya | Ne yapar |
|------|----------|
| `fast_submit.py` | GBDT pipeline: Türkçe normalize → TF-IDF hard negatif → leksikal/yapısal + `emb_cos` feature → 5-fold LGBM+CatBoost → `submission.csv`. **Artifact kaydeder** (`artifacts/`): `train_pairs.parquet` (term_id,item_id,label,fold), `gbdt_oof.npy`, `gbdt_test.npy`. ~5 dk. |
| `gen_embeddings.py` | Trendyol embedding modeliyle vektörler (checkpoint'li, GPU ~25 dk). `emb/` içine yazar. |
| `train_cross_encoder.py` | **Parametrik cross-encoder** (HF + manuel PyTorch, AMP). `train_pairs.parquet`'i okur (aynı negatif/fold). Yapısal alanları serialize eder. `--model --suffix --epochs --max_len ...`. Çıktı: `artifacts/ce_holdout_<suffix>.npy`, `ce_test_<suffix>.npy`, `ce_model_<suffix>/`. BERTurk ~45 dk. |
| `blend.py` | Tüm `ce_*` + GBDT skorlarını toplar, stacking + threshold. (DİKKAT: holdout yanıltıyor — bkz §2.) |
| `score_test.py` | Kaydedilmiş bir CE modelini test setinde skorlar (ayrı temiz süreç, küçük batch → VRAM güvenli). `--ce_model --suffix --basic_docs`. |
| `calibrate_submit.py` | Test skorlarını hedef pos_rate'lere kalibre edip submission yazar (en önemli araç — prevalans!). |
| `diagnose_hard_val.py` | (ARŞİV) zor-eval harness'i — pozitif-nadir varsaydığı için YANILTTI. Kullanma. |
| `gen_hard_negatives.py`, `remine_negatives.py`, `calibrate_by_prevalence.py` | (ARŞİV) zor-negatif yönü — LB'de başarısız oldu. |
| `tarama.md` | Strateji playbook'u. NOT: "test reranking/zor negatif" varsayımı LB'de YANLIŞ çıktı (test %70 pozitif). |
| `src/` | Eski (yiğit) polars pipeline. Referans. |

> ⚠️ `artifacts/`, `emb/`, `*.csv`, `*.npy` gitignore'da — repoda yok. Devralan kişi `gen_embeddings.py` + `fast_submit.py` çalıştırıp yeniden üretir.

## 5. YARIN — DOMAIN MODELİNİ MAKSİMİZE ET (kaldıraç bu!)

> 0.83 domain CE EKSİK eğitildi (600k/760k satır, batch 32, max_len 160, basic girdi, 2 epoch). Bolca yer var. Komut bu 0.83'ü verdi:
> `train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_ce --epochs 2 --max_len 160 --train_batch 32 --infer_batch 96 --max_train 600000 --basic_docs`

1. **🟢 PPR check (BEDAVA, retrain yok).** `submission_trendyol_p30.csv` ve `p36.csv` HAZIR diskte → gönder, domain modelinin optimal PPR'sini bul (0.33→0.83; belki 0.30/0.36 daha iyi). #2 GPU'da paralel dönerken yap.
2. **🔴 Domain CE'yi DOLU eğit.** Aynı model ama: `--max_train 0` (tüm veri), **`--max_len 256`** (gte uzun bağlamı verimli, ürün metni/öznitelik sığar), **rich girdi** (--basic_docs YOK), 3 epoch, istersen `--fgm`. ~2-3 saat. Beklenti 0.84-0.85.
3. **🔴 Ensemble: Trendyol CE + BERTurk.** Artık GERÇEK çeşitlilik (domain+genel). `ensemble_submit.py` ile `ce_test_trendyol_ce` + `_stale_easy/ce_test_berturk` rank-avg, PPR ~%33. (Önceki ensemble'lar genel+genel olduğu için fayda etmedi.)
4. **🟢 2. Trendyol seed** → 3-model domain ensemble.
5. **2 final:** 0.83 (`submission_trendyol_p33`) güvenli + yarınki en iyi. **Hedef 0.85-0.86.**

**ELENENLER (TEKRARLAMA):** zor/embedding-ANN negatif, denoising, GBDT blend, GENEL modeller (BERTurk/mDeBERTa ~0.80), rich+FGM (genel modelde 0.78), pseudo-labeling (0.79), per-query eşik (0.57), prevalansı %13/%69.5 sanmak (gerçek ~%30). Kalibrasyon hep PPR ~%33; holdout YANILTIYOR, sadece LB.

## 6. ORTAM

- Python 3.11, GPU RTX 5070 Ti Laptop (12.8 GB). Kurulu: numpy, pandas, sklearn, scipy, lightgbm, catboost, torch(cu128), sentence-transformers, transformers 4.57, sentencepiece. **`polars` YOK** (fast_submit pandas tabanlı).
- Embedding modeli `trust_remote_code=True` ister.
- Loglar UTF-16 (`*>`). Oku: `iconv -f UTF-16LE -t UTF-8 x.log` ya da PowerShell `Get-Content -Encoding Unicode`.
- mDeBERTa batch 64 max_len 160'ta VRAM'i doldurup throttle yaptı (adım süresi katlandı). Ağır modellerde batch'i küçült + gradient accumulation.

---

## 7. YENİ CLAUDE OTURUMU İÇİN PROMPT

```
Trendyol Datathon 2026 Kaggle (arama terimi–ürün alaka, binary, macro-F1).
Repo: trendyol-e-ticaret-yarismasi-2026-kaggle. Önce HANDOFF.md §1-2-5 oku.

Kanıtlanmış durum (LB ile): EN İYİ = LB 0.83 = Trendyol e-ticaret DOMAIN modeli
(Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) cross-encoder olarak, %33 kalibre.
KALDIRAÇ = DOMAIN modeli: GENEL modeller (BERTurk/mDeBERTa/ensemble) hep 0.80'de
tıkandı, domain modeli 0.83 (eksik eğitimli haliyle bile). Test prevalansı ~%30.
Holdout YANILTIYOR, sadece LB. Kalibrasyon hep PPR ~%33.

Bugün domain modelini MAKSİMİZE et (HANDOFF §5):
1. (bedava) submission_trendyol_p30/p33/p36.csv HAZIR -> PPR optimumunu bul.
2. Domain CE'yi DOLU eğit: train_cross_encoder.py --model
   "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix trendyol_full
   --epochs 3 --max_len 256 --train_batch 32 --infer_batch 96 --max_train 0 (rich
   girdi, --basic_docs YOK; istersen --fgm). score sonrası calibrate ~%33.
3. Ensemble: ce_test_trendyol_* + _stale_easy/ce_test_berturk (ensemble_submit.py),
   domain+genel gerçek çeşitlilik. Hedef 0.85-0.86.
GPU var (RTX 5070 Ti, 12.8GB). Her adımda önce kısa plan söyle, sonra uygula.
```
