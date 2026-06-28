# DEVİR NOTU — Trendyol Datathon 2026 (Kaggle aşaması)

> Çalışmayı devralacak kişi için. Önce bu dosyayı ve `tarama.md`'yi oku. En altta yeni bir Claude Code oturumuna yapıştırılabilecek hazır prompt var.

---

## 1. ŞU AN NEREDEYİZ (29 Haziran gecesi sonu)

- **Görev:** (arama terimi, ürün) çifti için **binary alaka**. Metrik: **macro-F1**. Submission: `id,prediction` (0/1).
- **EN İYİ LB (public) = 0.80** → `submission_easyCE_p33.csv` (= **kolay-negatif Cross-Encoder TEK BAŞINA**, %33 pozitife kalibre). Final seçimine işaretli olmalı.
- **LB geçmişi (yön bunlardan çıktı):**
  | Submission | pos_rate | LB |
  |-----------|----------|-----|
  | Eski GBDT+CE kolay blend | 0.327 | 0.76 |
  | Zor-negatif blend | 0.14 / 0.47 | 0.64 / 0.68 |
  | all-ones (prevalans probu) | 1.0 | **0.41** |
  | **Kolay CE TEK BAŞINA** | **0.33** | **0.80** ✅ |

## 2. 🔴 BU GECENİN ÜÇ KRİTİK BULGUSU

1. **TEST %69.5 POZİTİF (çoğunluk).** all-ones=0.41 → P=0.41/(1−0.41)=0.695. Eğitimimiz ~%18 pozitifti → model "pozitif nadir" sanıp az tahmin ediyor. **Doğru kalibrasyon (~%33 pozitif tahmin) tek başına 0.76→0.80 sıçramasının yarısı.** Optimal tahmin oranı ~%33-40 (daha fazlası DÜŞÜRÜYOR: 0.47→0.68; macro-F1 azınlık negatif sınıfı eşit ağırlıkladığı için %70'e çıkma).
2. **ZOR NEGATİFLER LB'DE İŞE YARAMADI.** Embedding-ANN hard negatif + denoising re-mining → LB 0.64-0.68, kolay negatiften (0.76-0.80) DAHA KÖTÜ. Sebep: test pozitif-çoğunluk; benzer ürünler çoğunlukla ALAKALI, ama biz modele "benzer=alakasız" öğrettik → önyargılı. **Offline harness (diagnose/holdout) bizi yanılttı çünkü pozitif-nadir varsaydı.** Holdout skorlarına GÜVENME; sadece LB.
3. **CROSS-ENCODER YILDIZ, GBDT SEYRELTİYOR.** Kolay CE tek başına (0.80) > GBDT+CE blend (0.76). GBDT'yi at ya da çok düşük ağırlık ver.

➡️ **DOĞRU YÖN: CE merkezli, kolay/temsili negatif, doğru kalibrasyon, CE-only ensemble.** (Zor negatif DEĞİL.)

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

## 5. YARIN — DOĞRU YÖN (taze 5 submission)

> Prevalans (0.695) ve "CE-only + kolay negatif" yönü artık LB ile KANITLI. Zor-negatif yönüne DÖNME.

1. **🔴 Kolay CE'yi yeniden üret + optimal oranı bul.** En iyimiz `submission_easyCE_p33.csv` (0.80) = `ce_model_berturk` (KOLAY negatiflerle eğitilen ilk CE) tek başına, %33 pozitif. Onu %28 ve %38'de gönder → optimal oranı haritala (belki >0.80).
   - Not: kolay CE skorları `artifacts/_stale_easy/ce_test_berturk.npy`'de. Kolay negatif üretimi: `fast_submit.py`'ı **USE_EXISTING_PAIRS olmadan** çalıştır (kendi rastgele+TF-IDF negatiflerini deterministik üretir, seed 42).
2. **🔴 CE-only ensemble.** `train_cross_encoder.py` ile KOLAY negatif `train_pairs` üzerinde 2-3 güçlü CE eğit (farklı seed/epoch/max_len; istersen XLM-R / ConvBERT-tr çeşitliliği). **GBDT'yi blend'e KATMA** (seyreltiyor). Test skorlarını ortalayıp ~%33'e kalibre et.
3. **🟠 Daha güçlü tek CE:** kolay neg + zengin girdi (build_docs rich) + 3 epoch.
4. **Kalibrasyon her zaman:** `calibrate_submit.py` ile hedef pos_rate ~0.30-0.40 arası üret. Holdout'a GÜVENME (yanıltıyor), sadece LB eğrisine (0.327→0.76, 0.33→0.80, 0.47→0.68).
5. **2 final:** en iyi tek CE + en iyi CE-ensemble.

**Denenip ELENENLER (tekrarlama):** zor/embedding-ANN negatif, denoising re-mining, GBDT'yi blend'e katmak, holdout skoruna güvenmek, prevalansı düşük (~%13) sanmak.

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

Kanıtlanmış durum (LB ile): EN İYİ = kolay-negatif Cross-Encoder TEK BAŞINA,
%33 pozitife kalibre -> LB 0.80. Test %69.5 POZİTİF (all-ones probu). Zor negatif
ve GBDT'yi blend'e katmak DENENDİ, LB'de KÖTÜ (HANDOFF §2). Holdout YANILTIYOR,
sadece LB'ye güven.

Bugün (taze 5 submission), HANDOFF §5'i uygula:
1. emb/ yoksa python gen_embeddings.py. Kolay negatif train_pairs için
   fast_submit.py'ı USE_EXISTING_PAIRS OLMADAN çalıştır.
2. train_cross_encoder.py ile KOLAY negatif üzerinde 2-3 güçlü CE eğit
   (farklı seed/epoch; GBDT'yi blend'e KATMA). score_test.py ile test skorla.
3. calibrate_submit.py ile pos_rate ~0.30-0.40 arası submission üret, gönder,
   optimal oranı bul. CE-only ensemble'ı kalibre et.
GPU var (RTX 5070 Ti, 12.8GB). Her adımda önce kısa plan söyle, sonra uygula.
```
