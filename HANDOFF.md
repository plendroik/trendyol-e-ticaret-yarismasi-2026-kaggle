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
  | prob (muhtemelen all-ZEROS → π≈%30) | — | 0.41 |
  | **Kolay CE TEK BAŞINA** | **0.33** | **0.80** ✅ |
  | rich+FGM ensemble (ens2) | 0.31 | 0.78 |
  | **BERTurk+mDeBERTa ensemble** | 0.33 | **0.80** (fark yok → veri darboğaz) |

## 2. 🔴 KRİTİK BULGULAR (LB ile kanıtlı)

1. **TEST PREVALANSI ~%30 POZİTİF** (önceki "%69.5" YANLIŞTI). O 0.41'lik prob muhtemelen **all-ZEROS**'tı (all-ones değil): %30 prevalansta all-zeros = (1−π)/(2−π) = 0.41. **Kesin kanıt:** kolay CE %33 tahminle 0.80 aldı; eğer test %69.5 pozitif olsaydı %33 tahminde mükemmel model bile en fazla **0.635** alabilirdi → imkânsız. Yani π≈%30. **Optimal tahmin oranı (PPR) ~%30-33** (0.33→0.80 kanıtlı; daha düşük/yüksek kötü: ens@0.31→0.78, hard@0.47→0.68).
2. **MODEL DARBOĞAZ DEĞİL — VERİ DARBOĞAZ (29 Haz akşamı kanıtı).** BERTurk(0.80) ≈ mDeBERTa(holdout aynı) ≈ **BERTurk+mDeBERTa ensemble @0.33 = 0.80.** İki FARKLI mimari aynı 0.80'i veriyor → **aynı hataları yapıyorlar** → tavan modelden değil, **sentetik negatif/etiket** kalitesinden. rich girdi + FGM de geçemedi (ens@0.31→0.78). Daha büyük/farklı model BOŞA — veriye yüklen.
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

## 5. YARIN — VERİYE YÜKLEN (model değil!)

> §2.2 KANITLI: model/mimari/ensemble değiştirmek 0.80'de tıkanıyor. Darboğaz **sentetik negatif/etiket kalitesi.** Yeni model eğitmek yerine VERİYİ iyileştir. Kalibrasyon ~%33 sabit, CE-only.

1. **🔴 PSEUDO-LABELING (en güçlü kaldıraç — KDD Cup kazananı).** Ensemble test skorlarını (`ce_test_berturk` + `ce_test_mdeberta_easy`, rank-avg) al → çok emin olanları pseudo-etiketle (skor üst ~%10 → pozitif, alt ~%40 → negatif; orta belirsiz kısmı AT). Bunları kolay-negatif `train_pairs`'e EKLE → CE'yi yeniden eğit. Bu, **gerçek test sorgularını (cold-start!) ve gerçek test negatif dağılımını** modele verir → sentetik-negatif tavanını kırabilir. Güven eşiklerini iterate et.
2. **🟠 TF-IDF / LLM sorgu genişletme.** Kısa-gürültülü sorguları zenginleştir (KDD Cup "day-day-up": sorgunun pozitif ürünlerinden top TF-IDF kelimeler). DİKKAT: test sorgularını da tutarlı genişlet (cold-start → pozitif yok; sorgunun 104 adayının başlıklarından TF-IDF ile genişlet ya da pseudo-relevance feedback).
3. **🟠 Daha temsili negatif.** Test negatifleri leksikal/kategori-orta-ilişkili. "Kök-kategori dışı rastgele" (weak categorical) veya in-batch negatif dene. Negatif RECİPESİNİ değiştir, modeli değil.
4. **Kalibrasyon:** her submission'da `calibrate_submit.py` / `ensemble_submit.py` ile PPR ~%33. LB eğrisi: 0.327→0.76, 0.33→0.80, 0.47→0.68, 0.31→0.78. Holdout YANILTIYOR — kullanma.
5. **2 final:** 0.80 (`submission_easyCE_p33`) güvenli + yarınki en iyi.

**Denenip ELENENLER (TEKRARLAMA):** zor/embedding-ANN negatif, denoising re-mining, GBDT blend, rich girdi+FGM, farklı/daha-büyük backbone (BERTurk≈mDeBERTa≈ensemble=0.80), holdout skoruna güvenmek, prevalansı %13/%69.5 sanmak (gerçek ~%30).

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
%33 pozitife kalibre -> LB 0.80. Test prevalansı ~%30 pozitif (NOT 0.695). KANIT:
model/mimari/ensemble değiştirmek 0.80'de tıkanıyor (BERTurk≈mDeBERTa≈ensemble=0.80)
-> darboğaz MODEL DEĞİL, sentetik negatif/etiket VERİSİ. Holdout YANILTIYOR, sadece LB.

Bugün VERİYE yüklen (HANDOFF §5), yeni model eğitme:
1. PSEUDO-LABELING: ensemble test skorlarını (artifacts/ce_test_berturk +
   ce_test_mdeberta_easy, rank-avg) al; üst ~%10 pozitif, alt ~%40 negatif pseudo-
   etiketle (orta belirsizi at); kolay train_pairs'e ekle; CE'yi yeniden eğit.
   Bu cold-start test sorgularını + gerçek negatif dağılımını verir.
2. TF-IDF/LLM sorgu genişletme; daha temsili negatif recipe (model değil veri).
3. Her submission'da calibrate_submit.py ile PPR ~%33. Holdout'a GÜVENME.
GPU var (RTX 5070 Ti, 12.8GB). Her adımda önce kısa plan söyle, sonra uygula.
```
