# DEVİR NOTU — Trendyol Datathon 2026 (Kaggle aşaması)

> Çalışmayı devralacak kişi için. Önce bu dosyayı ve `tarama.md`'yi oku. En altta yeni bir Claude Code oturumuna yapıştırılabilecek hazır prompt var.

---

## 1. ŞU AN NEREDEYİZ (güncel)

- **Görev:** (arama terimi, ürün) çifti için **binary alaka** (1=alakalı, 0=alakasız). Metrik: **macro-F1**. Submission: `id,prediction` (0/1).
- **Eğittiğimiz modeller ve HOLDOUT macro-F1 (fold-4, görülmemiş terimler):**
  | Model | Holdout macro-F1 |
  |------|------------------|
  | Leksikal-only GBDT | 0.831 |
  | GBDT + `emb_cos` (Trendyol embedding cosine) | 0.874 |
  | Cross-encoder BERTurk (tek) | 0.900 |
  | **Stacked blend (GBDT + BERTurk)** | **0.906** |
- **GERÇEK LB (public):** `submission_blend.csv` → **0.76**. (Arkadaşın eski en iyisi 0.75.)

## 2. 🔴 EN ÖNEMLİ BULGU — holdout 0.906 ama LB 0.76 (0.15 uçurum)

**Bu bir bug değil, dağılım uyumsuzluğu.** Kök sebep: **eğitim negatiflerimiz çok kolay.**
- Biz rastgele + TF-IDF orta-sıra negatif kullandık → model bunları kolay ayırıyor → holdout 0.90.
- Ama test bir **reranking** seti: her sorgu için ~104 aday (Trendyol arama motorunun getirdiği, sorguyla zaten benzeşen ürünler). Testteki "alakasız"lar **zor negatif.**
- Model hiç zor negatif görmedi → testte alakasızları reddedemiyor → 0.76. Holdout serap. Arkadaşın da aynı duvardan 0.75'te takıldı.

➡️ **Çözüm = negatifleri teste benzet:** embedding-ANN ile her sorgunun en yakın ürünlerinden hard negatif üret + false-negative filtresi. Bu hem modeli gerçekten öğretir hem threshold'u doğru kalibre eder. **Asıl 0.76→0.83+ sıçraması burada.** (Detay: `tarama.md` §2.)

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
| `blend.py` | Tüm `ce_*` + GBDT skorlarını otomatik toplar, holdout'u A/B'ye bölüp stacking (LogisticRegression) + threshold kalibre eder. Çıktı: `submission_blend.csv` (stacked), `submission_robust.csv` (en iyi tek model). |
| `tarama.md` | Strateji playbook'u. Oku. |
| `src/` | Eski (yiğit) polars pipeline. Referans; `fast_submit.py` daha temiz. |

> ⚠️ `artifacts/`, `emb/`, `*.csv`, `*.npy` gitignore'da — repoda yok. Devralan kişi `gen_embeddings.py` + `fast_submit.py` çalıştırıp yeniden üretir.

## 5. SIRADAKİ İŞ (öncelik sırası) — ASIL DEĞER

1. **🔴 Zor negatifler (embedding-ANN) + retrain.** `emb/item_emb.npy` + `query_emb.npy` hazır. Her train sorgusu için en yakın ~150 ürünü çek (brute-force GPU matmul ya da FAISS), pozitifleri çıkar, rank ~5-150'yi hard negatif yap, false-negative filtresi (emb-sim > 0.95×max_pozitif_sim olanı at — NV-Retriever). Yeni `train_pairs.parquet` yaz → `fast_submit.py` (GBDT) + `train_cross_encoder.py` yeniden çalıştır → `blend.py`. **0.76 duvarını bu kırar.**
2. **🔴 LB prevalans probe.** 1 submission `all-ones` → macro-F1 formülünden test pozitif oranını çöz → threshold'u o orana göre kilitle. (`tarama.md` §5.)
3. **🟠 Daha çok backbone:** mDeBERTa-v3 (DİKKAT: bu GPU'da batch 64'te yavaş/throttle — batch 16-24 + grad-accum kullan), XLM-R, Türkçe ELECTRA. Çeşitlilik + seed.
4. **🟢 2 final submission:** biri robust tek model, biri max blend.

## 6. ORTAM

- Python 3.11, GPU RTX 5070 Ti Laptop (12.8 GB). Kurulu: numpy, pandas, sklearn, scipy, lightgbm, catboost, torch(cu128), sentence-transformers, transformers 4.57, sentencepiece. **`polars` YOK** (fast_submit pandas tabanlı).
- Embedding modeli `trust_remote_code=True` ister.
- Loglar UTF-16 (`*>`). Oku: `iconv -f UTF-16LE -t UTF-8 x.log` ya da PowerShell `Get-Content -Encoding Unicode`.
- mDeBERTa batch 64 max_len 160'ta VRAM'i doldurup throttle yaptı (adım süresi katlandı). Ağır modellerde batch'i küçült + gradient accumulation.

---

## 7. YENİ CLAUDE OTURUMU İÇİN PROMPT

```
Trendyol Datathon 2026 Kaggle (arama terimi–ürün alaka, binary, macro-F1).
Repo: trendyol-e-ticaret-yarismasi-2026-kaggle. Önce HANDOFF.md ve tarama.md oku.

Durum: GBDT+emb ve BERTurk cross-encoder eğittik. Holdout blend 0.906 AMA public
LB sadece 0.76 — çünkü eğitim negatiflerimiz (rastgele + TF-IDF) test reranking
setinin zor negatiflerine benzemiyor (HANDOFF §2). Asıl iş bunu düzeltmek.

Yapmanı istediğim (sırayla):
1. Tüm script'lerdeki DATA_DIR'ı bu makinedeki CSV klasörüne ayarla.
2. emb/ yoksa: python gen_embeddings.py (GPU, checkpoint'li).
3. HANDOFF §5.1'i uygula: embedding-ANN ile ZOR negatifler üret (item_emb/query_emb
   kullan), false-negative filtreli; yeni artifacts/train_pairs.parquet yaz.
   Sonra fast_submit.py (use-existing-pairs modu eklemen gerekebilir) + 
   train_cross_encoder.py + blend.py ile yeniden eğit/blend.
4. LB prevalans probe (all-ones) ile threshold'u kalibre et.
GPU var (RTX 5070 Ti). Her adımda önce kısa plan söyle, sonra uygula.
```
