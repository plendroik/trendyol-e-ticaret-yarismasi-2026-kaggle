# DEVİR NOTU — Trendyol Datathon 2026 (Kaggle aşaması)

> Bu dosya, çalışmayı yarın **şarjla** devam ettirecek kişi içindir. Önce "Hızlı devam"ı uygula, sonra "Yol haritası"na geç. En altta, yeni bir Claude Code oturumuna yapıştırılabilecek hazır bir prompt var.

---

## 1. ŞU AN NEREDEYİZ (özet)

- **Görev:** (arama terimi, ürün) çifti için **binary alaka** tahmini (1=alakalı, 0=alakasız). Metrik: **macro-F1**. Submission: `id,prediction` (0/1).
- **Elimizdeki submission:** `<DATA_DIR>/submission.csv` — sadece leksikal/yapısal feature'lı GBDT (LightGBM+CatBoost).
  - **OOF macro-F1 = 0.831** (threshold 0.42). Sınıf-1 (alakalı) recall=0.70 → **iyileştirilecek asıl yer burası.**
  - ⚠️ Bu skor *bizim ürettiğimiz negatif dağılımı* üzerinde. Gerçek LB farklı çıkar (bkz. Yol Haritası #1 — prevalans probe).
- **Yarım kalan iş:** Trendyol embedding modeliyle vektörler üretiliyordu (GPU). Batarya bitti, **5 chunk'tan 3'ü `emb/` içinde cache'li** (checkpoint'li, baştan başlamaz).

## 2. KRİTİK VERİ GERÇEKLERİ (stratejiyi bunlar belirliyor)

- **TAM COLD-START:** Eğitim terimleri (17.968) ile test terimleri (32.185) **HİÇ örtüşmüyor** (0 ortak). Model görülmemiş sorgulara genelleme yapmak zorunda → **semantik (embedding/cross-encoder) ezberden çok daha önemli.** Validation'da `GroupKFold(term_id)` şart (zaten kullanılıyor).
- **TEST NEGATİFLERİ ZOR:** Her test sorgusunda ~104 aday ürün var; bunlar Trendyol'un arama motorunun getirdiği, sorguyla zaten ilişkili adaylar. Yani testteki "alakasız"lar **rastgele değil, zor negatif.** Eğitim negatifleri de ağırlıklı **hard** olmalı (rastgele değil) → yoksa threshold yanlış kalibre olur. (Eski çözümün %33 rastgele negatifi bu yüzden hatalıydı.)
- **Bütçe:** günde **5 submission**, finalde **2 seçim**. LB'de hill-climbing YAPMA; lokal CV'ye güven, LB'yi sadece prevalans doğrulamak için kullan.
- **Veri kolonları** (doğrulandı): items=`item_id,title,category,brand,gender,age_group,attributes` · terms=`term_id,query` · training_pairs=`id,term_id,item_id,label`(hepsi 1) · submission_pairs=`id,term_id,item_id`.

## 3. DOSYALAR

| Dosya | Ne yapar |
|------|----------|
| `fast_submit.py` | Ana pipeline: Türkçe normalize → TF-IDF hard negatif örnekleme → leksikal/yapısal feature → (varsa) **emb_cos** feature → 5-fold LGBM+CatBoost → threshold sweep → `submission.csv`. **~5 dk (embeddingsiz).** |
| `gen_embeddings.py` | Trendyol embedding modeliyle sorgu+ürün vektörleri üretir. **Checkpoint'li** (kaldığı chunk'tan devam). GPU'da ~25 dk. |
| `emb/` | Vektör cache: `query_emb.npy`, `item_emb.npy`, `_item_chunk_*.npy` (yarım kalan chunk'lar). |
| `tarama.md` | Literatür/strateji playbook'u (PU learning, negatif örnekleme, macro-F1, Türkçe NLP). Oku. |
| `src/` | Eski (yiğit) pipeline — polars + cross-encoder'sız GBDT. Referans, ama `fast_submit.py` daha hızlı/temiz; bunu ileri taşı. |

## 4. HIZLI DEVAM (şarj gelince, ~15 dk)

> Not: `fast_submit.py` ve `gen_embeddings.py` içindeki `DATA_DIR` değişkeni CSV'lerin bulunduğu klasöre işaret etmeli. Şu an: `C:\Users\ASUS\Desktop\trendyol`. **Başka makinedeysen kendi yoluna güncelle.**

```bash
# 1) Embeddings'i tamamla (aynı makinede cache'den devam eder; başka makinede sıfırdan ~25 dk)
python gen_embeddings.py

# 2) Pipeline'ı embedding'li çalıştır -> emb_cos feature otomatik eklenir
python fast_submit.py
```
Çıktıdaki yeni **OOF macro-F1**'i 0.831 ile kıyasla. `emb_cos`'un feature importance'ına ve sınıf-1 recall'unun artıp artmadığına bak. Artıyorsa yeni `submission.csv`'yi yükle.

## 5. YOL HARİTASI (ROI sırasına göre — asıl değer burada)

1. **🔴 Prevalans probe + threshold kalibrasyonu (GPU'suz, en yüksek ROI).**
   1 submission `all-ones`, 1 submission `all-zeros` yükle. macro-F1 formülünden test pozitif oranını çöz; threshold'u o orana kilitle. Şu anki threshold (0.42) bizim kurgu dağılımımıza göre — gerçek prevalansa göre düzeltilince LB ciddi oynar. (Detay: `tarama.md` §5.)
2. **🔴 `emb_cos` feature (yukarıdaki Hızlı Devam).** Semantik sinyal, cold-start'ta sınıf-1 recall'una doğrudan vurur.
3. **🟠 Embedding tabanlı hard negatif + false-negative filtresi.** TF-IDF yerine/yanında Trendyol embedding ANN ile rank 101–500 negatif çek; pozitif skorun >%95'i olan adayları ele (NV-Retriever). Negatif kalitesi = kalibrasyon kalitesi.
4. **🟠 Cross-encoder reranker (en büyük accuracy kaldıracı).** BERTurk (`dbmdz/bert-base-turkish-cased`) veya XLM-R'ı pozitif+mined negatif üzerinde eğit; girdi olarak yapısal alanları serialize et: `query [SEP] brand:… category:… renk:… materyal:… title`. Skorunu GBDT'ye **feature olarak** ver (hem skor toplar hem açıklanabilirlik korunur — stage-2'de %10 puan).
5. **🟢 Ensemble + 2 final submission.** Biri robust (tek sağlam model), biri max-CV ensemble. Seed/backbone çeşitlendir → shake-up riskini düşür.

## 6. ORTAM NOTLARI

- Python 3.11, GPU: RTX 5070 Ti Laptop (CUDA OK). Kurulu: numpy, pandas, sklearn, scipy, lightgbm, catboost, torch(cu128), sentence-transformers, tqdm. **`polars` KURULU DEĞİL** (bu yüzden `fast_submit.py` pandas tabanlı; eski `src/` polars istiyor).
- Embedding modeli `trust_remote_code=True` ister (gte-multilingual tabanlı). İlk indirme ~12 dk sürmüştü.
- Loglar UTF-16 (`*>` yönlendirmesi). Okumak için: `iconv -f UTF-16LE -t UTF-8 run.log` veya PowerShell `Get-Content -Encoding Unicode`.

---

## 7. YENİ CLAUDE OTURUMU İÇİN YAPIŞTIRILABİLİR PROMPT

```
Trendyol Datathon 2026 Kaggle yarışmasında çalışıyorum (arama terimi–ürün alaka,
binary, metrik macro-F1). Repo: trendyol-e-ticaret-yarismasi-2026-kaggle.
Önce HANDOFF.md ve tarama.md dosyalarını oku — mevcut durum, veri gerçekleri
(tam cold-start, zor test negatifleri) ve yol haritası orada.

Durum: fast_submit.py ile leksikal GBDT baseline'ımız var (OOF macro-F1 0.831,
submission.csv hazır). Trendyol embedding'leri yarım kalmıştı; emb/ içinde 3/5
chunk cache'li.

Yapmanı istediğim (sırayla):
1. fast_submit.py ve gen_embeddings.py içindeki DATA_DIR'ı bu makinedeki CSV
   klasörüne göre ayarla (CSV'ler: items.csv, terms.csv, training_pairs.csv,
   submission_pairs.csv, sample_submission.csv).
2. `python gen_embeddings.py` çalıştır (cache'den devam eder), bitince
   `python fast_submit.py` çalıştır. Yeni OOF macro-F1'i 0.831 ile kıyasla,
   emb_cos feature importance'ına ve sınıf-1 recall'una bak.
3. Sonra yol haritası #1'i (LB prevalans probe + threshold) ve #4'ü (BERTurk/XLM-R
   cross-encoder reranker, skoru GBDT'ye feature olarak) uygula.
GPU var (RTX 5070 Ti). Her adımda önce planını söyle, sonra uygula.
```
