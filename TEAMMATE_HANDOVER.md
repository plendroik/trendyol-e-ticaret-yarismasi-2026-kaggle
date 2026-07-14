# 🚀 TRENDYOL 2026 - MODEL EĞİTİMİ DEVİR REHBERİ (HANDOVER)

Bu rehber, takım arkadaşınızın localde (12 GB VRAM GPU) veya Kaggle üzerinde eğitimleri kaldığı yerden devralıp tamamlaması için gereken tüm bilgileri içerir.

---

## 1. Depo Senkronizasyonu (Git Pull)
Local bilgisayarınızda terminali açıp repoyu en son commits ile güncelleyin:
```bash
git pull
```
*Tüm eğitim betikleri, kaggle scriptleri ve gerekli veri etiketleri (`etiketliveri/clean_labels_72b.csv`) repoda güncel olarak yer almaktadır.*

---

## 2. Local Model Eğitimi (12 GB VRAM RTX GPU)
Kullanıcının bilgisayarında **12 GB VRAM** bulunuyor. Bu kart, Türkçe alanında en başarılı olan **Trendyol Domain Model** (`TY-ecomm-embed-multilingual-base-v1.2.0`) eğitimi için fazlasıyla yeterlidir.

Eğitim sırasında internet kopma/bağlantı sorunları yaşamamak için `--local_files_only` (internetsiz/offline mod) parametresi eklenmiştir. Model ağırlıkları önbellekten çekilir.

### Seed 100 Domain Model Eğitimi:
```bash
python train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix 72b_domain_s100 --pairs_file train_pairs_72b.parquet --seed 100 --local_files_only
```

### Seed 2026 Domain Model Eğitimi:
```bash
python train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix 72b_domain_s2026 --pairs_file train_pairs_72b.parquet --seed 2026 --local_files_only
```

⚠️ **Bellek (OOM) Hatası Alınırsa:**
12 GB VRAM için varsayılan batch size (64) bazen fazla gelebilir. Eğer Out-of-Memory hatası alırsanız, batch size değerini düşürerek çalıştırabilirsiniz:
```bash
python train_cross_encoder.py --model "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0" --suffix 72b_domain_s100 --pairs_file train_pairs_72b.parquet --seed 100 --local_files_only --train_batch 32
```

---

## 3. Kaggle Üzerinde Eğitim (Ücretsiz & Bulutta)
`xlm-roberta-large` (2.2 GB) modeli çok büyük olduğundan Hugging Face indirme sınırları nedeniyle Kaggle'da direkt indirme sırasında kilitlenmeler yaşanmaktadır. Bunu aşmak için modeli Kaggle Dataset olarak ekleyip **internetsiz yükleme** yapısı kurulmuştur.

### A. XLM-Roberta-Large Eğitimi (`cloud_judge/train_kaggle_large.py`)
1. Kaggle'da yeni bir Notebook açın.
2. Sağ menüden **Accelerator:** **GPU T4 x2** (veya tek T4) seçin.
3. **+ Add Input** butonuna tıklayın:
   * Arama kısmına `"LM-Roberta-Large Pytorch Pytorch TPU"` veya `"xlm-roberta-large"` yazıp çıkan hazır model veri setini notebook'a bağlayın.
   * Kendi yarışma veri setinizi (`submission_pairs.csv`, `train_text.parquet`, `test_text_part*.parquet`) notebook'a bağlayın.
4. Repodaki [cloud_judge/train_kaggle_large.py](file:///c:/Users/ASUS/Desktop/trendyol/trendyol-e-ticaret-yarismasi-2026-kaggle/cloud_judge/train_kaggle_large.py) dosyasının kodunu kopyalayıp Kaggle hücresine yapıştırın.
5. **Save Version** (Save & Run All) diyerek arka planda eğitimi başlatın.

### B. Domain Model Alternatif Eğitimi (`cloud_judge/train_kaggle_domain.py`)
Eğer takım arkadaşınız kendi bilgisayarını yormak istemiyorsa, yukarıdaki domain model eğitimini de Kaggle'da çalıştırabilir:
1. Kaggle'da yeni bir Notebook açıp **GPU T4 x1** seçin.
2. Yarışma verilerini bağlayın.
3. Repodaki [cloud_judge/train_kaggle_domain.py](file:///c:/Users/ASUS/Desktop/trendyol/trendyol-e-ticaret-yarismasi-2026-kaggle/cloud_judge/train_kaggle_domain.py) kodunu yapıştırıp çalıştırın.

---

## 4. Çıktıların Birleştirilmesi ve Stacker Çalıştırılması
Tüm eğitimler bittiğinde oluşan `.npy` dosyalarını yerel bilgisayardaki `artifacts/` klasörüne yerleştirin:
* `ce_holdout_72b_xlmrL2.npy` ve `ce_test_72b_xlmrL2.npy` (XLM-R Large çıktıları)
* `ce_holdout_72b_domain_s100.npy` ve `ce_test_72b_domain_s100.npy` (Seed 100 çıktıları)
* `ce_holdout_72b_domain_s2026.npy` ve `ce_test_72b_domain_s2026.npy` (Seed 2026 çıktıları)

Son adımda, open-source stacker'ı çalıştırarak yeni tahminleri ensemble edin:
```bash
python build_stacker_pure_os.py
```
Bu işlem sonunda oluşacak yeni submission dosyası skorunuzu daha da yukarı taşıyacaktır!
