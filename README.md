# Trendyol Search Relevance Datathon 2026 - ML Pipeline

Bu depo, Trendyol Search Relevance Datathon yarışması için geliştirilmiş, yüksek performanslı ve optimize edilmiş uçtan uca makine öğrenmesi boru hattını içermektedir. Çözüm, **0.87036 out-of-fold Macro-averaged F1** skoru elde etmekte olup, tamamen modüler ve laptop donanım sınırlarına (aşırı ısınma/kapanma) göre optimize edilmiştir.

---

## 🎯 Proje Amacı ve Hedef Metrik
Yarışmanın amacı, Türkçe kullanıcı aramaları (sorgular) ile ürün başlıkları arasındaki uygunluğu (relevance) sınıflandırmaktır. Model doğrudan **Macro-averaged F1-Score**'u maksimize edecek şekilde optimize edilmiştir:

$$\text{Score} = \frac{F1_{\text{relevant}} + F1_{\text{irrelevant}}}{2}$$

---

## 🏗️ Boru Hattı Mimarisi (Modular Architecture)
Proje, aşağıdaki modüler yapı üzerine kurulmuştur:
* `src/config.py`: Dizin yolları, tohumlama (seed), çapraz doğrulama parametreleri ve termal sınırlar.
* `src/text_normalization.py`: Türkçe karakterleri koruyan case folding, kısaltma genişletme ve marka isimlerini koruyan conservative kök bulucu.
* `src/data_processing.py`: Polars ile hızlı veri yükleme, TF-IDF ile benzerlik tespiti ve gelişmiş negatif örnekleme.
* `src/generate_embeddings.py`: SentenceTransformer (`multilingual-e5-base`) kullanarak ürün başlığı vektörlerinin üretilmesi.
* `src/feature_engineering.py`: Karakter/kelime benzerlikleri, marka/cinsiyet/yaş çelişki kontrolleri ve dense kosinüs benzerliği.
* `src/train.py`: Folds bazında GroupKFold çapraz doğrulama, LightGBM & CatBoost eğitimi ve dinamik eşik taraması.
* `src/predict.py`: K-Fold modelleri ile topluluk (ensemble) tahmini, en iyi eşik değerinin uygulanması ve uygunluk testleri.
* `main.py`: Tüm süreci uçtan uca koordine eden yönetici script.

---

## 💡 Neyi Neden Yaptık? (Tasarım ve Optimizasyon Kararları)

### 1. Negatif Örnekleme (Negative Sampling)
* **Ne Yaptık?**: Orijinal eğitim verisi (`training_pairs.csv`) sadece pozitif (1) etiketlerden oluşuyordu. Biz, pozitiflerin yanına 1:3 oranında (250k pozitif : 750k negatif) olmak üzere toplam 1.000.000 satırlık dengeli bir eğitim kümesi oluşturduk. Negatifleri 3 gruptan ürettik: Random (Coarse), Kategori Uyumlu (Medium) ve TF-IDF tabanlı Lexical Hard (Fine).
* **Neden Yaptık?**: Makine öğrenmesi modelinin "neyin alakasız olduğunu" öğrenebilmesi için karar sınırlarının net çizilmesi gerekir. Random negatifler modelin genel aramaları ayırt etmesini sağlarken; kategori uyumlu (örneğin aynı kök kategori ama farklı alt kategorideki ürünler) ve TF-IDF tabanlı (sözcük benzerliği yüksek ama eşleşmeyen ürünler) negatifler modelin ince detayları öğrenmesini sağlar.

### 2. Termal Koruma ve Checkpoint Sistemi (Laptop Isınma Çözümü)
* **Ne Yaptık?**: Ürün başlıklarının dense vektörlerini (embeddings) üretirken GPU batch size değerini `512`'den `64`'e düşürdük, batch aralarına `0.2` saniye soğuma uykusu (`time.sleep`) ekledik ve vektör üretimini her 100.000 üründe bir disk üzerine kaydettik (checkpointing).
* **Neden Yaptık?**: 962.873 ürünün tamamını kesintisiz olarak maksimum GPU yükünde işlemek dizüstü bilgisayarlarda aşırı sıcaklığa ve termal kapanmaya sebep oluyordu. Düşük batch boyutu ve soğuma aralıkları sayesinde RTX 4080 GPU sıcaklığı güvenli sınırlarda tutuldu. Checkpoint yapısı ise elektrik kesintisi veya kapanma durumunda kaldığı yerden devam edebilmesini sağladı.

### 3. Ön-Normalleştirme Önbelleği (Pre-normalization Caching)
* **Ne Yaptık?**: 50.153 sorguyu ve 962.873 ürün başlığını boru hattının en başında normalize edip temizleyerek parquet formatında diske kaydettik.
* **Neden Yaptık?**: Eğitim kümesindeki 1.000.000 çift ve test kümesindeki 3.359.679 çift üzerinde öznitelik çıkarırken her bir çift için tekrar tekrar metin normalleştirmek ve marka sözlüğü/Levenshtein kontrolü yapmak saatler sürüyordu. Normalleştirmeyi tekil veri boyutunda bir kez yapıp eşleştirdiğimizde, öznitelik çıkarma döngüsünün hızı saniyede 1.000'den **24.000 satıra çıktı** ve CPU ısınması sıfırlandı.

### 4. Bellek Yönetimi ve Batch Benzerlik Hesaplama (RAM Koruma)
* **Ne Yaptık?**: Test kümesindeki 3.3 milyon verinin dense kosinüs benzerliği hesaplanırken, tüm matrisi tek seferde float32 formatında belleğe almak yerine **200.000'lik paketler (batch)** halinde işledik.
* **Neden Yaptık?**: Tek seferde yapılan matris çarpımı bilgisayarda **36 GB RAM** ihtiyacı yaratıyor ve 32 GB fiziksel RAM olmasına rağmen sistemin swap yapmasına ve kilitlenmesine sebep oluyordu. Batch yapısıyla anlık RAM kullanımı **1 GB'ın altına** düşürüldü.

### 5. İşlem Parçalama ve OpenMP Çakışması Engelleme (LGBM & CatBoost Hızlandırma)
* **Ne Yaptık?**: LightGBM ve CatBoost modellerini eğitirken çekirdek kullanımını sınırladık (`MAX_THREADS = 1` / `n_jobs=1` / `thread_count=1`).
* **Neden Yaptık?**: NumPy, PyTorch, LightGBM ve CatBoost kütüphanelerinin çoklu iş parçacığı (multi-threading) OpenMP yönetimi Windows üzerinde birbiriyle çakışarak CPU'yu %100 yükte kilitliyor ve işlemi durma noktasına getiriyordu. Sınırlama sonrasında 100k verideki eğitim süresi **64 saniyeden 0.98 saniyeye (60 kat hızlanma)** düştü. Full veri eğitimi ise sadece saniyeler sürdü.

### 6. Sorgu Sızıntısını Önleyen CV Mimarisi (GroupKFold)
* **Ne Yaptık?**: Çapraz doğrulamada (cross-validation) folds ayrımını rastgele yapmak yerine `term_id` (sorgu bazlı) sütununa göre gruplayarak `GroupKFold` olarak uyguladık.
* **Neden Yaptık?**: Aynı arama sorgusunun hem eğitim hem de doğrulama foldlarında bulunması modelin skoru ezberlemesine (leakage) ve yerel doğrulama skorunun aldatıcı şekilde yüksek çıkmasına sebep olur. GroupKFold sayesinde model hiç görmediği sorgular üzerinden test edilerek Kaggle liderlik tablosu (leaderboard) ile mükemmel korelasyon sağlandı.

---

## 📈 Çapraz Doğrulama Sonuçları (Validation Results)

### Folds Detayları (Macro F1)
* **Fold 1 F1**: `0.85783`
* **Fold 2 F1**: `0.87303`
* **Fold 3 F1**: `0.87564`
* **Fold 4 F1**: `0.87571`
* **Fold 5 F1**: `0.86983`

### Out-of-Fold (OOF) Skorları
* **LightGBM OOF F1**: `0.87103` (Eşik: 0.45)
* **CatBoost OOF F1**: `0.86820` (Eşik: 0.46)
* **Ensemble (LGBM + CatBoost) OOF F1**: **`0.87036`** (Eşik: 0.46)

---

## 🚀 Sistemi Çalıştırma

### 1. Gereksinimler
Gerekli kütüphaneleri yüklemek için:
```bash
pip install polars scikit-learn lightgbm catboost sentence-transformers tqdm pandas numpy
```

### 2. Boru Hattını Uçtan Uca Koordine Etmek
Tüm süreci (veri işleme, vektör üretimi, öznitelik mühendisliği, eğitim ve tahmin) tek seferde güvenli ve optimize şekilde çalıştırmak için ana dizindeki `main.py` dosyasını çalıştırmanız yeterlidir:
```bash
python main.py
```

İşlem tamamlandığında proje ana dizininde Kaggle'a doğrudan yüklenebilecek en iyi şekilde kalibre edilmiş **`submission.csv`** dosyası oluşturulacaktır.
