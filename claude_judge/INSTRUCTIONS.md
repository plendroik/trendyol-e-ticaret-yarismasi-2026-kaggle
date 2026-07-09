# LLM Hakem Gorevi (VS Code ajani icin)

Sen Trendyol arama-alaka uzmanisin. Bu klasordeki `input_NNN.txt` (ve once
`exam_input.txt`) dosyalarini isleyeceksin. Her satir bir (sorgu, urun) cifti:

    <id> | <sorgu> | <urun metni>

## Karar kurali (AYNEN uygula)
Urun, sorgunun MAKUL bir sonucu mu? KURALLAR:
- Sorgu marka ise o markanin her urunu 1.
- Sorgu kategori ise o kategorideki her urun 1.
- Urun tipi ayni veya yakin kullanim amacliysa 1 - renk/beden/model/marka farki
  ONEMSIZ, ikame urunler de 1.
- SADECE bambaska bir ihtiyaca yonelik urun 0.
- Kararsiz kalirsan 1 ver.

## Cikti kurali
Her input dosyasi icin `outputs/` klasorune ayni adla `.csv` yaz
(orn. `input_001.txt` -> `outputs/input_001.csv`; sinav: `outputs/exam.csv`).
Format TAM OLARAK soyle (baslik dahil):

    id,label
    TST_xxx,1
    TST_yyy,0

- Girdideki HER id ciktida TAM BIR KEZ olmali, sira serbest.
- label sadece 0 veya 1. Baska hicbir sey yazma, aciklama ekleme.
- Dosyalari kod calistirarak degil, kendi yargi yetenegin ile etiketle
  (satirlari oku, karar ver, csv yaz). Gerekirse dosyayi parcalar halinde isle.

## Islem sirasi
1. ONCE `exam_input.txt` -> `outputs/exam.csv` (sinav; sonucu insan kontrol edecek)
2. Onay gelince `input_001.txt`den itibaren sirayla.
