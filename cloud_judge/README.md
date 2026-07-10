# Bulut Hakem (Kaggle 2xT4, kurallara uygun acik-kaynak etiketleme)

## Kurulum (5 dk)
1. kaggle.com -> Datasets -> New Dataset (PRIVATE!) -> bu klasordeki 4 adet
   band_ids_partN.csv.gz dosyasini yukle, adi: cloud-judge-inputs
2. Yeni Notebook ac: Add Data -> yarisma verisi + cloud-judge-inputs
   Settings: Accelerator=GPU T4 x2, Internet=ON, Persistence=Files
3. judge_kaggle.py icerigini tek hucreye yapistir; PART=1 (senin hesap),
   arkadas hesabi PART=2 ... (4 parca, 4 oturum)
4. Run. Once SINAV kosar (RECALL>=0.95 sart), sonra etiketler.
   Cikti: /kaggle/working/labels_partN.csv -> indir, repoya koy.

## Notlar
- Oturum ~12 saatte kapanir; script RESUME-SAFE degil Kaggle'da dosya kalicivligi
  icin Persistence=Files ac ve ayni notebook'u yeniden Run et (done-set okur).
- Hiz: ~15-40 cift/sn beklenir -> part basina ~1.5-3 saat.
- Model: Qwen3-14B-AWQ (acik kaynak, self-host = kurallara uygun).
- Yarisma verisi Kaggle disina CIKMAZ (private dataset + Kaggle compute).
