# 2450110_BLM5110_HW2


## Açıklama
Bu kod reposunda Makine Öğrenmesi (BLM5110) dersi kapsamında Python kullanılarak geliştirilmiş Yapay Nöron Ağı ve Destek Vektör Makinesi modeli bulunmaktadır. Bu modeller, Makine Öğrenmesi dersinin 2. ödevi için hazırlanmıştır.

___

## Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler ve versiyonları requirements.txt dosyasında bulunmaktadır. Bu kütüphaneler 

``` pip install -r requirements.txt ```

komutu ile yüklenebilir.
___

## Çalıştırma

Projenin eğitim kısmı train.py dosyasının içerisindeki fonksiyonlar ile yapılmaktadır. Bu dosya

``` python train.py ```

 komutu kullanılarak çalıştırılabilir. 
 Bu dosya çalıştırıldığında, modellerin ağırlıkları kaydedilecektir. 

Kaydedilen ağırlıklar ile değerlendirmeler test.py dosyasından yapılmaktadır. Bu dosya 

``` python test.py ```

 komutu kullanılarak çalıştırılabilir.
___

## Klasör Yapısı

    proje_klasoru/

        eval.py            # Model değerlendirme dosyası

        train.py            # Eğitimlerin yapıldığı dosya

        utils.py          # Yardımcı fonksiyonların bulunduğu dosya

        dataset.py          # Verilerle ilgili işlemlerin bulunduğu dosya

        requirements.txt   # Gerekli kütüphaneler

        data/               # Veri setini içerir

            makemoons_dataset.csv     # make_moons ile oluşturulan eğitim, doğrulama ve test verileri

        results/           # Eğitilen modellerin tarihçelerinin saklandığı klasör

        saved_models/       # Kaydedilen model ağırlıkları burada bulunur.

        train/              # Kodlar çalıştırılırsa eğitilen modellerin ağırlıkları buraya kaydedilecektir.

        graphs/          # Üretilen grafikler burada bulunur.

            confusion_matrices/         # Karmaşıklık matrisleri

            decision_boundaries/        # Çizdirilen karar sınırları

            loss_graphs/                # Epoch-Loss grafikleri

            neural_network_visualization/ # Yapay Nöron Ağı Görselleştirmeleri
___

### Öğrenci Bilgileri

24501100 - Aleyna Nil Uzunoğlu

YTÜ Bilgisayar Mühendisliği Tezli YL Öğrencisi
___
