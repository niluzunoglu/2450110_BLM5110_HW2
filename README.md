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

        results/           # Sonuçların saklandığı klasör

            costs-per-epoch/ # Ayrıntılı logların (her bir epoch için kayıp miktarı) saklandığı dizin

            TEST_metrics.txt    # Test metriklerini barındıran dosya.

            TRAIN_metrics.txt   # Eğitim metriklerini barındıran dosya.

            VALIDATION_metrics.txt  # Validasyon metriklerini barındıran dosya

        saved_models/       # Kaydedilen model ağırlıkları burada bulunur.

        graphs/          # Üretilen grafikler burada bulunur.
___

### Öğrenci Bilgileri

24501100 - Aleyna Nil Uzunoğlu

YTÜ Bilgisayar Mühendisliği Tezli YL Öğrencisi
___
