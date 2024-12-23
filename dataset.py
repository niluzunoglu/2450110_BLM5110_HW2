from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import numpy as np

def generate_dataset(noise=None, plot=True):
    
    """
    Veri seti oluşturan fonksiyon.

    Girdi: 
      noise (float): Veride ne kadar gürültü olacağını belirleyen noise parametresi. Default None alır.
            (Ödevde default değer olan None ile çalışılmıştır.)
      plot (boolean): Veri setini görselleştirmeyi belirten için plot parametresi. Default True alır.

    Çıktı: Veri seti ve etiketleri. (X, y)

    """

    X, y = make_moons(n_samples=400, noise=noise, random_state=42)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    if plot:
      plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
      plt.xlabel('Özellik 1')
      plt.ylabel('Özellik 2')
      plt.title('make_moons dataset')
      plt.show()

    return X,y

def split_dataset(X,y):

    """
      Aldığı veri seti(X) ve etiketleri(y) alıp, train, validation ve test olarak ayıran fonksiyon.

      Girdi: Veriler (X) ve etiketleri (y)
      Çıktı: X_train, X_val, X_test, y_train, y_val, y_test verilerini içeren sözlük.

    """

    # Train - Test - Validation için 2 kez train_test_split metodu çağırılır.
    # İlk split metodu Train - Test olarak ayırma yapar. (%80 Train - %20 Test)
    # İkinci split metodu Train verisini Train - Validation olarak ayırı. 
    # (İlk veriye göre %60 Train - %20 Validation olarak)
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return {"X_train":X_train, "X_val": X_val, "X_test": X_test,"y_train": y_train, "y_val":y_val, "y_test": y_test}
  

def save_dataset_to_csv(data, filename):
  
    """
    Verilen veri setini belirtilen bir CSV dosyasına kaydeden fonksiyon.

    Girdi:
      data (dict): X_train, X_val, X_test, y_train, y_val, y_test verilerini içeren sözlük.
      filename (str): Verilerin kaydedileceği dosya ismi. (Örnek: 'dataset.csv')

    Çıktı: 
      Çıktısı yoktur. Verileri belirtilen dosyaya yazar.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Başlık satırı yazılır
        writer.writerow(['Set', 'Feature 1', 'Feature 2', 'Label'])
        
        # Eğitim verileri
        for i in range(len(data['X_train'])):
            writer.writerow(['Train', data['X_train'][i][0], data['X_train'][i][1], data['y_train'][i]])
        
        # Doğrulama verileri
        for i in range(len(data['X_val'])):
            writer.writerow(['Validation', data['X_val'][i][0], data['X_val'][i][1], data['y_val'][i]])
        
        # Test verileri
        for i in range(len(data['X_test'])):
            writer.writerow(['Test', data['X_test'][i][0], data['X_test'][i][1], data['y_test'][i]])
    
    print(f"Veri seti '{filename}' dosyasına başarıyla kaydedildi.")

def read_dataset_from_csv(filename):
    """
    Belirtilen bir CSV dosyasından veri setini okuyan fonksiyon.

    Girdi:
      filename (str): Verilerin okunacağı dosya ismi. ('dataset.csv' gibi)

    Çıktı:
      X_train, X_val, X_test, y_train, y_val, y_test verilerini içeren sözlük.
    """
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Başlık satırını atla

        for row in reader:
            set_type, feature1, feature2, label = row
            if set_type == 'Train':
                X_train.append([float(feature1), float(feature2)])
                y_train.append(int(label))
            elif set_type == 'Validation':
                X_val.append([float(feature1), float(feature2)])
                y_val.append(int(label))
            elif set_type == 'Test':
                X_test.append([float(feature1), float(feature2)])
                y_test.append(int(label))

    return {
        "X_train": np.array(X_train),
        "X_val": np.array(X_val),
        "X_test": np.array(X_test),
        "y_train": np.array(y_train),
        "y_val": np.array(y_val),
        "y_test": np.array(y_test)
    }