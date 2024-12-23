from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt
from dataset import generate_dataset, split_dataset

class NeuralNetwork:

  def __init__(self,data, gradient_descent_method,verbose, learning_rate=0.001, num_epochs=100, num_hidden_layer=1, batch_size_for_mbgd=32):

    """
    Neural network classının constructor fonksiyonu. Veriyi, gradient descent metodunu, epoch sayısını, gizli katman sayısını, batch size'ı alır.

    Girdi:
      data (dict): Train, valdiation ve test verilerini barındıran sözlük.
      gradient_descent_method (string): "sgd", "mini-batch" ve "batch" değerlerini alabilir. Gradient descent modunu belirtir.
      num_epochs(int): Epoch sayısını belirtir.
      num_hidden_layer(int): Gizli katman sayısını belirtir.
      batch_size_for_mbgd(int): Mini-batch için batch boyutunu belirtir.
      verbose: 0 ise run edilirken çıktı verilmez. 1 ise verilir.
      learning_rate: Öğrenme oranı. Default olarak 0.001 olur.

    Çıktı:
      Çıktısı yoktur. Değer atamalarını yapar.

    """

    self.X_train = data["X_train"]
    self.y_train = data["y_train"]
    self.X_val = data["X_val"]
    self.y_val = data["y_val"]
    self.X_test = data["X_test"]
    self.y_test = data["y_test"]

    self.model = None
    self.history = None
    self.gradient_descent_method = gradient_descent_method
    self.num_hidden_layer = num_hidden_layer
    self.num_epochs = num_epochs
    self.batch_size_for_mbgd = batch_size_for_mbgd
    self.verbose = verbose
    self.learning_rate = learning_rate

    # Modelin kaydedileceği dosyanın adını belirten değişken.
    self.identifier = f"Neural_{self.num_hidden_layer}layered_{self.gradient_descent_method}.weights.h5"

  def do_operations(self, save=False):
    """
      Bütün operasyonları (create_model, train_model, plot_graphs, plot_confusion matrix.. ) her bir
      işlem birimi için (1Layer SGD, 1Layer MBGD, ... , ... ) main metodunda ayrı ayrı çağırmak şık olmadığı için
      bu fonksiyon yazılmıştır.

      Girdi:
        Save (boolean): Eğer doğru ise, ağın parametreleri kaydedilir.

      Çıktı:
        Döndürdüğü bir değer yoktur. Modeli eğitir, epoch/loss grafiklerini çizdirir, karar sınırını çizdirir, karmaşıklık
        matrislerini yazdırır ve modeli değerlendirir. (Eval metodu ile.) Eğer save parametresi true ise kaydeder.

    """
    self.train_model(visualize=True)
    self.plot_epoch_loss_graph()
    self.plot_decision_boundary()
    self.plot_confusion_matrix()
    self.eval()

    if save == True:
      self.model.save_weights(self.identifier)

  def create_model(self):

    """
    Modeli oluşturan fonksiyon. 
    Sınıf değişkeni olarak alınan num_hidden_layer parametresi ile kaç katman oluşturulacağını kontrol eder.
    Bu parametreye göre gizli katman oluşturur ve modele ekler.

    Ara katmanlarda activation olarak "ReLU" fonksiyonu, çıktı katmanında aktivasyon fonksiyonu olarak "Sigmoid" kullanır.

    Tek katmanlı ağ için DSB nöronlu bir yapı,
    2 katmanlı bir ağ için 64-32 nöronlu bir yapı,
    3 katmanlı bir ağ için 32-64-32 nöronlu bir yapı kurulmuştur. 

    """

    if self.num_hidden_layer == 1:
      self.model = tf.keras.Sequential([
          tf.keras.layers.Dense(32, activation='relu', input_shape=(self.X_train.shape[1],)),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])


    elif self.num_hidden_layer == 2:
      self.model = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])

    elif self.num_hidden_layer == 3:
      self.model = tf.keras.Sequential([
          tf.keras.layers.Dense(32, activation='relu', input_shape=(self.X_train.shape[1],)),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])

  def train_model(self, visualize):

    """
    Modelin eğitimini gerçekleştiren fonksiyon.
    
    Girdi:
      Visualize(boolean): Eğer True ise, nöral ağı görselleştirir. 

    Çıktı:
      Çıktısı yoktur. Modeli oluşturur, compile eder, fit eder.
    """

    self.create_model()

    # Model derleme
    self.model.compile(optimizer= SGD(learning_rate=self.learning_rate),
                       loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall'])

    if self.gradient_descent_method == 'sgd':
      batch_size = 1
    elif self.gradient_descent_method == 'mini-batch':
      batch_size = self.batch_size_for_mbgd
    elif self.gradient_descent_method == 'batch':
      batch_size = len(self.X_train)

    # Model eğitimi
    self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=self.num_epochs, batch_size=batch_size, verbose=self.verbose)

    if visualize:
      plot_model(self.model, to_file=f"neural_network_{self.num_hidden_layer}_layered_{self.gradient_descent_method}.png", show_shapes=True, show_layer_names=True)

  def plot_epoch_loss_graph(self):

    """
    Epoch/loss grafiğini çizen fonksiyon. 
    Girdisi yoktur, çıktısı epoch/loss grafiğidir. 
    """

    history_detailed = self.history.history
    plt.figure(figsize=(8, 6))
    plt.plot(history_detailed['loss'], label='Train Loss')
    if 'val_loss' in history_detailed:
        plt.plot(history_detailed['val_loss'], label='Validation Loss')
    plt.title("Epoch vs. Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

  def plot_confusion_matrix(self):
      """
      Hem eğitim, hem doğrulama, hem de test setleri için confusion matrix grafikleri çizen fonksiyon.
      Grafikleri alt alta olacak şekilde düzenler.

      Girdisi yoktur, class değişkenlerini kullanır. Çıktısı, eğitim doğrulama ve test verileri için
      karmaşıklık matrislerinin görselleridir.

      """

      # Eğitim 
      y_train_pred = (self.model.predict(self.X_train, batch_size=40) >= 0.5).astype(int)
      cm_train = confusion_matrix(self.y_train, y_train_pred)

      # Doğrulama 
      y_val_pred = (self.model.predict(self.X_val, batch_size=40) >= 0.5).astype(int)
      cm_val = confusion_matrix(self.y_val, y_val_pred)

      # Test
      y_test_pred = (self.model.predict(self.X_test, batch_size=40) >= 0.5).astype(int)
      cm_test = confusion_matrix(self.y_test, y_test_pred)

      # Confusion matrix'leri alt alta çizdirme
      fig, axes = plt.subplots(3, 1, figsize=(6, 18))  

      # Eğitim seti
      axes[0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
      axes[0].set_title("Confusion Matrix (Train)")
      axes[0].set_xticks(np.arange(2))
      axes[0].set_yticks(np.arange(2))
      axes[0].set_xticklabels(['Class 0', 'Class 1'])
      axes[0].set_yticklabels(['Class 0', 'Class 1'])
      for i, j in np.ndindex(cm_train.shape):
          axes[0].text(j, i, format(cm_train[i, j], 'd'),
                      horizontalalignment="center",
                      color="white" if cm_train[i, j] > cm_train.max() / 2 else "black")

      # Doğrulama seti
      axes[1].imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
      axes[1].set_title("Confusion Matrix (Validation)")
      axes[1].set_xticks(np.arange(2))
      axes[1].set_yticks(np.arange(2))
      axes[1].set_xticklabels(['Class 0', 'Class 1'])
      axes[1].set_yticklabels(['Class 0', 'Class 1'])
      for i, j in np.ndindex(cm_val.shape):
          axes[1].text(j, i, format(cm_val[i, j], 'd'),
                      horizontalalignment="center",
                      color="white" if cm_val[i, j] > cm_val.max() / 2 else "black")

      # Test seti
      axes[2].imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
      axes[2].set_title("Confusion Matrix (Test)")
      axes[2].set_xticks(np.arange(2))
      axes[2].set_yticks(np.arange(2))
      axes[2].set_xticklabels(['Class 0', 'Class 1'])
      axes[2].set_yticklabels(['Class 0', 'Class 1'])
      for i, j in np.ndindex(cm_test.shape):
          axes[2].text(j, i, format(cm_test[i, j], 'd'),
                      horizontalalignment="center",
                      color="white" if cm_test[i, j] > cm_test.max() / 2 else "black")

      # Genel ayarlar
      for ax in axes:
          ax.set_ylabel('True Label')
          ax.set_xlabel('Predicted Label')
      plt.tight_layout()
      plt.show()

  def evaluate_on_test_data(self):

    """
      Girdisi yoktur. 
      Modeli test datası üzerinde evaluate eder.
      Sonuçları yazdırır.
    """
    print("Evaluate on test data")
    results = self.model.evaluate(self.X_test, self.y_test)
    print("test loss, test acc, precision, recall:", results)

  def plot_decision_boundary(self):
      """
      Karar sınırlarını çizmek için kullanılan fonksiyon.
      Öğrenme bittiktan sonra kullanılacaktır.

      """

      x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
      y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1

      xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                          np.arange(y_min, y_max, 0.01))

      grid_points = np.c_[xx.ravel(), yy.ravel()]
      Z = (self.model.predict(grid_points, batch_size=40) >= 0.5).astype(int)
      Z = Z.reshape(xx.shape)

      plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

      plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, edgecolor='k', cmap=plt.cm.coolwarm, label='Train Data')
      plt.scatter(self.X_val[:, 0], self.X_val[:, 1], c=self.y_val, edgecolor='k', marker='x', cmap=plt.cm.coolwarm, label='Validation Data')

      plt.title(f"Decision Boundary ({self.gradient_descent_method.capitalize()} Gradient Descent)")
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.legend()
      plt.show()


def train_all_neural_models_for_all_tasks(data):
  
    # veri : 240 eğitim, 80 validasyon, 80 test
    print("[1 Katman] Çalışmalar...")
    neural_network_one_layered_sgd = NeuralNetwork(data,gradient_descent_method="sgd",learning_rate=0.01, num_epochs=30,num_hidden_layer=1,verbose=0)
    neural_network_one_layered_sgd.do_operations(save=True)
    print("[*] 1 katmanlı SGD yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_one_layered_mbgd = NeuralNetwork(data,gradient_descent_method="mini-batch", batch_size_for_mbgd=24, learning_rate=0.001, num_epochs=1000,num_hidden_layer=1,verbose=0)
    neural_network_one_layered_mbgd.do_operations(save=True)
    print("[*] 1 katmanlı Mini batch (64 batch size ile) gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_one_layered_bgd = NeuralNetwork(data,gradient_descent_method="batch",learning_rate=0.01, num_epochs=2000,num_hidden_layer=1,verbose=0)
    neural_network_one_layered_bgd.do_operations(save=True)
    print("[*] 1 katmanlı batch gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    print("[2 Katman] Çalışmalar...")
    neural_network_two_layered_sgd = NeuralNetwork(data,gradient_descent_method="sgd",num_epochs=50,num_hidden_layer=2,verbose=0)
    neural_network_two_layered_sgd.do_operations(save=True)
    print("[*] 2 katmanlı SGD yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_two_layered_mbgd = NeuralNetwork(data,gradient_descent_method="mini-batch", batch_size_for_mbgd=64,num_epochs=300,num_hidden_layer=2, verbose=0)
    neural_network_two_layered_mbgd.do_operations(save=True)
    print("[*] 2 katmanlı Mini batch (30 batch size ile) gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_two_layered_bgd = NeuralNetwork(data,gradient_descent_method="batch",num_epochs=3000,num_hidden_layer=2, verbose=0)
    neural_network_two_layered_bgd.do_operations(save=True)
    print("[*] 2 katmanlı batch gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    print("[3 Katman] Çalışmalar...")
    neural_network_three_layered_sgd = NeuralNetwork(data,gradient_descent_method="sgd",num_epochs=1000,num_hidden_layer=3,verbose=0)
    neural_network_three_layered_sgd.do_operations(save=True)
    print("[*] 3 katmanlı SGD yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_three_layered_mbgd = NeuralNetwork(data,gradient_descent_method="mini-batch", batch_size_for_mbgd=16,num_epochs=3000,num_hidden_layer=3, verbose=0)
    neural_network_three_layered_mbgd.do_operations(save=True)
    print("[*] 3 katmanlı Mini batch (30 batch size ile) gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")

    neural_network_three_layered_mbgd = NeuralNetwork(data,gradient_descent_method="batch",num_epochs=6000,num_hidden_layer=3, verbose=0)
    neural_network_three_layered_mbgd.do_operations(save=True)
    print("[*] 3 katmanlı batch gd yöntemiyle öğrenen ağ eğitimi tamamlandı. ")
  
if __name__ == "__main__":

    print("[*] Now Training --- Neural Network ")

    X,y = generate_dataset(noise=0.2, plot=False)
    print("[*] Data oluşturuldu. ")
    data = split_dataset(X,y)
    print("[*] Data train, test ve validation olarak ayrıldı. ")

    # veri : 240 eğitim, 80 validasyon, 80 test
    train_all_models_for_task(data)