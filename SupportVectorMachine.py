from dataset import generate_dataset, split_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import os

class SupportVectorMachine:

    def __init__(self, data, kernel='linear', verbose=0):
        
        """
        SVM sınıfının constructor fonksiyonu. Veriyi ve kernel türünü alır.

        Girdi:
          data (dict): Train, validation ve test verilerini barındıran sözlük.
          kernel (string): SVM kernel türü ('linear', 'poly', 'rbf' vb.).
          verbose (int): 0 ise çıktı verilmez, 1 ise işlem sırasında çıktılar gösterilir.

        Çıktı:
          Çıktısı yoktur. Değer atamalarını yapar.
        """
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

        self.kernel = kernel
        self.verbose = verbose
        self.model = None

    def tune_hyperparameters(self):
        """
        SVM için GridSearch ile en iyi hiperparametreleri bulur.
        Kernel türüne göre farklı hiperparametreler ayarlanır.

        Çıktı:
          Best model ve parametrelerini döndürür.
        """
        if self.kernel == 'linear':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif self.kernel == 'poly':
            param_grid = {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'coef0': [0, 1]}
        elif self.kernel == 'rbf':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.1, 0.01, 0.001]}
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

        grid_search = GridSearchCV(SVC(kernel=self.kernel), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        if self.verbose:
            print(f"En iyi parametreler: {grid_search.best_params_}")
            print(f"Doğrulama seti doğruluğu: {grid_search.best_score_:.4f}")

        return self.model, grid_search.best_params_

    def plot_decision_boundary(self, title="SVM Decision Boundary"):
        """
        Karar sınırını çizdiren fonksiyon.

        Girdi:
          title (string): Grafik başlığı.

        Çıktı:
          Çıktısı yoktur. Karar sınırını ve veriyi görselleştirir.
        """
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, edgecolor='k', cmap=plt.cm.coolwarm)
        plt.title(title)
        plt.show()

    def evaluate_model(self):
        """
        Eğitim, doğrulama ve test setlerinde modeli değerlendirir ve metrikleri gösterir.

        Çıktı:
          Classification report ve confusion matrix grafiği.
        """
        print("Eğitim Verisi:")
        y_train_pred = self.model.predict(self.X_train)
        print(classification_report(self.y_train, y_train_pred))

        print("\nDoğrulama Verisi:")
        y_val_pred = self.model.predict(self.X_val)
        print(classification_report(self.y_val, y_val_pred))

        print("\nTest Verisi:")
        y_test_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_test_pred))

        cm = confusion_matrix(self.y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.show()

def train_all_svm_models_for_all_tasks(data):
    
    """
    
    Bu fonksiyon, ödev için gereken tüm parametrelerle eğitimi gerçekleştirir.
    Train.py dosyasından çağırılıp kullanılmak üzere tasarlanmıştır.
    Bu fonksiyon çağırıldığında modeller eğitilir ve parametreleri dosyalara kaydedilir.
    
    Girdi: 
        Data: Train, test ve validation verilerini içerir.
        
    """
    
    kernels = ['linear', 'poly', 'rbf']
    
    for kernel in kernels:
        print(f"\nKernel: {kernel.upper()}")
        svm = SupportVectorMachine(data, kernel=kernel, verbose=1)
        svm.tune_hyperparameters()
        svm.plot_decision_boundary(title=f"SVM Decision Boundary ({kernel.upper()})")

       # Model kaydetme
        directory = "train/svm"
        if not os.path.exists(directory):
            os.makedirs(directory)  # Klasör yoksa oluştur
            print(f"[+] Klasör oluşturuldu: {directory}")
            
        # Model kaydetme
        filepath = f"train/svm/svm_{kernel}.joblib"
        joblib.dump(svm.model, filepath)
        print(f"[+] Model kaydedildi: {filepath}")
        
def eval_all_svm_models_for_all_tasks(data, model_filepaths):
    for kernel, filepath in model_filepaths.items():
        print(f"\nKernel: {kernel.upper()}")
        print(f"Model yükleniyor: {filepath}")
        model = joblib.load(filepath)
        
        # Test verileriyle değerlendirme
        y_test_pred = model.predict(data["X_test"])
        print(classification_report(data["y_test"], y_test_pred))
        
        cm = confusion_matrix(data["y_test"], y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix ({kernel.upper()})")
        plt.show()
        
if __name__ == "__main__":
    
    X,y = generate_dataset(noise=0.2, plot=False)
    print("[*] Data oluşturuldu. ")
    data = split_dataset(X,y)
    print("[*] Data train, test ve validation olarak ayrıldı. ")
    
    train_all_svm_models_for_all_tasks(data)
    
    model_filepaths = {
        "linear": "train/svm/svm_linear.joblib",
        "poly": "train/svm/svm_poly.joblib",
        "rbf": "train/svm/svm_rbf.joblib"
    }
    eval_all_svm_models_for_all_tasks(data, model_filepaths)    