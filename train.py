from NeuralNetwork import train_all_neural_models_for_all_tasks
from SupportVectorMachine import train_all_svm_models_for_all_tasks
from dataset import generate_dataset, split_dataset, save_dataset_to_csv

if __name__ == "__main__":
    
    print("[*] Train.py ")
    X,y = generate_dataset(noise=0.2, plot=True)
    print("[*] Data oluşturuldu. ")
    data = split_dataset(X,y)
    print("[*] Data train, test ve validation olarak ayrıldı.")
    save_dataset_to_csv(data=data, filename="data/makemoons_dataset.csv")
    print("[*] Veriler data/makemoons_dataset.csv pathine kaydedildi.")
    
    """
    Burada yazılan train fonksiyonları, yapay nöron ağları ve svm için tüm taskları gerçekleştirir.
    Eğitim yapıp parametrelerini train/ dizini altına atar.
    Eğitim yapmadan parametrelere erişmek isterseniz saved_models dizinini kullanabilirsiniz.
    """
    train_all_neural_models_for_all_tasks(data=data)
    train_all_svm_models_for_all_tasks(data=data)

    
    