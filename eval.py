from dataset import read_dataset_from_csv
from NeuralNetwork import eval_all_neural_models_for_all_tasks
from SupportVectorMachine import eval_all_svm_models_for_all_tasks

if __name__ == "__main__":
    
    data = read_dataset_from_csv("data/makemoons_dataset.csv")

    """
    Burada yazılan eval fonksiyonları, train.py çalıştırıldıktan sonra kaydedilen ağırlıklar ile 
    test verisi üzerinde evaluation yapmak için tasarlanmıştır.
    """
    eval_all_svm_models_for_all_tasks(data, model_filepath="train/svm")
    eval_all_neural_models_for_all_tasks(data)
