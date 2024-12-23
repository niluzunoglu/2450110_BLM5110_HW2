from dataset import read_dataset_from_csv
from NeuralNetwork import eval_all_neural_models_for_all_tasks
from SupportVectorMachine import eval_all_svm_models_for_all_tasks

data = read_dataset_from_csv("data/makemoons_dataset.csv")

eval_all_svm_models_for_all_tasks(data, model_filepath="saved_models/svm")
eval_all_neural_models_for_all_tasks(data, model_filepath="saved_models/neural")
