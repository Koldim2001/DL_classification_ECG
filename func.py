import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools

import os
import random

import seaborn as sns
import torch
import mlflow
from matplotlib import pyplot as plt
import numpy as np 
import torch.nn as nn 
import mlflow.pytorch
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import pyedflib
import scipy



def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def discrete_signal_resample(signal, time, new_sampling_rate):
    ## Производит ресемплирование
    # Текущая частота дискретизации
    current_sampling_rate = 1 / np.mean(np.diff(time))

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * new_sampling_rate / current_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)
    new_time = np.linspace(time[0], time[-1], num_points_new)

    return new_signal, new_time


def plot_signals_with_subplot(signal):
    subplot_shape = (signal.shape[0],)
    # subplot_shape - кортеж, указывающий количество строк в сетке subplot
    
    # Проверяем, достаточно ли сигналов для создания всех подграфиков
    if signal.shape[0] < subplot_shape[0]:
        raise ValueError("Недостаточно сигналов для создания всех подграфиков в сетке subplot.")
    
    # Создаем сетку subplot
    fig, axes = plt.subplots(subplot_shape[0], figsize=(8, 7))
    
    # Регулируем расположение графиков
    plt.tight_layout()
    
    # Строим графики для каждого сигнала
    for i in range(subplot_shape[0]):
        axes[i].plot(signal[i])
        #axes[i].set_title(f'Signal {i+1}')
    
    # Показываем графики
    plt.show()


def get_full_signal(path, Fs_new=500, f_sreza=0.7):
    # Открываем EDF файл
    f = pyedflib.EdfReader(path)

    # Получаем информацию о каналах
    num_channels = f.signals_in_file
    channels = f.getSignalLabels()

    # Читаем данные по каналам
    raw_data = []
    for i in range(num_channels):
        channel_data = f.readSignal(i)
        raw_data.append(channel_data)

    # Получаем частоту дискретизации
    fd = f.getSampleFrequency(0)

    # Закрываем файл EDF после чтения
    f.close()
    raw_data = np.array(raw_data)

    # Создаем DataFrame
    df = pd.DataFrame(data=raw_data.T,   
            index=range(raw_data.shape[1]), 
            columns=channels)  

    # Переименование столбцов при необходимости:
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
        channels = df.columns

    # Создание массива времени    
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    # Ресемлинг:
    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        new_ecg, time_new = discrete_signal_resample(sig, t, Fs_new)
        df_new[graph] = pd.Series(new_ecg) 
    df = df_new.copy()

    # ФВЧ фильтрация артефактов дыхания:
    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        sos = scipy.signal.butter(1, f_sreza, 'hp', fs=Fs_new, output='sos')
        avg = np.mean(sig)
        filtered = scipy.signal.sosfilt(sos, sig)
        filtered += avg
        df_new[graph] = pd.Series(filtered)
    df = df_new.copy()

    # Выбор нужных столбцов
    selected_columns = ['ECG I', 'ECG II', 'ECG V1', 'ECG V2', 'ECG V3', 'ECG V4', 'ECG V5', 'ECG V6']
    selected_data = df[selected_columns]

    # Преобразование данных в массив NumPy
    numpy_array = selected_data.values.T
    return numpy_array
   

def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def weighted_avg_f1(output, labels):
    """Функция расчета weighted avg F1-меры"""
    # Преобразование списков в массивы numpy
    output = np.array(output)
    labels = np.array(labels)

    # Создание тензоров PyTorch
    output = torch.tensor(output)
    labels = torch.tensor(labels)

    predictions = torch.argmax(output, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    weighted_f1 = f1_score(labels, predictions, average='weighted')
    weighted_f1 = np.nan_to_num(weighted_f1, nan=0.0)  # Замена NaN на 0 при делении на 0

    return weighted_f1




def accuracy(output,labels):
    """Функция расчета accuracy"""
    # Преобразование списков в массивы numpy
    output = np.array(output)
    labels = np.array(labels)

    # Создание тензоров PyTorch
    output = torch.tensor(output)
    labels = torch.tensor(labels)

    predictions = torch.argmax(output,dim=1)
    correct = (predictions == labels).sum().cpu().numpy()
    return correct / len(labels)


def evaluate_model(model, data_loader):
    """Функция для логирования артефактов в MLflow"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data['signal'].to(device).float(), data['category'].to(device)
 
            outputs, _, _ = model(inputs.transpose(1,2))
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)


    # Сохранение отчета в текстовый файл
    report = classification_report(true_labels, predicted_labels)
    output_file = "classification_report.txt"
    with open(output_file, "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")
        
    # Save confusion matrix as CSV artifact
    df = pd.DataFrame(cm)
    new_columns = [f"predicted class {i}" for i in range(df.shape[1])]

    # Установка новых названий столбцов
    df.columns = new_columns

    # Добавление индексов
    df.insert(0, "real\pred", [f"real class {i}" for i in range(df.shape[0])])

    # Сохранение измененной таблицы 
    df.to_csv("confusion_matrix.csv", index=False)
    mlflow.log_artifact("confusion_matrix.csv")
    os.remove("confusion_matrix.csv")

    


def train_pointnet(model_pointnet, dataloader_train, dataloader_val, batch_size, 
                     name_save, start_weight=None, 
                     name_experiment=None, lr=1e-4, epochs=100,
                     scheduler=True, scheduler_step_size=10, dataset_name=None,
                     f_sampling=700, seed=42, n_points=512,
                     normalize='Centering and max value scaling', gamma=0.5, noise_std=0):
    """Обучение классификационной сверточной сети

    Args:
        model_pointnet: Класс модели pytorch

        dataloader_train: Обучающий даталоудер

        dataloader_val: Валидационный даталоудер

        batch_size: Размер одного батча

        name_save: Имя модели для сохранения в папку models

        start_weight: Если указать веса, то сеть будет в режиме fine tune. Defaults to None.

        name_experiment:  Имя эксперимента для MLflow. Нужно при mlflow_tracking=True. Defaults to None.
        
        lr: Скорость обучения. Defaults to 1e-4.

        epochs: Число эпох обучения. Defaults to 100.

        scheduler (bool): Включение/выключение lr шедулера. Defaults to True.

        scheduler_step_size (int): Шаг шедулера при scheduler=True. Defaults to 10.

        dataset_name: Имя датасета для логирования в MLflow. Defaults to None.

        seed (int): Seed рандома. Defaults to 42.
        
        normalize: Вид нормализации

        gamma: Величина коэффициента lr шедулера 

        noise_std: Величина std шума на трейне

    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if dataset_name != None:
        dataset_name = dataset_name.split('/')[-1]

    directory_save = 'models'

    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    
    if name_experiment == None:
        name_experiment = name_save
    with mlflow.start_run(run_name=name_experiment) as run:
        
        model = model_pointnet()
        mlflow.log_param("Model", 'PointNet')
        if start_weight != None:
            model.load_state_dict(torch.load(start_weight))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if scheduler:
            gamma_val = gamma
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=scheduler_step_size,
                                                        gamma=gamma_val)
        model = model.to(device)
        
        mlflow.log_param("Normalize", normalize)
        mlflow.log_param("Training random noise std", noise_std)
        mlflow.log_param("Input shape", f'torch.Size([batch_size, {n_points}, 3])')
        mlflow.log_param("F sampling ECG", f_sampling)
        mlflow.log_param("Points samping", n_points)
        if scheduler:
            mlflow.log_param("scheduler", 'On')
            mlflow.log_param("scheduler_step_size", scheduler_step_size)
            mlflow.log_param("scheduler_gamma", gamma_val)
        else:
            mlflow.log_param("scheduler", 'Off')

        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", 'Adam')
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("loss", 'NLLLoss + 0.0001*Loss_reg')
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("seed", seed)
        if start_weight != None:
            mlflow.log_param("Fine-tuning", True)
        else:
            mlflow.log_param("Fine-tuning", False)

        max_epoch_f1_val = 0
        for epoch in range(epochs): 
            model.train()
            running_loss = 0.0
            all_outputs = []
            all_targets = []
            for i, data in enumerate(dataloader_train, 0):
                inputs, labels = data['signal'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
                
                loss = pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                all_outputs.extend(outputs.cpu().detach().numpy())
                all_targets.extend(labels.cpu().numpy())

                
            train_epoch_loss = running_loss / len(dataloader_train)
            train_epoch_acc = accuracy(all_outputs, all_targets)
            train_epoch_f1 = weighted_avg_f1(all_outputs, all_targets)

            mlflow.log_metric("train_epoch_accuracy", train_epoch_acc, step=(epoch+1))
            mlflow.log_metric("train_epoch_loss", train_epoch_loss, step=(epoch+1))
            mlflow.log_metric("train_epoch_f1", train_epoch_f1, step=(epoch+1))


            # validation
            model.eval()
            with torch.no_grad():
                running_loss = 0.0
                all_outputs = []
                all_targets = []
                for data in dataloader_val:
                    inputs, labels = data['signal'].to(device).float(), data['category'].to(device)
                    outputs, m3x3, m64x64 = model(inputs.transpose(1,2))
                    loss = pointnetloss(outputs, labels, m3x3, m64x64)

                    running_loss += loss.item()

                    all_outputs.extend(outputs.cpu().detach().numpy())
                    all_targets.extend(labels.cpu().numpy())


            val_epoch_loss = running_loss / len(dataloader_val)
            val_epoch_acc = accuracy(all_outputs, all_targets)
            val_epoch_f1 = weighted_avg_f1(all_outputs, all_targets)

            mlflow.log_metric("validation_epoch_accuracy", val_epoch_acc, step=(epoch+1))
            mlflow.log_metric("validation_epoch_loss", val_epoch_loss, step=(epoch+1))
            mlflow.log_metric("validation_epoch_f1", val_epoch_f1, step=(epoch+1))

            if scheduler:
                lr_scheduler.step()

            # Вывод значения функции потерь на каждой 5 эпохе
            if ((epoch+1) % 5 == 0) or epoch==0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_epoch_loss:.4f},'
                    f' Train Aсс: {train_epoch_acc:.4f}'
                    f' Val Loss: {val_epoch_loss:.4f}, Val Acc:{val_epoch_acc:.4f} ')


            if epoch >= 1 and val_epoch_f1 > max_epoch_f1_val:
                max_epoch_f1_val = val_epoch_f1
                acc_model = val_epoch_acc
                model_to_save = model
                epoch_best = epoch + 1
                name_save_model = directory_save + '/' + name_save +'.pth'
                torch.save(model_to_save.state_dict(), name_save_model)
                evaluate_model(model=model, data_loader=dataloader_val)
            
        print('Обучение завершено')
        print(f'Сохранена модель {name_save_model} с лучшим weighted avg f1 на валидации = {max_epoch_f1_val}')  
        print('Accuracy данной модели равно', acc_model) 

        mlflow.log_metric("max f1 saved model", max_epoch_f1_val)
        mlflow.log_metric("accuracy of model", acc_model)
        mlflow.log_metric("epoch of save", epoch_best)

        mlflow.log_artifact(name_save_model)
        mlflow.log_artifact('model.py')



###########################################

