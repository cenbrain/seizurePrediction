"""
Created on Mon Jun 17 09:39:44 2019

@author: xuyankun
"""

from model import *
from extract_edf import *
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical

sample_rate = 256

def find_seizure(save_path, num_seizure):
    seizure_intervals = []
    dataset = h5py.File(save_path, 'r')
    start = 0
    for i in range(num_seizure):
        tmp_ss = np.min(np.where(dataset['label'][start:] == 1)[0]) + start
        tmp_se = np.min(np.where(dataset['label'][tmp_ss:] == 0)[0]) + tmp_ss
        tmp_interval = [tmp_ss, tmp_se-1]
        seizure_intervals.append(tmp_interval)
        start = tmp_se
    
    return seizure_intervals

def find_non_seizure(save_path, seizure_intervals, num_interval):
    non_seizure_intervals = []
    start = 0
    dataset = h5py.File(save_path, 'r')
    end = len(dataset['label'])
    for i in range(num_interval):
        tmp_ns = start 
        if i == num_interval-1:
            tmp_ne = end
        else:
            
            tmp_ne = seizure_intervals[i][0] - 1
            start = seizure_intervals[i][1] + 1
            
        tmp_interval = [tmp_ns, tmp_ne]
        non_seizure_intervals.append(tmp_interval)

    return non_seizure_intervals

def data_generator(dataset, MIN, non_seizure_intervals, batch_size, time_window, inter_val_sample, pre_val_sample):
    
    while True:
        X = np.zeros((batch_size, dataset['data'].shape[0], time_window * sample_rate))
        y = np.zeros((batch_size,))
        
        while True:
            random.seed()
            inter_interval_idx = random.randint(0, len(non_seizure_intervals)-1)
            random.seed()
            pre_interval_idx = random.randint(0, len(non_seizure_intervals)-2)
    
            preictal_start = non_seizure_intervals[pre_interval_idx][1] - MIN * 60 * sample_rate
            preictal_end = non_seizure_intervals[pre_interval_idx][1] - time_window * sample_rate
            interictal_start = non_seizure_intervals[inter_interval_idx][0]
            interictal_end = non_seizure_intervals[inter_interval_idx][1] - (MIN*60 + time_window) * sample_rate
            
            if (interictal_end - interictal_start) > 0 and preictal_start > 0:  # keep we have valid interval to sample randomly
                break
        
        for i in range(int(batch_size/2)):
            while True:
                random.seed()
                inter_idx = random.randint(interictal_start, interictal_end)
                if inter_idx not in inter_val_sample: # check our training samples if existed in val set
                    if np.sum(np.isnan(dataset['label'][inter_idx : inter_idx + time_window*sample_rate])) == 0: # check there is no nan value
                        X[i] = dataset['data'][:, inter_idx : inter_idx + time_window*sample_rate]
                        y[i] = 0
                        break
        
        for i in range(int(batch_size/2)):
            while True:
                random.seed()
                pre_idx = random.randint(preictal_start, preictal_end)
                if pre_idx not in pre_val_sample: # check our training samples if existed in val set
                    if np.sum(np.isnan(dataset['label'][pre_idx : pre_idx + time_window*sample_rate])) == 0: # check there is no nan value
                        X[i+int(batch_size/2)] = dataset['data'][:, pre_idx : pre_idx + time_window*sample_rate]
                        y[i+int(batch_size/2)] = 1
                        break
                    
        X = X.reshape(X.shape + (1,))
        y = to_categorical(y)
        
        yield X, y

def validation_idx(dataset, MIN, non_seizure_intervals, time_window, num_seizure):
    
    inter_val_sample = []    
    pre_val_sample = []

    for i in range(num_seizure): # preictal
        
        preictal_start = non_seizure_intervals[i][1] - MIN * 60 * sample_rate
        preictal_end = non_seizure_intervals[i][1] - time_window * sample_rate
        
        if preictal_start > 0:
            tmp_pre = random.sample(range(preictal_start,preictal_end), 50)
            for j in range(len(tmp_pre)):
                if np.sum(np.isnan(dataset['label'][tmp_pre[j] : tmp_pre[j] + time_window*sample_rate])) == 0:
                    pre_val_sample.append(tmp_pre[j])
    
    for i in range(num_seizure+1): # interictal
        
        interictal_start = non_seizure_intervals[i][0]
        interictal_end = non_seizure_intervals[i][1] - (MIN*60 + time_window) * sample_rate
        
        if (interictal_end - interictal_start) > 0:
            tmp_inter = random.sample(range(interictal_start,interictal_end), 50)
            for j in range(len(tmp_pre)):
                if np.sum(np.isnan(dataset['label'][tmp_inter[j] : tmp_inter[j] + time_window*sample_rate])) == 0:
                    inter_val_sample.append(tmp_inter[j])
    
    return inter_val_sample, pre_val_sample
    

def validation_data(dataset, MIN, time_window, inter_val_sample, pre_val_sample):
        
    X1 = np.zeros((len(inter_val_sample), dataset['data'].shape[0], time_window * sample_rate))
    X2 = np.zeros((len(pre_val_sample), dataset['data'].shape[0], time_window * sample_rate))
    y1 = np.zeros((len(inter_val_sample),))
    y2 = np.zeros((len(pre_val_sample),))
    
    for i in range(len(inter_val_sample)):
        X1[i] = dataset['data'][:, inter_val_sample[i] : inter_val_sample[i] + time_window*sample_rate]
        y1[i] = 0
        
    for i in range(len(pre_val_sample)):
        X2[i] = dataset['data'][:, pre_val_sample[i] : pre_val_sample[i] + time_window*sample_rate]
        y2[i] = 1
        
    X = np.concatenate((X1,X2), axis=0)
    y = np.concatenate((y1,y2), axis=0)
    
    X = X.reshape(X.shape + (1,))
    y = to_categorical(y) 
    
    return X, y

def plot_roc(x_val, y_val):
    
    n_classes = y_val.shape[1]
    
    y_score = model.predict(x_val)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    
#    plt.plot(fpr["micro"], tpr["micro"], color='orange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    
#    plt.plot(fpr[0], tpr[0], color='red',
#         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_model(path, x, y):
    model.load_weights(path)
    acc = model.evaluate(x, y, verbose=0)
    print("Accuracy: %.2f%%" % (acc[1]*100))
    plot_roc(x, y)
    return acc[1]

if __name__ == '__main__':

    database_path = '/home/sftp/upload/data-MIT' # data path
    patientID = "chb03" 
    save_path = join(database_path, patientID + '.h5')
    seizureInfo, _ = database_summary(database_path, patientID)

    num_seizure = len(seizureInfo)
    seizure_intervals = find_seizure(save_path, num_seizure) 
    non_seizure_intervals = find_non_seizure(save_path, seizure_intervals, len(seizure_intervals)+1)
    
    dataset = h5py.File(save_path, 'r')

    _MINUTES_OF_PREICTAL = 20
    time_window = 30
    batch_size = 32  
    
    inter_val_sample, pre_val_sample = validation_idx(dataset, _MINUTES_OF_PREICTAL, non_seizure_intervals, time_window, num_seizure)

    input_size = (dataset['data'].shape[0], time_window*sample_rate, 1)

      
    model = get_model4(input_size)
#        model.fit_generator(data_generator(dataset, _MINUTES_OF_PREICTAL, non_seizure_intervals, batch_size, time_window, inter_val_sample, pre_val_sample),
#                        steps_per_epoch=500,
#                        epochs=30)
   
    model_path = './model-pre' + str(_MINUTES_OF_PREICTAL) + '-win' + str(time_window) + '.h5'
#        model.save(model_path)
    
    x_val, y_val = validation_data(dataset, _MINUTES_OF_PREICTAL, time_window, inter_val_sample, pre_val_sample)
    
    acc_tmp = evaluate_model(model_path, x_val, y_val)
    acc.append(acc_tmp)
        
        
    
        
        
        
        
        
        
        
        