"""
Created on Tue Jun 11 10:28:51 2019

@author: xuyankun

extract_edf is for extracting all .edf file within each patient 
from CHB-MIT database into single .h5 file
filename is (patientsID + '.h5')
each file has two classes - 'data' and 'label' 
"""


import pyedflib
import os
from os import listdir
from os.path import isfile, isdir, join
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import h5py

sample_rate = 256 # fixed sample rate from device setting

class PreIntData:
    start=0
    end=0
    def __init__(self, s, e):
        self.start=s
        self.end=e

class FileData:
    start=0
    end=0
    nameFile=""
    def __init__(self, s, e, nF):
        self.start=s
        self.end=e
        self.nameFile=nF


def getTime(dateInString):
    '''
    modify the hour(>=24) into (next day + (hour-24)) 
    '''
    time=0
    try:
        time = datetime.strptime(dateInString, '%H:%M:%S')
    except ValueError:
        if('24' in dateInString):
            dateInString = dateInString.replace('24', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=1)
        elif('25' in dateInString):
            dateInString = dateInString.replace('25', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=2)
        elif('26' in dateInString):
            dateInString = dateInString.replace('26', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=3)
        elif('27' in dateInString):
            dateInString = dateInString.replace('27', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=4)    
    return time

def createArrayIntervalData(fSummary):
    seizureInterval = []
    files = []
    oldTime = datetime.min 
    line = fSummary.readline() # read the context of summary line by line
    while(line): 
        data = line.split(':')
        if(data[0] == "File Name"): # when read the line "File Name: chb01_01.edf"
            nF = data[1].strip() # erase " "(space)
            s = getTime((fSummary.readline().split(": "))[1].strip()) 
            # read the line "File Start Time: 11:42:54"
            # and get s = "11:42:54"
            while s < oldTime: # default day is 1, so s need to be added until larger than oldTime
                s = s + timedelta(hours=24)
            oldTime = s # update oldTime for in comparsion with endTime
            endTimeFile = getTime((fSummary.readline().split(": "))[1].strip())
            # read the line "File End Time: 12:42:54"
            # and get endTime = "12:42:54"
            while endTimeFile < oldTime: # if endTime is in next day
                endTimeFile = endTimeFile + timedelta(hours=24)
            oldTime = endTimeFile # update the oldTime for reading next edf file
            files.append(FileData(s, endTimeFile, nF))
            
            for j in range(0, int((fSummary.readline()).split(':')[1])):
                '''
                read next line "Number of Seizures in File:"
                if number is 0, this loop does not work
                if number is larger than 0, do loop
                '''
                secSt = int(fSummary.readline().split(': ')[1].split(' ')[0]) # read Seizure Start Time
                secEn = int(fSummary.readline().split(': ')[1].split(' ')[0]) # read Seizure End Time
                seizureInterval.append(PreIntData(s + timedelta(seconds = secSt), s + timedelta(seconds = secEn)))
        line=fSummary.readline()
        
    fSummary.close()
    
    return seizureInterval, files


def database_summary(database_path,patientID):
    
    summary_path = join(database_path, patientID, (patientID + '-summary.txt'))
    f_summary = open(summary_path,'r')
    seizureInfo, filesInfo = createArrayIntervalData(f_summary)
        
    return seizureInfo, filesInfo


def load_edf(file_path):
    
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_voltage = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        signal_voltage[i, :] = f.readSignal(i)
        
    return signal_voltage, signal_labels


def extract_data(seizureInfo, filesInfo, data_path, save_path):
    
    if os.path.isfile(save_path):
        os.remove(save_path)
    f = h5py.File(save_path,'w')
    
    edf_path = join(data_path, (filesInfo[0].nameFile).split('_')[0], filesInfo[0].nameFile)
    voltage, labels = load_edf(edf_path)
    
    f.create_dataset('data', 
                     shape = (voltage.shape[0], 0),
                     maxshape=(voltage.shape[0], None), 
                     chunks = True, 
                     dtype='float32')
    
    f.create_dataset('label', 
                     shape = (0, ),
                     maxshape=(None, ), 
                     chunks = True, 
                     dtype='float32')
    
    dataset = h5py.File(save_path, 'a')
    last_end = filesInfo[0].start
    
    for edf in filesInfo:
        now_start = edf.start
        time_diff = str(now_start - last_end).split(':')
        time_diff_sec = int(time_diff[0]) * 60 * 60 + int(time_diff[1]) * 60 + int(time_diff[2])
        pad_sample = time_diff_sec * sample_rate
        tmp_len1 = dataset['data'].shape[1]
        dataset['data'].resize([voltage.shape[0], (tmp_len1+pad_sample)])
        dataset['data'][:, tmp_len1:] = 0
        dataset['label'].resize([(tmp_len1+pad_sample), ])
        dataset['label'][tmp_len1:] = np.nan
        
        edf_path = join(data_path, (edf.nameFile).split('_')[0], edf.nameFile)
        signal_voltage, signal_labels = load_edf(edf_path)
        edf_len = signal_voltage.shape[1]
        tmp_len2 = dataset['data'].shape[1]
        dataset['data'].resize([voltage.shape[0], (tmp_len2+edf_len)])
        dataset['data'][:, tmp_len2:] = signal_voltage
        dataset['label'].resize([(tmp_len2+edf_len), ])
        dataset['label'][tmp_len2:] = 0
        
        for seizure in seizureInfo:
            if seizure.start >= edf.start and seizure.end <= edf.end:
                ss_diff = str(seizure.start - edf.start).split(':') # ss = seizure_start
                ss_diff_sec = int(ss_diff[0]) * 60 * 60 + int(ss_diff[1]) * 60 + int(ss_diff[2])
                se_diff = str(seizure.end - edf.start).split(':') # se = seizure_end
                se_diff_sec = int(se_diff[0]) * 60 * 60 + int(se_diff[1]) * 60 + int(se_diff[2])
                dataset['label'][(tmp_len2 + ss_diff_sec*sample_rate):(tmp_len2 + se_diff_sec*sample_rate)] = 1
        
        last_end = edf.end
        
    f.close()
    


if __name__ == '__main__':

    database_path = '/home/sftp/upload/data-MIT' # data path
    patients = ["chb01","chb02","chb03","chb05","chb06","chb07","chb08","chb10"]
    for i in range(len(patients)):    
        save_path = join(database_path, patients[i] + '.h5')
    
        seizureInfo, filesInfo = database_summary(database_path, patients[i])
        extract_data(seizureInfo, filesInfo, database_path, save_path)
    
    
    
    