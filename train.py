# -*- coding: utf-8 -*

import autokeras as ak
from numpy import *
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import  csv
import random
from test import *
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
#random.seed ( 8 )
import tool

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
def topredict(List):
    predicted1 = []
    for ii in List:

        if ii >= 0.5:
            predicted1.extend('1')

        else:
            predicted1.extend('0')

    return list(map(float, predicted1))
def train(SampleFeature,Ncv):


    
    data = []

    num=int(len(SampleFeature)/2)

    data1 = ones((1, num), dtype=int)
    data2 = zeros((1, num))

    data.extend(data1[0])

    data.extend(data2[0])


    SampleLabel = data



    print('Start training the model.')
    
    cv = StratifiedKFold(n_splits=Ncv)

    SampleFeature = np.array(SampleFeature)
    SampleLabel = np.array(SampleLabel)
    permutation = np.random.permutation(SampleLabel.shape[0])
    SampleFeature = SampleFeature[permutation, :]
    SampleLabel = SampleLabel[permutation]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    num=0
    result=[]
    for train, test in cv.split(SampleFeature, SampleLabel):
        x_train, x_val, y_train, y_val = train_test_split(SampleFeature[train], SampleLabel[train], test_size=0.2)
        x_test = SampleFeature[test]
        y_test=SampleLabel[test]
        clf = ak.StructuredDataClassifier()
        clf.fit(x_train, y_train,validation_data=(x_val, y_val),verbose=0)
        best_model = clf.tuner.get_best_model()
        path = "best_"+str(num) +"_model"
        best_model.save(path, save_format="tf")
        loaded_model = load_model(path, custom_objects=ak.CUSTOM_OBJECTS)
        predict = loaded_model.predict(x_test).tolist()
        prob=[x[0] for x in predict]
        pred=topredict(prob)
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)


def block(path,List):
    SampleFeature = []
    ReadMyCsv(SampleFeature, "feature_" + path + ".csv")
    try:
        os.mkdir('./'+str(List))
    except:
        pass
    retval = os.getcwd()

    os.chdir(retval + '/'+str(List))
    train(SampleFeature,List)
    os.chdir(retval)

if __name__ == "__main__":

    path = "GIN_64_M2V_64_kmer_kmerSparseMatrix"

    List=[2,5,10]
    for i in List:
        block(path,i)

    
