import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime
import random
import time
import os
import pandas as pd

# confusion_matrix = confusion_matrix

tag_indexs = ("摩羯座", "水瓶座", "雙魚座", "牡羊座", "金牛座", "雙子座", "巨蟹座", "獅子座", "處女座", "天秤座", "天蠍座", "射手座")

constellation_date = (
    ("12/22", "01/19"), # 摩羯座
    ("01/20", "02/19"), # 水瓶座
    ("02/20", "03/20"), # 雙魚座
    ("03/21", "04/20"), # 牡羊座
    ("04/21", "05/20"), # 金牛座
    ("05/21", "06/21"), # 雙子座
    ("06/22", "07/22"), # 巨蟹座
    ("07/23", "08/22"), # 獅子座
    ("08/23", "09/22"), # 處女座
    ("09/23", "10/22"), # 天秤座
    ("10/23", "11/21"), # 天蠍座
    ("11/22", "12/21")  # 射手座
)

def timeStrToDay(timeStr): # ex: timeStr="11/11"
    struct_time = time.strptime(timeStr, "%m/%d")
    tm_yday = struct_time.tm_yday
    return tm_yday
    
def dayToTimeStr(day, year=2022):
    first_day = datetime.datetime(year, 1, 1)
    add_day = datetime.timedelta(days=day - 1)
    return datetime.datetime.strftime(first_day + add_day, "%m/%d")
    
def timeStrToConstellation(timeStr):
    tm_yday = timeStrToDay(timeStr)
    for i, data in enumerate(constellation_date):
        if i == 0 and (tm_yday >= timeStrToDay(data[0]) or tm_yday <= timeStrToDay(data[1])):
            return 0
        elif tm_yday >= timeStrToDay(data[0]) and tm_yday <= timeStrToDay(data[1]):
            return i

def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    # cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), predictions_classes))
    cnf_matrix = (confusion_matrix(y_test, y_pred))

    np.set_printoptions(precision=2)

    plt.figure()
    tag_indexs_str = [f"{i}" for i in range(len(tag_indexs))]
    # Plot non-normalized confusion matrix
    _plot_confusion_matrix(cnf_matrix, classes=tag_indexs_str,
                        title='Confusion matrix')
    #plt.figure()
    # Plot normalized confusion matrix
    #_plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
    #                      title='Normalized confusion matrix')
    #plt.figure()
    plt.show()