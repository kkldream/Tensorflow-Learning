import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


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

def plot_confusion_matrix(y_test, y_pred, classes_num, figsize=None, dpi=None):
    # Compute confusion matrix
    # cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), predictions_classes))
    cnf_matrix = (confusion_matrix(y_test, y_pred))

    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize, dpi=dpi)
    tag_indexs_str = [f"{i}" for i in range(classes_num)]
    # Plot non-normalized confusion matrix
    _plot_confusion_matrix(cnf_matrix, classes=tag_indexs_str,
                        title='Confusion matrix')
    #plt.figure()
    # Plot normalized confusion matrix
    #_plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
    #                      title='Normalized confusion matrix')
    #plt.figure()
    plt.show()
