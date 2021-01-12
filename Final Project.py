import pandas as pd
import numpy as np
import time
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import metrics
from matplotlib import pyplot as plt
import itertools


start = time.time()


pd.set_option('display.max_columns', None)
df2 = pd.read_csv('edited.csv')

def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        See full source and example:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

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
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="gray" if cm[i, j] > thresh else "black",fontsize=16)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

df2.info()
maxcol=df2.max()
print(maxcol)
X=df2.drop(columns=["Great_App"])
y=df2["Great_App"]


X_train = pd.read_csv('xtrain.csv')
X_test = pd.read_csv('xtest.csv')
y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')

RFModel = pickle.load(open('RandomForest_model.sav', 'rb'))
result = RFModel.score(X_test, y_test)
print('Accuracy',result)


y_pred = RFModel.predict(X_test)
cmatrix_rf = confusion_matrix(y_test,y_pred)

print('Prediction : ', y_pred)
print('Confusion matrix : \n',cmatrix_rf)
print('ROC AOC SCORE : ',roc_auc_score(y_test,y_pred,  average=None))

ConfusionMatrixDisplay(cmatrix_rf).plot()
plt.show()

print("Prediction : ", y_pred)


print('Classification Report : \n')
print(classification_report(y_test, y_pred))




model = pickle.load(open('DecisionTree_model.sav', 'rb'))
result = model.score(X_test, y_test)
print("Accuracy :",result)

predictions = model.predict(X_test)
cmatrix_dt = confusion_matrix(y_test,predictions)
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print('ROC AOC SCORE : ',roc_auc_score(y_test, predictions))
print('Confudsion matrix : \n',cmatrix_dt)
ConfusionMatrixDisplay(cmatrix_dt).plot()
plt.show()
print('Classification Report : \n')
print(classification_report(y_test, predictions))


end = time.time()
print(f"Runtime of the program is {end - start}")


average_precision_rf = average_precision_score(y_test, y_pred)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision_rf))

prc_rf = plot_precision_recall_curve(RFModel, X_test, y_test)
prc_rf.ax_.set_title('Random Forest Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision_rf))
plt.show()

average_precision_dt = average_precision_score(y_test, predictions)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision_dt))
prc_dt = plot_precision_recall_curve(model, X_test, y_test)
prc_dt.ax_.set_title('Decision Tree Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision_dt))
plt.show()



roc_rf = metrics.plot_roc_curve(RFModel, X_test, y_test)
plt.show()

metrics.plot_roc_curve(model, X_test, y_test)
plt.show()



from yellowbrick.model_selection import FeatureImportances
fi_rf = FeatureImportances(RFModel)
fi_rf.fit(X, y)
fi_rf.show()

fi_dt = FeatureImportances(model)
fi_dt.fit(X, y)
fi_dt.show()


