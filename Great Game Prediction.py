import pandas as pd
import numpy as np
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from matplotlib import pyplot as plt
import itertools

pd.set_option('display.max_columns', None)
df = pd.read_csv('appstore_games.csv')

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
      #  plt.show()
def in_app_p(row):

    x=row["In-app Purchases"]
    if isinstance(x,np.float64) or pd.isnull(x):
        row["In_App_Count"]= 0
        row["In_App_Max"]  = 0
    else:
        x_list=[float(x) for x in row["In-app Purchases"].split(",")]
        row["In_App_Count"]=len(x_list)
        row["In_App_Max"]=max(x_list)
    return row

def languages(row):
    if pd.isnull(row["Languages"]):
        row["Languages"]="EN"
    if "EN" in row["Languages"]:
        row["Language_EN"]=1
    else:
        row["Language_EN"]=1
    row["Languages_Count"]=len(row["Languages"].split(","))

    return row

def genres(row):
    row["Genres_Count"] = len(row["Genres"].split())
    return row


df2 = df.loc[df["User Rating Count"]>=5,:].copy()

df2 = df2.assign(Great_App=lambda x: np.where(x["Average User Rating"]>=4.5,1,0))\
         .assign(Subtitle_Available=lambda x: np.where(x["Subtitle"].isnull(),0,1))\
         .assign(Free_Price=lambda x: np.where(x["Price"]==0,1,0))\
         .assign(Age_Rating=lambda x: x["Age Rating"].str.replace("+","").astype(int))\
         .assign(Description_Length=lambda x: x["Description"].str.len())\
         .apply(genres,axis=1)\
         .apply(languages,axis=1)\
         .apply(in_app_p,axis=1)\
         .drop(columns=["URL","ID","Name","Subtitle","Icon URL","Primary Genre","Price", "In-app Purchases","Developer","Description","Languages","Average User Rating",
                        "Original Release Date","Current Version Release Date","Genres","Age Rating"])

print(df2.head())
df2.to_csv('edited.csv', index=False)
X=df2.drop(columns=["Great_App"])
y=df2["Great_App"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

X_train.to_csv('xtrain.csv',index=False)
X_test.to_csv('xtest.csv',index=False)
y_train.to_csv('ytrain.csv',index=False)
y_test.to_csv('ytest.csv',index=False)


rf = RandomForestClassifier(n_estimators=100)

param_grid = {
    'n_estimators': [300, 500, 750],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [10,15,20],
    'criterion' :['gini']
}

cv_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring="roc_auc", cv= 5)

cv_rf.fit(X_train,y_train)



final_model = cv_rf.best_estimator_
start = time.time()
final_model.fit(X_train,y_train)
end = time.time()
filename_RandomForest = 'RandomForest_model.sav'
pickle.dump(final_model, open(filename_RandomForest, 'wb'))


y_pred = final_model.predict(X_test)
cmatrix_rf = confusion_matrix(y_test,y_pred)



print(cmatrix_rf)
print(roc_auc_score(y_test,final_model.predict(X_test)))
plot_confusion_matrix(cmatrix_rf, classes=['True', 'False'], title='Random Forest Confusion Matrix')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

model = DecisionTreeClassifier()
startdt = time.time()
model.fit(X_train, y_train)
enddt = time.time()
filename_DecisionTree = 'DecisionTree_model.sav'
pickle.dump(model, open(filename_DecisionTree, 'wb'))


predictions = model.predict(X_test)
cmatrix_dt = confusion_matrix(y_test,model.predict(X_test))
print(roc_auc_score(y_test, model.predict(X_test)))
print(cmatrix_dt)
plot_confusion_matrix(cmatrix_dt, classes=['True', 'False'], title='Decision Tree Confusion Matrix')




print(f"Runtime of the Random Forest is {end - start}")
print(f"Runtime of the Decision tree is {enddt - startdt}")
