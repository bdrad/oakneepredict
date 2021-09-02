#Uploading data (KL234 unbalanced all, KL234 top 10 with imaging, KL234 top 7 without imaging)
from google.colab import files
uploaded = files.upload()

#Saving original state random # seeds.
trainval_test_rand_seed=45
train_val_rand_seed=23

#confirm the current directory
import os
os.getcwd()

import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, KFold

#SVM and Random Forest Classifiers

dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed.csv') #model with 112 predictors
#dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_3.csv') #hybrid model with womac pain
#dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_4(noimage).csv') #no imaging 8 parameters

dataset=dataset.dropna()
print(dataset.shape)

X = dataset.drop(['id','KL234'], axis=1)
y = dataset['KL234']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=101,stratify=y)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svclassifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#svcclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred_svm = svclassifier.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier()
rfclassifier.fit(X, y)
y_pred_rf = rfclassifier.predict(X_test)

#from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))

print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))

#print(y_test)





### XGBoost for cross-validation and ROC with confidence intervals for the validation sets

#dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed.csv') #model with 112 predictors
dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_3.csv') #hybrid model with womac pain
#dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_4(noimage).csv') #no imaging 8 parameters

### XGBoost for cross-validation and ROC with confidence intervals for the validation sets continued
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score,precision_score, f1_score

dataset_num_col=dataset.shape[1]
X = dataset.iloc[:, 2:dataset_num_col] #obtan values all the way up until colum n113
y = dataset.iloc[:, 1]

# split data into train and validation (labeled test) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=101)
folds = [(train,test) for train, test in cv.split(X_train, y_train)]

# Parameters
metrics = ['auc', 'fpr', 'tpr', 'thresholds', 'aucpr']
results = {
    'train': {m:[] for m in metrics},
    'val'  : {m:[] for m in metrics},
    'test' : {m:[] for m in metrics}
}
params = {'n_estimators': 100, 
          'learning_rate':0.2,
          'lambda': 20,
          'eta': 0.2,
          'gamma': 1.5,
          'min_child_weight': 3,  
          'max_depth': 8,  
          'min_child_weight': 3,
          'reg_lambda': 0,
          'reg_alpha': 1,
          'max_delta_step': 100,
          'subsample': 0.5,
          'colsample_bytree': 0.6,
          'objective': 'binary:logistic',
          'eval_metric': 'auc'}

dtest = xgb.DMatrix(X_test, label=y_test)
for train, test in tqdm(folds, total=len(folds)):
    dtrain = xgb.DMatrix(X_train.iloc[train,:], label=y_train.iloc[train])
    dval   = xgb.DMatrix(X_train.iloc[test,:], label=y_train.iloc[test])
    model  = xgb.train(
        dtrain                = dtrain,
        params                = params, 
        evals                 = [(dtrain, 'train'), (dval, 'val')],
        num_boost_round       = 1000,
        verbose_eval          = False,
        early_stopping_rounds = 10,
    )
    sets = [dtrain, dval, dtest]
    for i,ds in enumerate(results.keys()):
        y_preds              = model.predict(sets[i])
        labels               = sets[i].get_label()
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        results[ds]['fpr'].append(fpr)
        results[ds]['tpr'].append(tpr)
        results[ds]['thresholds'].append(thresholds)
        results[ds]['auc'].append(roc_auc_score(labels, y_preds))

from sklearn.metrics import confusion_matrix
model_threshold = 0.30 #important threshold number that's currently empirically set.
y_pred=model.predict(sets[1])>model_threshold
y_true=sets[1].get_label()
#print(y_pred)
#print(y_true)

conf_matrix = confusion_matrix(y_true, y_pred)
# Printing the Confusion Matrix

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#print(classification_report(y_true1,y_pred1_th, target_names=target_names))

kind = 'val'
c_fill      = 'rgba(52, 152, 219, 0.2)'
c_line      = 'rgba(52, 152, 219, 0.5)'
c_line_main = 'rgba(41, 128, 185, 1.0)'
c_grid      = 'rgba(189, 195, 199, 0.5)'
c_annot     = 'rgba(149, 165, 166, 0.5)'
c_highlight = 'rgba(192, 57, 43, 1.0)'
fpr_mean    = np.linspace(0, 1, 100)
interp_tprs = []
for i in range(100):
    fpr           = results[kind]['fpr'][i]
    tpr           = results[kind]['tpr'][i]
    interp_tpr    = np.interp(fpr_mean, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
tpr_mean     = np.mean(interp_tprs, axis=0)
tpr_mean[-1] = 1.0
tpr_std      = 2*np.std(interp_tprs, axis=0)
tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
tpr_lower    = tpr_mean-tpr_std
auc          = np.mean(results[kind]['auc'])
fig = go.Figure([
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_upper,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'upper'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_lower,
        fill       = 'tonexty',
        fillcolor  = c_fill,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'lower'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_mean,
        line       = dict(color=c_line_main, width=2),
        hoverinfo  = "skip",
        showlegend = True,
        name       = f'AUC: {auc:.3f}')
])
fig.add_shape(
    type ='line', 
    line =dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig.update_layout(
    template    = 'plotly_white', 
    title_x     = 0.5,
    xaxis_title = "1 - Specificity",
    yaxis_title = "Sensitivity",
    width       = 800,
    height      = 800,
    legend      = dict(
        yanchor="bottom", 
        xanchor="right", 
        x=0.95,
        y=0.01,
    )
)
fig.update_yaxes(
    range       = [0, 1],
    gridcolor   = c_grid,
    scaleanchor = "x", 
    scaleratio  = 1,
    linecolor   = 'black')
fig.update_xaxes(
    range       = [0, 1],
    gridcolor   = c_grid,
    constrain   = 'domain',
    linecolor   = 'black')







#### XGBoost - 112 predictors from Optimized Version from April 2021 ####

#Convert data into dataframe and check data shape for confirmation
dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed.csv') #model with 112 predictors
dataset.shape

#XGboost param
param = {'n_estimators': 100, 
          'learning_rate':0.2,
          'lambda': 20,
          'eta': 0.2,
          'gamma': 1.5,
          'min_child_weight': 3,  
          'max_depth': 8,  
          'objective': 'binary:logistic',
          'min_child_weight': 3,
          'reg_lambda': 0,
          'reg_alpha': 1,
          'max_delta_step': 100,
          'subsample': 0.5,
          'colsample_bytree': 0.6,
          'eval_metric': 'auc'}

###Dataset splitting and preprocessing ###
roc_auc_all_1=[]
dataset_num_col=dataset.shape[1]

#Organize Dataset & Generate Train/Test/Val split.
X = dataset.iloc[:, 2:dataset_num_col].values #obtan values all the way up until colum n113
y = dataset.iloc[:, 1].values

#Feature extraction
feature_names = dataset.head()
featurenames3 = list(feature_names)
featurenames4 = featurenames3[2:dataset_num_col] #this is the final list of features.

#Splitting Data
from sklearn.model_selection import train_test_split
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, y, test_size=0.2, random_state=trainval_test_rand_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=train_val_rand_seed)

#Setting the data as featurenames4
D_train = xgb.DMatrix(X_train, label=Y_train, feature_names=featurenames4)
D_val = xgb.DMatrix(X_val, label=Y_val, feature_names=featurenames4)
D_test = xgb.DMatrix(X_test, label=Y_test, feature_names=featurenames4)

### MODEL TRAINING ###

#Other Hyperparameters.
num_round = 20
watchlist = [(D_val,'val'), (D_train,'train')]
#watchlist = [(Y_val,'val'), (Y_train,'train')]
steps = 30 # The number of training iterations

#Model Training
evals_result = {}
model = xgb.train(param, D_train, num_round, watchlist, evals_result=evals_result)

from xgboost import cv
xgb_cv = cv(dtrain=D_train, params=param, nfold=10,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
print(xgb_cv)

### Model Evaluation & VISUALIZATION - ROC ###
#model.predict(D_test) #tell u the prediction values
prediction =  model.predict(D_test)
#prediction =  model.predict(X_test)
#np.savetxt('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/pred_1_30.csv', prediction, delimiter=",")

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

fpr1, tpr1, threshold1 = roc_curve(Y_test, prediction)
roc_auc1 = auc(fpr1, tpr1)
#print(roc_auc1)
roc_auc_all_1.append(roc_auc1)

#Visualization - ROC
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
lw = 2
ax.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='ROC curve for F1 (area = %0.2f)' % roc_auc1, linestyle="-.")

ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve', fontsize=14)
#prediction



### VISUALIZATION - Feature Importance ###
plot_importance(model, max_num_features=20)

#Saving figures
pyplot.savefig('featureimportane all 114.png',dpi=500)
pyplot.show()
from google.colab import files
files.download('featureimportane all 114.png')

### VISUALIZATION - Confusion Matrix and other accuracy measures ###
from sklearn.metrics import confusion_matrix
model_threshold = 0.3
y_pred1=model.predict(D_test)
y_pred1_th=model.predict(D_test)>model_threshold
y_true1=Y_test
#print(y_pred)
#print(y_true)

conf_matrix = confusion_matrix(y_true1, y_pred1_th)
# Printing the Confusion Matrix

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)


#Saving figures
plt.savefig('cm all 114.png',dpi=500)
plt.show()
from google.colab import files
files.download('cm all 114.png')

target_names = ['KL Grade 0-1 (All 114 model)', 'KL Grade 2+ (All 114 model)']
print(classification_report(y_true1,y_pred1_th, target_names=target_names))

# compute and print accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_true1, y_pred1_th)))









#HYBRID 10 MODELS from April 2021
param = {'n_estimators': 100,
     'lambda': 20,
     'eta': 0.3,
     'gamma': 0,
     'min_child_weight': 1,  
     'max_depth': 10,  
     'objective': 'binary:logistic',
     'min_child_weight': 4,
     'reg_lambda': 0,
     'reg_alpha': 2,
     'max_delta_step': 100,
     'subsample': 0.6,
     'colsample_bytree': 0.6,
     'eval_metric': 'auc'}

dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_3.csv') #hybrid model with womac pain
dataset.shape

###Dataset splitting and preprocessing ###
roc_auc_all_2=[]
dataset_num_col=dataset.shape[1]

#Organize Dataset & Generate Train/Test/Val split.
X = dataset.iloc[:, 2:dataset_num_col].values #obtan values all the way up until colum n113
y = dataset.iloc[:, 1].values

#Feature extraction
feature_names = dataset.head()
featurenames3 = list(feature_names)
featurenames4 = featurenames3[2:dataset_num_col] #this is the final list of features.

#Splitting Data
from sklearn.model_selection import train_test_split
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, y, test_size=0.2, random_state=trainval_test_rand_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=train_val_rand_seed)

#Setting the data as featurenames4
D_train = xgb.DMatrix(X_train, label=Y_train, feature_names=featurenames4)
D_val = xgb.DMatrix(X_val, label=Y_val, feature_names=featurenames4)
D_test = xgb.DMatrix(X_test, label=Y_test, feature_names=featurenames4)

### MODEL TRAINING ###

#Other Hyperparameters.
num_round = 20
watchlist = [(D_val,'val'), (D_train,'train')]
#watchlist = [(Y_val,'val'), (Y_train,'train')]
steps = 30 # The number of training iterations

#Model Training
evals_result = {}
model = xgb.train(param, D_train, num_round, watchlist, evals_result=evals_result)

from xgboost import cv
xgb_cv = cv(dtrain=D_train, params=param, nfold=10,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
print(xgb_cv)

#xgb=XGBClassifier()
#XGBClassifier(param)
#model = xgb.fit(X_train, Y_train)

#model.save_model('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/1_30_opt1.model') # save model

### Model Evaluation & VISUALIZATION - ROC ###
#model.predict(D_test) #tell u the prediction values
prediction =  model.predict(D_test)
#prediction =  model.predict(X_test)
#np.savetxt('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/pred_1_30.csv', prediction, delimiter=",")

fpr2, tpr2, threshold2 = roc_curve(Y_test, prediction)
roc_auc2 = auc(fpr2, tpr2)
print(roc_auc2)
roc_auc_all_2.append(roc_auc2)

#Visualization - ROC
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
lw = 2
ax.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='ROC curve for F1 (area = %0.2f)' % roc_auc1, linestyle="-")
ax.plot(fpr2, tpr2, color='c',
         lw=lw, label='ROC curve for F1 (area = %0.2f)' % roc_auc2, linestyle="-.")

ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve', fontsize=14)
#prediction

print(threshold2)
print(tpr2)
print(fpr2)
print(tpr2+(1-fpr2))

### VISUALIZATION - Feature Importance ###
plot_importance(model, max_num_features=20)

#Saving figures
pyplot.savefig('featureimportane with imaging.png',dpi=500)
pyplot.show()
from google.colab import files
files.download('featureimportane with imaging.png')

### VISUALIZATION - Confusion Matrix and other accuracy measures ###
from sklearn.metrics import confusion_matrix
model_threshold = 0.3
y_pred2 = model.predict(D_test)
y_pred2_th=model.predict(D_test)>model_threshold
y_true2=Y_test
#print(y_pred)
#print(y_true)

conf_matrix = confusion_matrix(y_true2, y_pred2_th)
# Printing the Confusion Matrix

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)

#Saving figures
from google.colab import files
plt.savefig('cm with imaging.png',dpi=500)
plt.show()
files.download('cm with imaging.png')

from sklearn.metrics import classification_report
target_names = ['KL Grade 0-1 (Clinical model with imaging)', 'KL Grade 2+ (Clinical model with imaging)']
print(classification_report(y_true2,y_pred2_th, target_names=target_names))

# compute and print accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_true2, y_pred2_th)))





# no imaging datsets from April 2021::
param = {'n_estimators': 100,
    'lambda': 20,
    'eta': 0.4,
    'gamma': 0,
    'min_child_weight': 5,  
    'max_depth': 10,  
    'objective': 'binary:logistic',
    'reg_lambda': 1,
    'reg_alpha': 2,
    'max_delta_step': 100,
    'subsample': 0.4,
    'colsample_bytree': 0.4,
    'eval_metric': 'auc'}
dataset = pd.read_csv('KL234_unbalanced_11_3_20_imputed_hybrid10_4(noimage).csv') #no imaging 8 parameters
dataset.shape

###Dataset splitting and preprocessing ###
roc_auc_all_3=[]
dataset_num_col=dataset.shape[1]

#Organize Dataset & Generate Train/Test/Val split.
X = dataset.iloc[:, 2:dataset_num_col].values #obtan values all the way up until colum n113
y = dataset.iloc[:, 1].values

#Feature extraction
feature_names = dataset.head()
featurenames3 = list(feature_names)
featurenames4 = featurenames3[2:dataset_num_col] #this is the final list of features.

#Splitting Data
from sklearn.model_selection import train_test_split
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, y, test_size=0.2, random_state=trainval_test_rand_seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=train_val_rand_seed)

#Setting the data as featurenames4
D_train = xgb.DMatrix(X_train, label=Y_train, feature_names=featurenames4)
D_val = xgb.DMatrix(X_val, label=Y_val, feature_names=featurenames4)
D_test = xgb.DMatrix(X_test, label=Y_test, feature_names=featurenames4)

### MODEL TRAINING ###

#Other Hyperparameters.
num_round = 20
watchlist = [(D_val,'val'), (D_train,'train')]
#watchlist = [(Y_val,'val'), (Y_train,'train')]
steps = 30 # The number of training iterations

#Model Training
evals_result = {}
model = xgb.train(param, D_train, num_round, watchlist, evals_result=evals_result)

from xgboost import cv
xgb_cv = cv(dtrain=D_train, params=param, nfold=10,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
print(xgb_cv)
#xgb=XGBClassifier()
#XGBClassifier(param)
#model = xgb.fit(X_train, Y_train)

#model.save_model('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/1_30_opt1.model') # save model

### Model Evaluation & VISUALIZATION - ROC ###
#model.predict(D_test) #tell u the prediction values
prediction =  model.predict(D_test)
#prediction =  model.predict(X_test)
#np.savetxt('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/pred_1_30.csv', prediction, delimiter=",")

fpr3, tpr3, threshold3 = roc_curve(Y_test, prediction)
roc_auc3 = auc(fpr3, tpr3)
roc_auc_all_3.append(roc_auc3)

#Visualization - ROC
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
lw = 2
ax.plot(fpr1, tpr1, color='r',
         lw=lw, label='All Pred: 112 (area = %0.2f)' % roc_auc1, linestyle="-")
ax.plot(fpr2, tpr2, color='g',
         lw=lw, label='With Imaging Pred: 10 (area = %0.2f)' % roc_auc2, linestyle="-.")
ax.plot(fpr3, tpr3, color='b',
         lw=lw, label='No Imaging Pred: 7 (area = %0.2f)' % roc_auc3, linestyle="--")

ax.plot([0, 1.05], [0, 1.05], color='k', lw=lw, linestyle='-')

ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve', fontsize=14)
ax.legend(loc="lower right")

#Saving figures
plt.savefig('roc_all_2021-08-29.png',dpi=500)
from google.colab import files
files.download('roc_all_2021-08-29.png')

### VISUALIZATION - Feature Importance ###
plot_importance(model, max_num_features=20)

#Saving figures
pyplot.savefig('featureimportane no imaging.png',dpi=500)
pyplot.show()
from google.colab import files
files.download('featureimportane no imaging.png')

### VISUALIZATION - Confusion Matrix and other accuracy measures ###
from sklearn.metrics import confusion_matrix
model_threshold = 0.3
y_pred3 = model.predict(D_test)
y_pred3_th = model.predict(D_test)>model_threshold
y_true3=Y_test
#print(y_pred)
#print(y_true)

conf_matrix = confusion_matrix(y_true3, y_pred3_th)
# Printing the Confusion Matrix

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)

#Saving figures
from google.colab import files
plt.savefig('cm no imaging.png',dpi=500)
plt.show()
files.download('cm no imaging.png')

from sklearn.metrics import classification_report
target_names = ['KL Grade 0-1 (Clinical model no imaging)', 'KL Grade 2+ (Clinical model no imaging)']
print(classification_report(y_true3,y_pred3_th, target_names=target_names))

# compute and print accuracy score
from sklearn.metrics import accuracy_score
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_true3, y_pred3_th)))







### ground truth, ypred1, ypred2, ypred3
import pandas as pd
temp=np.array([y_true1, y_pred1, y_pred1_th, y_pred2, y_pred2_th, y_pred3, y_pred3_th])
#temp=temp.transpose
#print(temp)
#temp.shape
df=pd.DataFrame(temp)

df.to_csv('ytrue_ypred.csv')
files.download('ytrue_ypred.csv')

### De Long Test Module for ROC AUC comparison ###
### https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py ###

import pandas as pd
import numpy as np
import scipy.stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

### Perform De Long Test on above 3 models ###

x_distr = scipy.stats.norm(0.5, 1)
y_distr = scipy.stats.norm(-0.5, 1)
sample_size_x = 7
sample_size_y = 14
n_trials = 1000
aucs = numpy.empty(n_trials)
variances = numpy.empty(n_trials)
numpy.random.seed(1234235)
labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
for trial in range(n_trials):
    scores = numpy.concatenate([
        x_distr.rvs(sample_size_x),
        y_distr.rvs(sample_size_y)])
    aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
    auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
        labels, scores)

print(f"Experimental variance {variances.mean():.4f}, "
      f"computed vairance {aucs.var():.4f}, {n_trials} trials")

roc_auc_all_3
for trial in range(3):
  scores = numpy.concatenate([
    x_distr.rvs(sample_size_x),
    y_distr.rvs(sample_size_y)])
  aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
  auc_delong, variances[trial] = delong_roc_variance(labels, scores)

import scipy.stats

x_distr = scipy.stats.norm(0.5, 1)
y_distr = scipy.stats.norm(-0.5, 1)

x_distr









## DATA LOADER & PREPROCESSOR

#Saving original state.
trainval_test_rand_seed=45
train_val_rand_seed=23

roc_auc_all=[]

#Organize Dataset & Generate Train/Test/Val split.
for x in range (1,1):
    #trainval_test_rand_seed=x
    #train_val_rand_seed=x

#Data Extraction
    X = dataset.iloc[:, 2:113].values #obtan values all the way up until colum n113
    X_113 = dataset.iloc[:, 2:113].values #obtan values all the way up until colum n113
    X_10 = dataset.iloc[:, 2:12].values #obtan values all the way up until colum n10
    X_8 = dataset.iloc[:, 2:10].values #obtan values all the way up until colum 
    y = dataset.iloc[:, 1].values

#Feature extraction
    feature_names = dataset.head()
    featurenames3 = list(feature_names)
    featurenames4 = featurenames3[2:113] #this is the final list of features.

#Splitting Data
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, y, test_size=0.2, random_state=trainval_test_rand_seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=train_val_rand_seed)

#Setting the data as featurenames4
    D_train = xgb.DMatrix(X_train, label=Y_train, feature_names=featurenames4)
    D_val = xgb.DMatrix(X_val, label=Y_val, feature_names=featurenames4)

#D_val.feature_names
#featurenames4

## MODEL TRAINING & EVALUATION

# Additional hyperparameters.
    num_round = 20
    watchlist = [(D_val,'val'), (D_train,'train')]
    steps = 30 # The number of training iterations

# train our model 
    evals_result = {}
    model = xgb.train(param, D_train, num_round, watchlist, evals_result=evals_result)
    #model.save_model('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/1_30_opt1.model') # save model
    
#Model Evaluation
    D_test = xgb.DMatrix(X_test, label=Y_test, feature_names=featurenames4)
    #model.predict(D_test) #tell u the prediction values
    prediction =  model.predict(D_test)
#np.savetxt('/Users/gabby/Google Drive/Jae_ho_project/OA project/model_output/pred_1_30.csv', prediction, delimiter=",")

    fpr1, tpr1, threshold4 = roc_curve(Y_test, prediction)
    roc_auc1 = auc(fpr1, tpr1)
    #print(roc_auc1)
    roc_auc_all.append(roc_auc1)

#112 predictors
roc_auc_all_112 = roc_auc_all
from statistics import mean, median, pvariance, stdev, variance
#print(mean(roc_auc_all), median(roc_auc_all), stdev(roc_auc_all), variance(roc_auc_all))
print(stdev(roc_auc_all))

#10 predictors
roc_auc_all_10 = roc_auc_all
from statistics import mean, median, pvariance, stdev, variance
#print(mean(roc_auc_all), median(roc_auc_all), stdev(roc_auc_all), variance(roc_auc_all))
print(stdev(roc_auc_all))

#8 predictors without imaging
roc_auc_all_6 = roc_auc_all
from statistics import mean, median, pvariance, stdev, variance
#print(mean(roc_auc_all), median(roc_auc_all), stdev(roc_auc_all), variance(roc_auc_all))
print(stdev(roc_auc_all))

#NOT WORKING AS OF NOW

#from scipy import stats

#112 AUC: 0.7919536 +/- 0.04671451977190963
#10 AUC: 0.7727437 +/- 0.04255390733743138
#6 AUC: 0.66872055 +/- 0.04222372182236481

#print(stats.ttest_ind(roc_auc_all_112,roc_auc_all_10)) 
#print(stats.ttest_ind(roc_auc_all_10,roc_auc_all_6))
#print(stats.ttest_ind(roc_auc_all_112,roc_auc_all_6))

#### Hyperparameter Search Code ####

# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)

# grid search
model = XGBClassifier()
#n_estimators = [25,50,75,250]
#learning_rate=[0.2, 0.1, 0.05, 0.005]
#gamma=[0, 0.1,0.2,0.5]
#max_depth = [2,4,5,10,30,100]
#min_child_weight=[1,2,3,5,10,30]
n_estimators = [25,50]
learning_rate=[0.2, 0.1, 0.05]
gamma=[0.2, 0.1, 0]
max_depth = [30, 10, 5]
min_child_weight=[3, 2]

param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, gamma=gamma, 
                  learning_rate=learning_rate,min_child_weight=min_child_weight)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, label_encoded_y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))

param=grid_result.best_params_
param.update([('objective', 'binary:logistic'), ('eval_metric', 'auc')])
param



#ROC Curve Generation

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
lw = 2
ax.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='ROC curve for F1 (area = %0.2f)' % roc_auc1, linestyle="-.")

ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve', fontsize=14)





import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu
import scipy.stats

x_distr = scipy.stats.norm(0.5, 1)
y_distr = scipy.stats.norm(-0.5, 1)
sample_size_x = 7
sample_size_y = 14
n_trials = 1000
aucs = numpy.empty(n_trials)
variances = numpy.empty(n_trials)
numpy.random.seed(3141592)
labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
for trial in range(n_trials):
    scores = numpy.concatenate([
        x_distr.rvs(sample_size_x),
        y_distr.rvs(sample_size_y)])
    aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
    auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
        labels, scores)

print(f"Experimental variance {variances.mean():.4f}, "
      f"computed vairance {aucs.var():.4f}, {n_trials} trials")

import compare_auc_delong_xu
from compare_auc_delong_xu import compute_midrank
from compare_auc_delong_xu import calc_pvalue

prediction2 = prediction + 0.02

delong_roc_test(Y_test, prediction, prediction2)

prediction