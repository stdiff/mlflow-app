# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Credit Card Fraud Detection
#
# The aim of this analysis is to construct a predictive model which can give an actionable result.
#
# ## Environment of Analysis

# %load_ext watermark
# %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn

# +
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# +
from IPython.display import display, HTML
plt.style.use("fivethirtyeight")

from pylab import rcParams
rcParams['figure.figsize'] = 14, 6

pd.set_option('display.max_rows', None)

# +
from configparser import ConfigParser
import mlflow

config = ConfigParser()
config.read("../config.ini")

client = mlflow.tracking.MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
# -

import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
from lib import enrichment
from lib import modeling

import warnings
warnings.filterwarnings("ignore")

# ## Data
#
# The data set to analyze is provided by "[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)". We have a randomly resampled subset (60%). The data set has four kinds of fields.
#
# - `Time` : Number of seconds elapsed between this transaction and the first transaction in the dataset
# - `Amount` : Transaction amount
# - `V1`&ndash;`V28` : PCA components
# - `Class` : 1 for fraudulent transactions, 0 otherwise
#
# Each row corresponds to one transaction and has a label: fraudulent or not. Note that there is no identifier for the card holders.
#
# The explanation of `Time` sounds at least literally wrong. If was is true, then the number of 0 in `Time` would agree with the number of the users. But there are only two rows with `Time = 0`.

run_uuid, artifact_uri = enrichment.look_up_run_load(client, retrieval_time="1547417275",
                                                     table="transaction", dataset="train")
df= enrichment.get_artifact(client, run_uuid, artifact_uri, artifact_path="data")
df.head()

# As the following histogram shows, all transactions happen within two days.

(df.Time/(60*60)).hist(bins=100)
plt.plot([48,48], [0,3000], "k:", linewidth=2)
plt.title("Number of transactions")
plt.xlabel("Hour");

# As we described above, the dateset that we are currently looking at is a subset of the original one.

df.shape

# Fraudulent transactions are rare events. Namely the proportion of the fraudulent transactions is very small. 

df.Class.mean() ## the proportion of the fraudulent transactions in the dataset

# The following scatter plot uses the first two principal components. The color corresponds to the label. The size of a point describes the amount of a transaction, but it is difficult to find any useful information about the class and the amount, by looking at the scatter plot.

sns.scatterplot(data=df, x="V1", y="V2", size="Amount", hue="Class", alpha=0.5)
plt.xlim(-35,3)
plt.ylim(-50,21);

# Here is another scatter plot. This shows that fraudulent transactions have some patten in the dataset. 

sns.scatterplot(data=df, x="V16", y="V14", hue="Class", alpha=0.5);

# The following diagrams are (estimated) density of each variable by the label. As we see above, the fraudulent transactions are rare, therefore it is difficult to see the difference between the labels, by looking at histograms. 

# +
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_pivot = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)\
             .melt(id_vars="Class")

sns.FacetGrid(data=df_pivot, col="variable", hue="Class", col_wrap=2, height=2, aspect=3, 
              xlim=(0,1), ylim=(0,50))\
   .map(sns.kdeplot, "value", shade=True).add_legend();
# -

# The following heatmap shows the proportions of the fraudulent transactions in buckets of variables. We create 25 bins according to the quantile and compute the proportion of the faudulent transactions. 

# +
from sklearn.preprocessing import KBinsDiscretizer

features = list(df.columns)[:-1]

k_bins = KBinsDiscretizer(n_bins=25, encode="ordinal", strategy="quantile")
df_bins = pd.DataFrame(k_bins.fit_transform(df.drop("Class",axis=1)),
                       columns=features, index=df.index)
df_bins["Class"] = df["Class"]

df_pos_rate = pd.concat([df_bins.groupby(col)["Class"].mean().rename(col) for col in features], axis=1)
sns.heatmap(df_pos_rate, center=df["Class"].mean(), cmap="RdBu_r");
# -

# ## Modelling
#
# ### Performance metric
#
# Before we train any mathematical model, we have to decide which metric we use to measure the performance of the metric. 
#
# The reason for detecting fraudulent transactions is usually to take some measures against fraudulent transactions. For example you need to make a phone call to check whether the suspected transaction was done by the card holder.
#
# The trivial predictive model saying "every transaction is not fraudulent" can achieve 99.8% accuracy, but you can not take any measure with the model, because you detect no fraudulent transactions.

1-df.Class.mean()

# On the other hand if we have a following predictive model 
#
# - If `V14 < -5`, then the transaction is probably fraudulent. (cf. the second scatter plot)
#
# then we obtain the following confusion matrix. 

pd.crosstab((df["V14"] < -5).map({True:1, False:0}).rename("prediction"), df["Class"])

# Therefore
#
# - 325 (=128+197) transactions are suspect. You will take a measure for them. 
# - 60% (197 in 325) of the suspect transactions are fraudulent. (Precision)
# - 65% (197 in 104+197) of the fraudulent transactions are detected. (Recall)
#
# So we want a predictive model which achieves high precision and high recall.
#
# It is easy to have a predictive model which achieve either high precision or high recall.
#
# - The predictive model saying "every transaction is fraudulent" achieves perfect recall, but the precision is zero.
# - The predictive model saying "most of the transactions are not fraudulent" achieves very high precision, but the recall is very small. (See below.)
#
# High precision and high recall are trade-off, namely it is difficult to achieve both high precision and high recall. 

pd.crosstab((df["V14"] < -15).map({True:1, False:0}).rename("prediction"), df["Class"])

# Moreover the number of suspect transactions is also important because it is directly relevant to the budget for taking a measure. (However we ignore this problem.) Therefore 
#
# - Train a predictive model in a usual way and use a predicted probability as a score.
# - Our prediction is: if the score is larger than a threshold, then the transaction is suspect.
# - Observe the performance metrics (precision, recall, etc.) by varying the threshold. (Usually if threshold is small, then the recall is high.)
#
# Let us look at an example. In the following code `y_score` are our predicted scores.

# +
y_score = np.round((1 - (df["V14"]-df["V14"].min())/(df["V14"].max() - df["V14"].min())),2)
y_score.rename("y_score", inplace=True)
y_true = df.Class.rename("y_true")

min_ind_pos = list(y_true).index(1)
pd.concat([y_score,y_true], axis=1)[(min_ind_pos-5):(min_ind_pos+5)]
# -

# Choosing a threshold we make a prediction. If we take 0.5 as a threshold, then we obtain the following confusion matrix.

pd.crosstab((y_score >= 0.5).map({True:1, False:0}).rename("prediction"), y_true)

# The following line charts show precision, recall and F1 score by threshold. As you see you can easily find the precision-recall trade-off.

roc_example = modeling.ROCCurve(y_true, y_score, pos_label=1)
roc_example.show_metrics()

# Here F1 score is a performance metric combining recall and precision.
#
# $$F_1 = \dfrac{2PR}{P+R}$$
#
# Here $P$ is the precision and $R$ is the recall. The following heat map shows the F1 scores.

# +
from itertools import product

seq = [round(0.1*(x+1),2) for x in range(10)]
df_f1 = pd.DataFrame([(p,r,(2*p*r)/(p+r)) for p,r in product(seq,seq)],
                     columns=["precision","recall","f1_score"])\
          .pivot(index="precision",columns="recall",values="f1_score")\
          .sort_index(ascending=False)
sns.heatmap(df_f1, annot=True, cmap="YlGnBu")
plt.title("F1 score");
# -

# While the performance metrics are important, we do not have a unified metric in order to compare two trained models yet. There are two options.
#
# 1. Use F1-score at the threshold = 0.5 as a unified performance metric.
# 2. Use the area under the ROC curve (AUC) as a unified performance metric.
#
# The ROC curve is the curve of (False-Positive rate, True-Positive rate) for thresholds. Here note that the true-positive rate is the same as recall. The ROC curve of the previous `y_score` looks like following.

roc_example.show_roc_curve()

# The AUC is the area under this curve as the name describes. If the scores are good, then the curve reaches top-left (FP=0, TP=1). Therefore better scores imply a larger area under the curve.
#
# NB. It is sometimes said that the precision-recall curve is better than ROC. (But we stick with the ROC curve.)
#
# We use the AUC as a unified metric, that is we choose a model (or hyperparameters) by looking at the AUC.
#
# ### Training 
#
# First we split the data into a training set and a validation set.

# +
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.drop("Class",axis=1), df["Class"],
                                                  test_size=0.4, random_state=3)
print("X_train.shape:", X_train.shape)
print("X_val.shape  : ", X_val.shape)
# -

# Then we train a logistic regression model.

# +
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

param = {"plr__C":[0.1,1,10], 
         "plr__penalty":["l1","l2"], 
         "plr__class_weight": [{0:1, 1:w} for w in [1,10,50,100,200]]}
pipeline = Pipeline([("scaler", MinMaxScaler()), ("plr", LogisticRegression(random_state=3))])
model = GridSearchCV(pipeline, param, cv=3, scoring="roc_auc")
# -

model.fit(X_train, y_train)
modeling.cv_results_summary(model)

# The following series shows the coefficients of the normalized variables. Note that we apply `MinMaxScaler` to the variables. The reason of the many zero coefficients is the L1-metric. 

modeling.show_coefficients(model.best_estimator_.steps[1][1], columns=X_train.columns)

# Let us evaluate the trained model with the validation set. The following diagram shows the ROC curve and the area under the curve (AUC).

# +
def compute_score(model, X:pd.DataFrame, y:pd.Series) -> pd.Series:
    col = list(model.classes_).index(1)
    return pd.Series(model.predict_proba(X)[:,col], name="score", index=y.index)

y_score_cw = compute_score(model, X_val, y_val)
roc_cw = modeling.ROCCurve(y_val, y_score_cw, pos_label=1)
roc_cw.show_roc_curve()
# -

# The following diegram shows the performance metrics by theshold. 

roc_cw.show_metrics()

# The threshold 0.0705428034068 achieves the best F1 score.

best_f1 = roc_cw.scores.sort_values(by="f1_score", ascending=False).iloc[0,:]
best_f1

# The following is the confusion matrix of the best threshold.

roc_cw.get_confusion_matrix(best_f1.name)

# If we apply the trained model, then
#
# - 0.18% of all transactions are suspect. You will take a measure for them. 
# - 78.4% of the suspect transactions are fraudulent. (Precision)
# - 75.3% of the fraudulent transactions are detected. (Recall)
