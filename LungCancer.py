#%%
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import warnings
warnings.simplefilter("ignore")

# Loading up the data
data = pd.read_csv("../LUNG CANCER/survey lung cancer.csv")
print(data.head())
print(data.info())
print(data.describe().T)
print(data.isna().sum())
data["LUNG_CANCER"].unique()
data["GENDER"].unique()

# Mapping numeric values to non-numeric values
data['GENDER'] = data['GENDER'].map({'F': 0, 'M': 1})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO': 0, 'YES': 1})
print(data.dtypes)

def custom_palette(custom_colors):
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both', labelsize=0, length = 0)
pal = ["#395e66","#387d7a","#32936f","#26a96c","#2bc016"]
custom_palette(pal)

fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, fmt='.1g', cmap=pal, cbar=False, linewidths=0.5, linecolor='grey')

print ('Total Healthy Patients : {} '.format(data.LUNG_CANCER.value_counts()[0]))
print ('Total Suspected Patients : {} '.format(data.LUNG_CANCER.value_counts()[1]))

values = data['LUNG_CANCER'].value_counts().tolist()
names = list(dict(data['LUNG_CANCER'].value_counts()).keys())

px.pie(data, values=values, names=names, hole = 0.5, color_discrete_sequence=["firebrick", "green"])

x = data['LUNG_CANCER'].value_counts().index.tolist()
y = data['LUNG_CANCER'].value_counts().tolist()
print('x:',x)
print(y)
fig1 = px.bar(x=x, y=y, color=["firebrick", "green"], color_discrete_map="identity",
            labels={
                'x': 'LUNG_CANCER',
                'y': 'count'
                },)
fig1.show()

plt.style.use("seaborn")
data.hist(figsize=(25,20), color=pal[3], bins=15)

plt.figure(figsize=(20,10))
sns.boxenplot(data = data, palette = pal)
plt.xticks(rotation=90, fontsize=18)
plt.show()

sns.kdeplot(x=data["GENDER"], y=data["AGE"], hue =data["LUNG_CANCER"], palette="rocket")
plt.show()

for i in data:
    sns.swarmplot(x = data["LUNG_CANCER"], y = data[i], color = "black", alpha = 0.8)
    sns.boxenplot(x = data["LUNG_CANCER"], y = data[i], palette="crest")
    plt.show()   

#Splitting the data into training and test sets

X = data.drop("LUNG_CANCER", axis=1)
X.head()
y = data["LUNG_CANCER"]
y.head()

# Adding randomized samples to the data as the data is imbalanced

from imblearn.over_sampling import RandomOverSampler

over_samp =  RandomOverSampler(random_state=0)
X_train_res, y_train_res = over_samp.fit_resample(X, y)
X_train_res.shape, y_train_res.shape

plt.style.use("seaborn")
plt.figure(figsize=(10,6))
plt.title("No. of samples after balancing", fontsize=20, y=1.02)
sns.countplot(x = y_train_res, palette=pal)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size = 0.2, random_state = 42)

len(X_train), len(X_test)

# Scaling the data 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.figure(figsize=(20,10))
plt.title("Data after Scaling", fontsize=25, y=1.02)
sns.boxenplot(data = X_train, palette = pal)
plt.show()

# Linear regression 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

LinearRegressionScore = lr.score(X_test,y_test)
print("Accuracy obtained by Linear Regression model:",LinearRegressionScore*100)

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train,y_train)

RandomForestClassifierScore = rfc.score(X_test, y_test)
print("Accuracy obtained by Random Forest Classifier model:",RandomForestClassifierScore*100)

# Confusion Matrix of Random Forest Classifier
from sklearn.metrics import confusion_matrix, classification_report

y_pred_rfc = rfc.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cf_matrix, annot=True, cmap=pal)
plt.title("Confusion Matrix for Random Forest Classifier", fontsize=14, fontname="Helvetica", y=1.03);
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred_rfc))

# K neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

KNeighborsClassifierScore = knn.score(X_test, y_test)
print("Accuracy obtained by K Neighbors Classifier model:",KNeighborsClassifierScore*100)

# Confustion Matrix of K Neighbors Classifier
y_pred_knn = knn.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cf_matrix, annot=True, cmap=pal)
plt.title("Confusion Matrix for K Neighbors Classifier", fontsize=14, fontname="Helvetica", y=1.03)
print(metrics.classification_report(y_test, y_pred_knn))

# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

DecisionTreeClassifierScore = dtc.score(X_test,y_test)
print("Accuracy obtained by Decision Tree Classifier model:",DecisionTreeClassifierScore*100)

# Confustion Matrix of Decision Tree classifier
y_pred_dtc = dtc.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred_dtc)
sns.heatmap(cf_matrix, annot=True, cmap=pal)
plt.title("Confusion Metrix for Decision Tree Classifier", fontsize=14, fontname="Helvetica", y=1.03)
print(metrics.classification_report(y_test, y_pred_dtc))

# Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

GradientBoostingClassifierScore = gb.score(X_test,y_test)
print("Accuracy obtained by Gradient Boosting Classifier model:",GradientBoostingClassifierScore*100)

# Confustion Matrix of  Gradient boosting classifier
y_pred_gb = gb.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cf_matrix, annot=True, cmap=pal)
plt.title("Confusion Matrix for Gradient Boosting Classifier", fontsize=14, fontname="Helvetica", y=1.03)
print(metrics.classification_report(y_test, y_pred_gb))

#XGB classifier

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

XGBClassifierScore = xgb.score(X_test,y_test)
print("Accuracy obtained by XGB Classifier model:",XGBClassifierScore*100)

# Confustion Matrix of XGB Classifier

y_pred_xgb = xgb.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cf_matrix, annot=True, cmap=pal)
plt.title("Confusion Matrix for XGB Classifier", fontsize=14, fontname="Helvetica", y=1.03)
print(metrics.classification_report(y_test, y_pred_xgb))

# Model comparison
#%%
plt.style.use("seaborn")

x = ["Linear Regression", 
     "Decision Tree Classifier", 
     "Random Forest Classifier", 
     "K Neighbors Classifier",  
     "Gradient Boosting Classifier",
     "XGB Classifier"]

y = [LinearRegressionScore, 
     DecisionTreeClassifierScore, 
     RandomForestClassifierScore, 
     KNeighborsClassifierScore,  
     GradientBoostingClassifierScore, 
     XGBClassifierScore]

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=x,y=y, palette=pal)
plt.ylabel("Model Accuracy",fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(rotation=40, fontsize=14)
plt.title("Model Comparison - Model Accuracy", fontsize=20, fontname="Helvetica", y=1.03)
plt.show()


# %%
