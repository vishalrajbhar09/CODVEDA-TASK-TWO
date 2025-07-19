#!/usr/bin/env python
# coding: utf-8

# # NAME :- VISHAL RAMKUMAR RAJBHAR
# ID :- CV/A1/18203
# DOMAIN :- Data Science Intern

# Task 2: Classification with Logistic
# Regression
# • Description: Build a decision tree classifier to predict
# a categorical outcome (e.g., predict species of
# flowers).

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier


# In[2]:


# Step 2: Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Binarize for ROC AUC (multiclass)
y_bin = label_binarize(y, classes=[0, 1, 2])

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
    X, y, y_bin, test_size=0.3, random_state=42)


# In[3]:


# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Logistic Regression
log_reg = OneVsRestClassifier(LogisticRegression())
log_reg.fit(X_train_scaled, y_bin_train)
y_pred_log = log_reg.predict(X_test_scaled)


# In[4]:


# Step 6: Evaluation
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_bin_test, y_pred_log))
print("Precision:", precision_score(y_bin_test, y_pred_log, average='macro'))
print("Recall:", recall_score(y_bin_test, y_pred_log, average='macro'))
print("ROC AUC:", roc_auc_score(y_bin_test, y_pred_log, average='macro'))


# In[5]:


# Step 7: Random Forest Classifier
rf = OneVsRestClassifier(RandomForestClassifier(random_state=42))
rf.fit(X_train_scaled, y_bin_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_bin_test, y_pred_rf))
print("Precision:", precision_score(y_bin_test, y_pred_rf, average='macro'))
print("Recall:", recall_score(y_bin_test, y_pred_rf, average='macro'))
print("ROC AUC:", roc_auc_score(y_bin_test, y_pred_rf, average='macro'))


# In[6]:


# Step 8: Support Vector Machine
svm = OneVsRestClassifier(SVC(probability=True))
svm.fit(X_train_scaled, y_bin_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\n=== Support Vector Machine ===")
print("Accuracy:", accuracy_score(y_bin_test, y_pred_svm))
print("Precision:", precision_score(y_bin_test, y_pred_svm, average='macro'))
print("Recall:", recall_score(y_bin_test, y_pred_svm, average='macro'))
print("ROC AUC:", roc_auc_score(y_bin_test, y_pred_svm, average='macro'))


# In[10]:


# Step 9: ROC Curve for Logistic Regression
y_score = log_reg.predict_proba(X_test_scaled)
fpr = dict()
tpr = dict()

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_bin_test[:, i], y_score[:, i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i}')
    
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

