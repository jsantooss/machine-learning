# Telecom Churn Prediction: Machine Learning Showdown
Welcome to the Telecom Churn Machine Learning project.

This project tackles one of telecom's biggest challenges, predicting which customers are likely to leave (churn) using binary classification and unsupervised clustering. 

# 🔍 What’s Inside?
This project delivers:
✅ Binary classification with Logistic Regression, Random Forest, and KNN

✅ Unsupervised learning via K-Means & DBSCAN to reveal hidden customer segments

✅ Data visualization with seaborn (boxplots, swarm plots, PCA clusters)

✅ Hyperparameter tuning using GridSearchCV and SMOTE for class imbalance

✅ Full performance metrics: Accuracy, Precision, Recall, F1, and ROC-AUC

# 🎯 Key Findings: 

Best classifier: Logistic Regression (80% accuracy)

Critical features: tenure, MonthlyCharges, Contract

Clustering insight: K-Means segments helped target retention campaigns!

# 🛠 Tools & Libraries
python
# Core ML
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans, DBSCAN

# Data viz
import seaborn as sns  # boxplots, swarm plots, PCA visualizations

import matplotlib.pyplot as plt

# Data processing
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# 📂 Files & Resources
File	Description

TP_ML_V3.ipynb	Jupyter Notebook with full code and visualizations
Telecom-Dataset Data set file


🙏 Credits
ML Wizard: João Santos

Guidance: Prof. Simão Paredes
