import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model

# 包含子图的matplotlib图的大小
plt.rc('figure', figsize=(10, 5))

# 包含子图的matplotlib图的大小
fizsize_with_subplots = (10, 10)

# matplotlib直方图箱的大小
bin_size = 10

df_train = pd.read_csv("./BankCustomerChurn.csv")

countrys = sorted(df_train['Geography'].unique()) #统计一下Geography有几种不同的值
countrys_mapping = dict(zip(countrys, range(0, len(countrys) + 1)))
sexes = sorted(df_train['Gender'].unique()) #统计一下sex有几种不同的值
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

df_train['Geography_Val'] = df_train['Geography'].map(countrys_mapping).astype(int)
df_train['Gender_Val'] = df_train['Gender'].map(genders_mapping).astype(int)
df_train.dtypes[df_train.dtypes.map(lambda x: x == 'object')]

# 如果列名正确，尝试删除
if 'Surname' in df_train.columns and 'Geography' in df_train.columns and 'Gender' in df_train.columns:df_train = df_train.drop(['Surname', 'Geography', 'Gender'], axis=1)

# Data with independent variables:

x = df_train.drop(['Exited'], axis= 1)

# Data from target variable:

y = df_train['Exited']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)
from sklearn.preprocessing import MinMaxScaler
specific_columns = ['CreditScore', 'Age', 'Balance', 'NumOfProducts','EstimatedSalary']

from sklearn.preprocessing import MinMaxScaler
specific_columns = ['CreditScore', 'Age', 'Balance', 'NumOfProducts','EstimatedSalary']
scaler = MinMaxScaler()
# Normalization of trainig data:

x_train[specific_columns] = scaler.fit_transform(x_train[specific_columns])

# Normalization of test data:

x_test[specific_columns] = scaler.transform(x_test[specific_columns])

import os
import pickle
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model

# 定义分类器及其名称
classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier()),
    ("Linear SVM", SVC(kernel="linear", random_state=42)),
    ("RBF SVM (optimized)", SVC(random_state=42)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Neural Net", MLPClassifier(random_state=42, max_iter=500)),
    ("AdaBoost (optimized)", AdaBoostClassifier(random_state=42)),
    ("Naive Bayes (Gaussian)", GaussianNB()),
    ("Naive Bayes (Multinomial)", MultinomialNB()),
    ("Naive Bayes (Bernoulli)", BernoulliNB()),
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Logistic Regression", linear_model.LogisticRegression(random_state=42, max_iter=10000))
]

# 参数网格
param_grids = {
    "RBF SVM (optimized)": {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
    "AdaBoost (optimized)": {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 1, 1.5]},
    "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    "Logistic Regression": {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
}

# 检查是否已有保存的classifiers文件
if os.path.exists("classifiers.pkl"):
    print("加载已保存的分类器...")
    with open("classifiers.pkl", "rb") as file:
        classifiers = pickle.load(file)
else:
    print("未检测到分类器文件，执行优化过程...")
    for name, clf in classifiers:
        if name in param_grids:
            grid_search = GridSearchCV(clf, param_grids[name], cv=10, scoring='accuracy', verbose=2, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            best_clf = grid_search.best_estimator_
            classifiers = [(n, c) if n != name else (name, best_clf) for n, c in classifiers]

    # 保存优化后的 classifiers
    with open("classifiers.pkl", "wb") as file:
        pickle.dump(classifiers, file)

# 输出最终的分类器列表
print(classifiers)
