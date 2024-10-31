# -*- codeing = utf-8 -*-
# @Time : 2024/10/31 12:05
# @Author : 何曼乔
# @File : 客户流失预测源代码.py
# @Software : PyCharm
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

df_train = pd.read_csv("D:\\0暨南大学\机器学习\实验\BankCustomerChurn.csv")

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
    ("RBF SVM (optimized)", SVC(random_state=42)),  # Optimized later using GridSearchCV
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Neural Net", MLPClassifier(random_state=42, max_iter=500)),  # Increased max_iter for better convergence
    ("AdaBoost (optimized)", AdaBoostClassifier(random_state=42)),  # Optimized later using GridSearchCV
    ("Naive Bayes (Gaussian)", GaussianNB()),
    ("Naive Bayes (Multinomial)", MultinomialNB()),  # Added for completeness, may not be relevant for digits dataset
    ("Naive Bayes (Bernoulli)", BernoulliNB()),  # Added for completeness, may not be relevant for digits dataset
    ("QDA", QuadraticDiscriminantAnalysis()),
    ("Logistic Regression", linear_model.LogisticRegression(random_state=42, max_iter=10000))
    # Increased max_iter for better convergence
]

# 设置参数网格以优化RBF SVM、AdaBoost、Decision Tree、Random Forest等模型
param_grids = {
    "RBF SVM (optimized)": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    },
    "AdaBoost (optimized)": {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 1, 1.5]
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    },
    "Logistic Regression": {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
}

# 优化各模型参数并替换到分类器列表
for name, clf in classifiers:
    if name in param_grids:
        grid_search = GridSearchCV(clf, param_grids[name], cv=10, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_clf = grid_search.best_estimator_
        classifiers = [(n, c) if n != name else (name, best_clf) for n, c in classifiers]

# 初始化DataFrame来存储结果
models_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Support"])

# 训练并评估每个分类器，使用交叉验证获取更准确的性能评估
for name, clf in classifiers:
    # 使用 cross_val_predict 获取每个折的预测结果
    y_pred = cross_val_predict(clf, x_train, y_train, cv=10)

    # 计算各种评估指标并生成分类报告
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')
    report = classification_report(y_train, y_pred, output_dict=True)

    # 获取所有类的总支持度
    total_support = sum([v['support'] for k, v in report.items() if isinstance(v, dict)])

    # 将结果存储到DataFrame中
    models_df = models_df.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Support": total_support
    }, ignore_index=True)

    # 打印每个模型的详细结果
    print(f"{name}:\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification Report:\n {pd.DataFrame(report).transpose()}\n")

# 输出汇总的DataFrame
print(models_df)