import cuml
from cuml.neighbors import KNeighborsClassifier as cumlKNN
from cuml.svm import SVC as cumlSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np

# 替换部分模型为GPU加速版本
classifiers = [
    ("Nearest Neighbors (GPU)", cumlKNN()),
    ("Linear SVM (GPU)", cumlSVC(kernel="linear")),
    ("RBF SVM (GPU)", cumlSVC(kernel="rbf")),
    # 其他模型可以继续使用CPU版本
]

# 训练和评估每个分类器
models_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Support"])

for name, clf in classifiers:
    if "GPU" in name:  # 只在GPU模型上运行
        clf.fit(x_train.to_numpy(), y_train.to_numpy())  # cuml 需要 NumPy 格式的数据
        y_pred = clf.predict(x_test.to_numpy())
    else:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

    # 评估和记录性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    # 将结果存储到 DataFrame 中
    models_df = models_df.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Support": sum([v['support'] for k, v in report.items() if isinstance(v, dict)])
    }, ignore_index=True)

    print(f"{name}:\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification Report:\n {pd.DataFrame(report).transpose()}\n")

print(models_df)