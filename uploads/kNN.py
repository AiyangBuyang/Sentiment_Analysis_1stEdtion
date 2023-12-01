import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris_data = pd.read_csv('fisheriris.csv')

# 2. 提取特征和标签
X = iris_data.iloc[:, :-1]  # 所有特征列
y = iris_data.iloc[:, -1]   # 标签列

# 3. 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 选择不同的k值，训练并评估模型
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 可以根据需要选择不同的k值
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 5. 输出不同k值下的分类精度
for k, accuracy in zip(k_values, accuracies):
    print(f'k = {k}, Accuracy = {accuracy}')
