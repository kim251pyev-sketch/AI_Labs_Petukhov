import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

input_file = 'income_data.txt'
X, y = [], []
count_class1, count_class2 = 0, 0
max_datapoints = 5000 # Зменшено для швидкості навчання нелінійних ядер

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
X_encoded = np.empty(X.shape, dtype=object)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


classifier = SVC(kernel='sigmoid')
classifier.fit(X_train, y_train)


y_test_pred = classifier.predict(X_test)
print("--- Гаусове ядро (RBF) ---")
print(f"F1 score: {round(100 * f1_score(y_test, y_test_pred, average='weighted'), 2)}%")
print(f"Accuracy: {round(100 * accuracy_score(y_test, y_test_pred), 2)}%")
print(f"Precision: {round(100 * precision_score(y_test, y_test_pred, average='weighted'), 2)}%")
print(f"Recall: {round(100 * recall_score(y_test, y_test_pred, average='weighted'), 2)}%")
