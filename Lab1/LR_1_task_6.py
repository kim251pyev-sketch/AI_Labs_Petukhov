import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# 1. Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 2. Розбивка на тренувальні та тестові дані (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# 3. Створення та навчання SVM класифікатора
# kernel='linear' будує лінійну межу, як і Байєс
classifier_svm = svm.SVC(kernel='linear')
classifier_svm.fit(X_train, y_train)

# 4. Прогнозування та оцінка
y_test_pred = classifier_svm.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]

print("Accuracy of SVM classifier =", round(accuracy, 2), "%")

# 5. Візуалізація
visualize_classifier(classifier_svm, X_test, y_test)

# 6. Порівняльна оцінка (cross_val_score)
num_folds = 3
accuracy_values = cross_val_score(classifier_svm, X, y, scoring='accuracy', cv=num_folds)
print("Mean Accuracy (SVM) via Cross-Val:", round(100 * accuracy_values.mean(), 2), "%")