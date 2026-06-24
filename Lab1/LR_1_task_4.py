import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
visualize_classifier(classifier, X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

accuracy_new = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy_new, 2), "%")
visualize_classifier(classifier_new, X_test, y_test)

num_folds = 3
scoring_types = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

print("\n--- Результати перехресної перевірки ---")
for score in scoring_types:
    values = cross_val_score(classifier, X, y, scoring=score, cv=num_folds)
    print(f"{score.capitalize()}: {round(100 * values.mean(), 2)}%")