import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Оглядовий аналіз
print("Форма датасету:", dataset.shape)
print("\nПерші 20 рядків:\n", dataset.head(20))
print("\nСтатистичне зведення:\n", dataset.describe())
print("\nРозподіл за класами:\n", dataset.groupby('class').size())


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()
scatter_matrix(dataset)
pyplot.show()


array = dataset.values
X = array[:,0:4].astype(float)
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)


models = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=200)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n--- Оцінка якості моделі (SVM) ---")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("Звіт:\n", classification_report(Y_validation, predictions))


X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)
print(f"\nПрогноз для квітки {X_new[0]}: {prediction[0]}")