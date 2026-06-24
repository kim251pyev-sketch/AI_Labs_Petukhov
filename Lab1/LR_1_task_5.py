import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    return find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)

def petukhov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def petukhov_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

def petukhov_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def petukhov_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def petukhov_f1_score(y_true, y_pred):
    recall = petukhov_recall_score(y_true, y_pred)
    precision = petukhov_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

assert np.array_equal(petukhov_confusion_matrix(df.actual_label.values, df.predicted_RF.values), confusion_matrix(df.actual_label.values, df.predicted_RF.values))
assert petukhov_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values)

fpr_RF, tpr_RF, _ = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label.values, df.model_LR.values)
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
# Виведення результатів для обох моделей при порозі 0.5
print("--- Метрики для моделі Random Forest (RF) ---")
print(f"Accuracy: {petukhov_accuracy_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"Recall:   {petukhov_recall_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"Precision:{petukhov_precision_score(df.actual_label.values, df.predicted_RF.values):.3f}")
print(f"F1 Score: {petukhov_f1_score(df.actual_label.values, df.predicted_RF.values):.3f}")

print("\n--- Метрики для моделі Логістичної регресії (LR) ---")
print(f"Accuracy: {petukhov_accuracy_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"Recall:   {petukhov_recall_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"Precision:{petukhov_precision_score(df.actual_label.values, df.predicted_LR.values):.3f}")
print(f"F1 Score: {petukhov_f1_score(df.actual_label.values, df.predicted_LR.values):.3f}")

# Виведення AUC
print(f"\nAUC RF: {auc_RF:.3f}")
print(f"AUC LR: {auc_LR:.3f}")