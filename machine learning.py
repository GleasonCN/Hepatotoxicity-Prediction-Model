import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt

# Data Import
df = pd.read_excel('.xlsx')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Partition data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialization model
models = {
    'RF': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'GB': GradientBoostingClassifier(n_estimators=100),
    'LR': LogisticRegression(max_iter=1000)
}

# K-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Store results
results = {}
confusion_matrices = {}
roc_data = {}

# Train the model and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if name != 'SVM' else model.decision_function(X_test)

    # Store the confusion matrix and calculate the evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = precision_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_test, y_proba[:, 1]) if name != 'SVM' else roc_auc_score(y_test, y_proba),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'MCC': mcc
    }

    # Store ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1]) if name != 'SVM' else roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr)

# Visual ROC curve
fig, ax = plt.subplots()
for name, (fpr, tpr) in roc_data.items():
    ax.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["AUC"]:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
plt.show()

# The visual confusion matrix is a bar graph
def plot_confusion_matrix_bar(cm, ax, name, title='Confusion Matrix'):
    bar_width = 0.35
    index = np.arange(len(cm))

    bar1 = ax.bar(index, cm[:, 0], bar_width, label='Predicted 0')
    bar2 = ax.bar(index + bar_width, cm[:, 1], bar_width, label='Predicted 1')

    ax.set_xlabel('True label')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Actual 0', 'Actual 1'])
    ax.legend()

    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for ax, (name, cm) in zip(axs.flatten(), confusion_matrices.items()):
    plot_confusion_matrix_bar(cm, ax, name, title=name)
plt.tight_layout()
plt.show()

# Output evaluation metrics to EXCEL
results_df = pd.DataFrame(results).T
results_df.to_excel('.xlsx')
