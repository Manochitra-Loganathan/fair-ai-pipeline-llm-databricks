
# Databricks Notebook - Fairness Evaluation Pipeline using AIF360

# Install packages (only if not already installed)
# %pip install aif360 pandas scikit-learn

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
dataset_orig = AdultDataset()

# Split into train/test
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# Apply reweighing mitigation
RW = Reweighing(unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
dataset_transf_train = RW.fit_transform(dataset_orig_train)

# Train logistic regression on original and transformed
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()

X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Metric before mitigation
metric_test = ClassificationMetric(
    dataset_orig_test,
    dataset_orig_test.copy(deepcopy=True).set_labels(y_pred),
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

print("Disparate Impact (original):", metric_test.disparate_impact())
print("Equal Opportunity Difference (original):", metric_test.equal_opportunity_difference())
print("Average Odds Difference (original):", metric_test.average_odds_difference())

# Train with mitigated data
X_train_mit = dataset_transf_train.features
y_train_mit = dataset_transf_train.labels.ravel()

lr_mit = LogisticRegression(solver='liblinear')
lr_mit.fit(X_train_mit, y_train_mit)
y_pred_mit = lr_mit.predict(X_test)

# Metric after mitigation
metric_mit = ClassificationMetric(
    dataset_orig_test,
    dataset_orig_test.copy(deepcopy=True).set_labels(y_pred_mit),
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

print("Disparate Impact (mitigated):", metric_mit.disparate_impact())
print("Equal Opportunity Difference (mitigated):", metric_mit.equal_opportunity_difference())
print("Average Odds Difference (mitigated):", metric_mit.average_odds_difference())
