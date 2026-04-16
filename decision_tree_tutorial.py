"""
Workshop Tutorial: Decision Trees with scikit-learn
=====================================================

A decision tree is a supervised learning algorithm that splits data into
branches based on feature values, forming a tree-like structure of decisions.

Each internal node:  a feature/threshold test
Each branch:         the outcome of the test
Each leaf:           a predicted class (classification) or value (regression)

We'll use the Iris dataset — 150 flower samples, 3 species, 4 features.
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------
# Step 1 — Load & explore the data
# ------------------------------------------------------------
iris = load_iris()

X = iris.data        # shape: (150, 4) — sepal/petal length and width
y = iris.target      # 0=setosa, 1=versicolor, 2=virginica

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}  →  {iris.feature_names}")
print(f"Classes:  {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}\n")

# ------------------------------------------------------------
# Step 2 — Split into train / test sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for evaluation
    random_state=42,  # reproducibility
    stratify=y,       # keep class proportions equal in both splits
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")

# ------------------------------------------------------------
# Step 3 — Train a decision tree
# ------------------------------------------------------------
# Key hyperparameters:
#   max_depth     — limits tree depth to prevent overfitting
#   criterion     — "gini" (impurity) or "entropy" (information gain)
#   min_samples_split — minimum samples needed to split a node

clf = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42,
)
clf.fit(X_train, y_train)

print("=== Trained Decision Tree (text) ===")
print(export_text(clf, feature_names=iris.feature_names))

# ------------------------------------------------------------
# Step 4 — Evaluate the model
# ------------------------------------------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"=== Evaluation ===")
print(f"Accuracy: {accuracy:.2%}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ------------------------------------------------------------
# Step 5 — Understand feature importance
# ------------------------------------------------------------
# Feature importance = how much each feature reduced impurity across all splits
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank}. {iris.feature_names[idx]:<30} {importances[idx]:.4f}")
print()

# ------------------------------------------------------------
# Step 6 — Visualise
# ------------------------------------------------------------

# 6a: The decision tree itself
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,        # colour nodes by majority class
    rounded=True,
    ax=ax,
)
ax.set_title("Decision Tree — Iris Dataset (max_depth=3)")
fig.tight_layout()
plt.show()

# 6b: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=iris.target_names,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.show()

# 6c: Feature importance bar chart
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    range(len(importances)),
    importances[sorted_idx],
    tick_label=[iris.feature_names[i] for i in sorted_idx],
)
ax.set_ylabel("Importance (Gini)")
ax.set_title("Feature Importances")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 7 — Overfitting demo: depth vs. accuracy
# ------------------------------------------------------------
print("=== Depth vs Accuracy (train / test) ===")
depths = range(1, 11)
train_scores, test_scores = [], []

for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, m.predict(X_train)))
    test_scores.append(accuracy_score(y_test, m.predict(X_test)))
    print(f"  depth={d:2d}  train={train_scores[-1]:.2%}  test={test_scores[-1]:.2%}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, train_scores, marker="o", label="Train accuracy")
ax.plot(depths, test_scores, marker="s", label="Test accuracy")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting: Train vs Test Accuracy by Tree Depth")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 8 — Make a single prediction (inference)
# ------------------------------------------------------------
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # a new, unseen flower
prediction = clf.predict(sample)[0]
probabilities = clf.predict_proba(sample)[0]

print("\n=== Single Prediction ===")
print(f"Input features: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Predicted class: {iris.target_names[prediction]}")
print("Class probabilities:")
for cls, prob in zip(iris.target_names, probabilities):
    print(f"  {cls:<12} {prob:.2%}")
