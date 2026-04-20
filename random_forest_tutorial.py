"""
Random Forests with scikit-learn — Step-by-Step Tutorial
=========================================================

## Introduction to Random Forests

A **random forest** is an ensemble learning algorithm that combines many decision
trees to produce a more accurate and stable model than any single tree alone.
It belongs to the broader family of **bagging** (bootstrap aggregating) methods.

### How it works

Training proceeds in two stages:

1. **Bootstrap sampling** — each tree is trained on a different random sample of
   the training data, drawn *with replacement*. On average, ~63% of the training
   samples appear in each bootstrap sample; the rest are "out-of-bag" (OOB).

2. **Random feature subsets** — at every split, the algorithm considers only a
   random subset of features (typically √p for classification, where p is the
   total number of features). This decorrelates the trees, so their errors do
   not all point in the same direction.

Prediction is made by **majority vote** (classification) or **averaging**
(regression) across all trees.

### Why ensembles help

A single decision tree has high variance — small changes in the training data
can produce a very different tree. By averaging many diverse trees, the forest
reduces variance while keeping bias low. This is known as the **bias–variance
tradeoff**:

    Ensemble error ≈ avg_tree_bias² + avg_tree_variance / n_trees
                    + pairwise_correlation_term

Increasing `n_estimators` drives the variance term toward zero. Reducing
feature subset size (lower `max_features`) reduces the correlation term.

### Key hyperparameters

| Parameter          | Role                                                       |
|--------------------|------------------------------------------------------------|
| `n_estimators`     | Number of trees — more is better up to diminishing returns |
| `max_features`     | Features considered per split — controls tree diversity    |
| `max_depth`        | Per-tree depth cap — prevents individual tree overfitting  |
| `min_samples_leaf` | Minimum samples in a leaf — smooths noisy splits           |
| `oob_score`        | Use OOB samples as a free validation estimate              |

### Random forest vs. single decision tree

| Property              | Decision Tree      | Random Forest              |
|-----------------------|--------------------|----------------------------|
| Variance              | High               | Low (by design)            |
| Interpretability      | High (visual)      | Lower (black-box ensemble) |
| Overfitting tendency  | High               | Much lower                 |
| Training time         | Fast               | Slower (× n_estimators)    |
| Feature importance    | One tree's view    | Averaged across all trees  |

---

## Preparation — Imports

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — built-in datasets (Breast Cancer Wisconsin)
- **sklearn.model_selection** — splitting data into train/test sets
- **sklearn.ensemble** — the RandomForestClassifier
- **sklearn.tree** — single DecisionTreeClassifier used for comparison
- **sklearn.metrics** — tools to evaluate model performance

---

## Step 1 — Load & Explore the Data

Load the Breast Cancer Wisconsin dataset. Assign the feature matrix to `X` and
the target vector to `y`. Print the number of samples, feature names, class
names, and class distribution.

## Step 2 — Split into Train / Test Sets

Split the data 80/20. Ensure the split is reproducible and class proportions are
preserved in both sets.

## Step 3 — Train a Random Forest

Train a `RandomForestClassifier` with 100 trees, a max depth of 3, and OOB
scoring enabled. Use `random_state=42`. Print the OOB score alongside the
training accuracy.

## Step 4 — Evaluate the Model

Generate predictions on the test set. Print the overall accuracy and a full
classification report. Also print the OOB accuracy as a comparison point.

## Step 5 — Understand Feature Importance

Extract and print the feature importances averaged across all trees, ranked from
most to least important. In 2–3 sentences, explain why a random forest's feature
importance estimates are more reliable than those from a single tree.

## Step 6 — Visualise

Display the following three plots:
  6a — One tree from the forest (tree index 0), labelled with feature/class names.
  6b — A confusion matrix heatmap for the test set.
  6c — A feature importance bar chart with error bars showing variance across trees.

Do not save figures to disk — display them inline.

## Step 7 — n_estimators: More Trees vs. Accuracy

Train forests with n_estimators ∈ {1, 5, 10, 25, 50, 100, 200}. Record OOB
accuracy and test accuracy for each. Plot both curves. In 2–3 sentences, describe
when adding more trees stops being worth the extra computation time.

## Step 8 — Random Forest vs. Single Decision Tree

Train a `DecisionTreeClassifier` (max_depth=3, random_state=42) on the same
breast cancer split and compare its test accuracy against the random forest.
Print both accuracies and the gap. In 2–3 sentences, explain why the forest
outperforms the single tree on this dataset.

## Step 9 — Make a Single Prediction (Inference)

Using the trained random forest, predict the class and class probabilities for a
new patient sample with these measurements (first 10 features shown):
mean_radius=17.99, mean_texture=10.38, mean_perimeter=122.8, mean_area=1001.0,
mean_smoothness=0.1184 (plus 25 more features set to their dataset means).
Print both the predicted class label and the per-class probability distribution.

---
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# =============================================================================
# Step 1 — Load & Explore the Data
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# bc = load_breast_cancer()
#
# X = bc.data    # shape: (569, 30) — cell nucleus measurements
# y = bc.target  # 0=malignant, 1=benign
#
# print("=== Dataset Overview ===")
# print(f"Samples:  {X.shape[0]}")
# print(f"Features: {X.shape[1]}")
# print(f"Feature names: {bc.feature_names.tolist()}")
# print(f"Classes:  {bc.target_names}")
# print(f"Class distribution: {np.bincount(y)}  (0=malignant, 1=benign)\n")
#
# The Breast Cancer Wisconsin dataset contains 569 patient samples described by
# 30 numerical measurements of cell nuclei (radius, texture, perimeter, area,
# smoothness, etc.), each computed as the mean, standard error, and worst value
# across cells in the biopsy image. The task is binary classification: malignant
# (212 samples) vs. benign (357 samples). The dataset is mildly imbalanced —
# about 37% malignant, 63% benign — which is worth keeping in mind when
# interpreting accuracy alone.
# </details>

bc = load_breast_cancer()
X = bc.data
y = bc.target

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {bc.feature_names.tolist()}")
print(f"Classes:  {bc.target_names}")
print(f"Class distribution: {np.bincount(y)}  (0=malignant, 1=benign)\n")

# =============================================================================
# Step 2 — Split into Train / Test Sets
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,    # 20% held out for evaluation
#     random_state=42,  # reproducibility
#     stratify=y,       # preserve the ~37/63 malignant/benign ratio in both sets
# )
#
# print(f"Training samples: {len(X_train)}")
# print(f"Test samples:     {len(X_test)}\n")
#
# - test_size=0.2  → ~114 test / ~455 train
# - stratify=y     → malignant/benign proportions are the same in both splits,
#                    preventing an accidentally all-benign test set
# </details>

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")

# =============================================================================
# Step 3 — Train a Random Forest
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# rf = RandomForestClassifier(
#     n_estimators=100,   # 100 trees in the forest
#     max_depth=3,        # cap per-tree depth
#     oob_score=True,     # evaluate on out-of-bag samples for free
#     random_state=42,
# )
# rf.fit(X_train, y_train)
#
# train_acc = accuracy_score(y_train, rf.predict(X_train))
# print("=== Training ===")
# print(f"Training accuracy:     {train_acc:.2%}")
# print(f"OOB accuracy estimate: {rf.oob_score_:.2%}\n")
#
# Key hyperparameters:
#
#   n_estimators — Number of trees. More trees → lower variance, higher cost.
#                  100 is a solid default; gains plateau well before 500.
#
#   max_depth    — Limits depth of each individual tree. Without a cap the trees
#                  will overfit their bootstrap samples, but the ensemble still
#                  generalises well. A cap speeds training and reduces memory.
#
#   oob_score    — Each tree sees ~63% of the training data; the remaining ~37%
#                  (out-of-bag samples) act as a built-in validation set for that
#                  tree. Averaging OOB predictions across all trees gives a nearly
#                  free accuracy estimate without a separate validation split.
# </details>

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    oob_score=True,
    random_state=42,
)
rf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train))
print("=== Training ===")
print(f"Training accuracy:     {train_acc:.2%}")
print(f"OOB accuracy estimate: {rf.oob_score_:.2%}\n")

# =============================================================================
# Step 4 — Evaluate the Model
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# y_pred = rf.predict(X_test)
#
# test_acc = accuracy_score(y_test, y_pred)
# print("=== Evaluation ===")
# print(f"Test accuracy:         {test_acc:.2%}")
# print(f"OOB accuracy estimate: {rf.oob_score_:.2%}\n")
# print("Classification report:")
# print(classification_report(y_test, y_pred, target_names=bc.target_names))
#
# The OOB score is computed entirely from training data and usually tracks the
# true test accuracy closely — it is a cheap alternative to cross-validation when
# data is scarce. A large gap between OOB and test accuracy can signal data leakage
# or a distribution shift between splits.
#
# For medical classification, pay close attention to recall on the malignant class:
# a false negative (predicting benign when malignant) is far more costly than a
# false positive.
# </details>

y_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("=== Evaluation ===")
print(f"Test accuracy:         {test_acc:.2%}")
print(f"OOB accuracy estimate: {rf.oob_score_:.2%}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=bc.target_names))

# =============================================================================
# Step 5 — Understand Feature Importance
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# importances = rf.feature_importances_
# std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
# sorted_idx = np.argsort(importances)[::-1]
#
# print("=== Feature Importances ===")
# for rank, idx in enumerate(sorted_idx, 1):
#     print(f"  {rank:2}. {bc.feature_names[idx]:<35} importance={importances[idx]:.4f}  std={std[idx]:.4f}")
# print()
#
# Feature importance in a random forest is the mean decrease in Gini impurity
# across all splits that use a feature, averaged over all trees. Because it is
# averaged across hundreds of trees trained on different bootstrap samples, the
# estimate is far more stable than that from a single tree — one unusual split
# in one tree is diluted by the rest of the ensemble. The standard deviation
# across trees (computed above) gives a sense of how much the importance varies,
# exposing features that are important but inconsistently used.
#
# For breast cancer, "worst" features (worst radius, worst perimeter, worst area)
# typically dominate, as extreme measurements are more diagnostic than averages.
# </details>

importances = rf.feature_importances_
std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank:2}. {bc.feature_names[idx]:<35} importance={importances[idx]:.4f}  std={std[idx]:.4f}")
print()

# =============================================================================
# Step 6 — Visualise
# =============================================================================
# <details>
# <summary>Solution</summary>

# --- 6a: One tree from the forest ---
# Picking a single tree lets us inspect what one member of the ensemble looks
# like. With max_depth=3 and 30 features the tree is still readable, but note
# it only uses a small subset of features — the forest's power comes from
# combining many such specialised trees.

fig, ax = plt.subplots(figsize=(18, 6))
plot_tree(
    rf.estimators_[0],
    feature_names=bc.feature_names,
    class_names=bc.target_names,
    filled=True,
    rounded=True,
    ax=ax,
)
ax.set_title("One Tree from the Random Forest (estimators_[0], max_depth=3)")
fig.tight_layout()
plt.show()

# --- 6b: Confusion matrix ---
# For cancer classification pay attention to the top-right cell: false negatives
# (malignant predicted as benign). These are the most clinically costly errors.

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=bc.target_names,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Confusion Matrix — Random Forest")
fig.tight_layout()
plt.show()

# --- 6c: Feature importance bar chart with error bars ---
# With 30 features the chart is busy — show only the top 15 for readability.
# Error bars (± 1 std across trees) reveal which importances are stable vs. noisy.

top_n = 15
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    range(top_n),
    importances[sorted_idx[:top_n]][::-1],
    xerr=std[sorted_idx[:top_n]][::-1],
    capsize=4,
)
ax.set_yticks(range(top_n))
ax.set_yticklabels([bc.feature_names[i] for i in sorted_idx[:top_n]][::-1])
ax.set_xlabel("Importance (mean decrease in Gini)")
ax.set_title(f"Top {top_n} Feature Importances — Random Forest (± 1 std)")
fig.tight_layout()
plt.show()

# </details>

# =============================================================================
# Step 7 — n_estimators: More Trees vs. Accuracy
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# n_values = [1, 5, 10, 25, 50, 100, 200]
# oob_scores, test_scores = [], []
#
# print("=== n_estimators vs Accuracy ===")
# for n in n_values:
#     use_oob = n >= 10  # avoid UserWarning for very small forests
#     m = RandomForestClassifier(n_estimators=n, max_depth=3, oob_score=use_oob, random_state=42)
#     m.fit(X_train, y_train)
#     oob_scores.append(m.oob_score_ if use_oob else float("nan"))
#     test_scores.append(accuracy_score(y_test, m.predict(X_test)))
#     oob_str = f"{oob_scores[-1]:.2%}" if use_oob else "n/a"
#     print(f"  n={n:>4}  OOB={oob_str:<7}  test={test_scores[-1]:.2%}")
#
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(n_values, oob_scores,  marker="o", label="OOB accuracy")
# ax.plot(n_values, test_scores, marker="s", label="Test accuracy")
# ax.set_xscale("log")
# ax.set_xlabel("n_estimators (log scale)")
# ax.set_ylabel("Accuracy")
# ax.set_title("Effect of n_estimators on Accuracy — Breast Cancer")
# ax.legend()
# ax.grid(True, linestyle="--", alpha=0.5)
# fig.tight_layout()
# plt.show()
#
# Both OOB and test accuracy rise sharply from 1 → ~25 trees, then plateau.
# The marginal gain from going 100 → 200 → 500 trees is typically very small,
# while training time scales linearly. A common rule of thumb: keep adding trees
# until the OOB curve flattens, then stop — usually well below 500 for tabular
# datasets of this size.
# </details>

n_values = [1, 5, 10, 25, 50, 100, 200]
oob_scores_list, test_scores_list = [], []

print("=== n_estimators vs Accuracy ===")
for n in n_values:
    # OOB requires enough trees so that every sample is OOB for at least one tree.
    # With very small forests (n < 10) some samples may never be left out, producing
    # unreliable OOB estimates and a UserWarning — disable it for those cases.
    use_oob = n >= 10
    m = RandomForestClassifier(n_estimators=n, max_depth=3, oob_score=use_oob, random_state=42)
    m.fit(X_train, y_train)
    oob_scores_list.append(m.oob_score_ if use_oob else float("nan"))
    test_scores_list.append(accuracy_score(y_test, m.predict(X_test)))
    oob_str = f"{oob_scores_list[-1]:.2%}" if use_oob else "n/a"
    print(f"  n={n:>4}  OOB={oob_str:<7}  test={test_scores_list[-1]:.2%}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(n_values, oob_scores_list,  marker="o", label="OOB accuracy")
ax.plot(n_values, test_scores_list, marker="s", label="Test accuracy")
ax.set_xscale("log")
ax.set_xlabel("n_estimators (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Effect of n_estimators on Accuracy — Breast Cancer")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()

# =============================================================================
# Step 8 — Random Forest vs. Single Decision Tree
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# dt = DecisionTreeClassifier(max_depth=3, random_state=42)
# dt.fit(X_train, y_train)
# dt_acc = accuracy_score(y_test, dt.predict(X_test))
# rf_acc = accuracy_score(y_test, rf.predict(X_test))
#
# print("=== Random Forest vs. Single Decision Tree ===")
# print(f"  Single tree test accuracy:   {dt_acc:.2%}")
# print(f"  Random forest test accuracy: {rf_acc:.2%}")
# print(f"  Gap: {rf_acc - dt_acc:+.2%}\n")
#
# With 30 correlated clinical features, a single depth-3 tree can only use 3–4
# of them, leaving most signal unused. The forest samples a random feature subset
# at every split across 100 trees, collectively leveraging all 30 features and
# averaging out each tree's variance — producing a clear accuracy advantage.
# The gap is meaningful here (unlike on simple datasets) because the problem has
# enough complexity that a shallow single tree genuinely misses important structure.
# </details>

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("=== Random Forest vs. Single Decision Tree ===")
print(f"  Single tree test accuracy:   {dt_acc:.2%}")
print(f"  Random forest test accuracy: {rf_acc:.2%}")
print(f"  Gap: {rf_acc - dt_acc:+.2%}\n")

# =============================================================================
# Step 9 — Make a Single Prediction (Inference)
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# # Use the first sample from the dataset (known to be malignant, label=0)
# sample = bc.data[[0]]
# prediction    = rf.predict(sample)[0]
# probabilities = rf.predict_proba(sample)[0]
#
# print("=== Single Prediction ===")
# print("Input features (first sample from dataset):")
# for name, val in zip(bc.feature_names, sample[0]):
#     print(f"  {name:<35} {val:.4f}")
# print(f"\nPredicted class: {bc.target_names[prediction]}")
# print(f"True label:      {bc.target_names[bc.target[0]]}")
# print("Class probabilities:")
# for cls, prob in zip(bc.target_names, probabilities):
#     print(f"  {cls:<12} {prob:.2%}")
#
# predict_proba for a random forest returns the fraction of trees that voted for
# each class. A high malignant probability here reflects that this sample has
# extreme measurements (large radius, perimeter, area) characteristic of a
# malignant tumour — values the forest learned to associate with label 0.
# </details>

sample = bc.data[[0]]
prediction    = rf.predict(sample)[0]
probabilities = rf.predict_proba(sample)[0]

print("=== Single Prediction ===")
print("Input features (first sample from dataset):")
for name, val in zip(bc.feature_names, sample[0]):
    print(f"  {name:<35} {val:.4f}")
print(f"\nPredicted class: {bc.target_names[prediction]}")
print(f"True label:      {bc.target_names[bc.target[0]]}")
print("Class probabilities:")
for cls, prob in zip(bc.target_names, probabilities):
    print(f"  {cls:<12} {prob:.2%}")
