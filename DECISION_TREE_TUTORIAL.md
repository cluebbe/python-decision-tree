# Decision Trees with scikit-learn — Step-by-Step Tutorial

## Introduction to Decision Trees

A **decision tree** is a supervised machine learning algorithm that learns to make predictions by repeatedly splitting data into smaller groups based on feature values. The result is a tree-like structure of nested if/else rules that is easy to follow and explain.

### How it works

At each step during training, the algorithm searches for the feature and threshold that best separates the data — for example, *"is petal width ≤ 0.80 cm?"*. Samples that satisfy the condition go left; the rest go right. This process repeats recursively at each resulting node until a stopping condition is met (such as a maximum depth or a minimum number of samples).

The final tree is made up of three types of nodes:

| Component | Role |
|---|---|
| Internal node | A feature/threshold test |
| Branch | The outcome of the test (true or false) |
| Leaf | A predicted class (classification) or value (regression) |

### How splits are chosen

The algorithm picks the split that maximally reduces a splitting criterion. Two common criteria are:

- **Gini impurity** — measures the probability of misclassifying a randomly chosen sample. A pure node (all one class) has Gini = 0.
- **Entropy (information gain)** — measures the reduction in uncertainty after a split, borrowed from information theory.

Both criteria produce similar trees in practice. Gini is slightly faster to compute and is the scikit-learn default.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Highly interpretable — rules can be read and explained | Prone to overfitting without depth constraints |
| No feature scaling required | High variance — small data changes can alter the tree significantly |
| Handles both numerical and categorical features | Axis-aligned splits only — struggles with diagonal decision boundaries |
| Fast to train and predict | Biased toward features with many possible split points |

### Overfitting and regularisation

An unconstrained decision tree will grow until every leaf contains a single sample, perfectly memorising the training data but generalising poorly to new data. This is **overfitting**. The most common way to prevent it is to limit the tree's complexity:

- **`max_depth`** — hard cap on how many levels the tree can have
- **`min_samples_split`** — require a minimum number of samples before a node can be split
- **`min_samples_leaf`** — require a minimum number of samples in every leaf

---

## Preparation — Environment Setup

Before running any code, install Python and set up an isolated environment.

**Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/). Verify it is available in your terminal:

```bash
python3 --version
```

> **Windows users:** during installation, tick **"Add Python to PATH"** so the `python` and `pip` commands are available in your terminal.

Then set up an isolated environment:

```bash
# 1. Create and enter your project folder
mkdir decision-tree && cd decision-tree

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install numpy matplotlib scikit-learn

# 5. When you're done, deactivate
deactivate
```

> **Why a virtual environment?** It keeps the packages for this project separate from your system Python and other projects, avoiding version conflicts.

## Preparation — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
```

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — built-in datasets (Iris)
- **sklearn.model_selection** — splitting data into train/test sets
- **sklearn.tree** — the decision tree model and text/visual export utilities
- **sklearn.metrics** — tools to evaluate model performance

---

## Step 1 — Load & Explore the Data

Load the Iris dataset and assign the feature matrix to `X` and the target vector to `y`. Print the number of samples, the feature names, the class names, and the class distribution.

<details>
<summary>Solution</summary>

```python
iris = load_iris()

X = iris.data        # shape: (150, 4) — sepal/petal length and width
y = iris.target      # 0=setosa, 1=versicolor, 2=virginica

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}  ->  {iris.feature_names}")
print(f"Classes:  {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}\n")
```

The **Iris dataset** contains 150 flower samples across 3 species, each described by 4 features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

`X` holds the feature matrix (150 rows × 4 columns) and `y` holds the integer class labels. Printing `np.bincount(y)` confirms the dataset is balanced — 50 samples per class.

</details>

---

## Step 2 — Split into Train / Test Sets

Split the data into training and test sets using an 80/20 ratio. Ensure the split is reproducible and that class proportions are preserved in both sets.

<details>
<summary>Solution</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for evaluation
    random_state=42,  # reproducibility
    stratify=y,       # keep class proportions equal in both splits
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")
```

- **test_size=0.2** — 20% of samples (30) are held out for evaluation; 80% (120) are used for training.
- **random_state=42** — fixes the random seed so results are reproducible.
- **stratify=y** — ensures each split contains the same class proportions as the full dataset, preventing an imbalanced split by chance.

</details>

---

## Step 3 — Train a Decision Tree

Train a `DecisionTreeClassifier` with a maximum depth of 3 and Gini impurity as the splitting criterion. Use a fixed random state for reproducibility, then use export_text to print a text representation of the learned tree rules labelled with feature names.

<details>
<summary>Solution</summary>

```python
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
```

Key hyperparameters:

| Parameter | Effect |
|---|---|
| `max_depth` | Limits how deep the tree grows; prevents overfitting |
| `criterion` | `"gini"` measures node impurity; `"entropy"` uses information gain — both work similarly in practice |
| `min_samples_split` | Minimum samples required to split a node (not set here, defaults to 2) |

`export_text(clf, feature_names=...)` prints a human-readable text representation of the learned tree rules.

</details>

---

## Step 4 — Evaluate the Model

Generate predictions on the test set. Print the overall accuracy as a percentage and a full classification report showing per-class precision, recall, and F1-score.

<details>
<summary>Solution</summary>

```python
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"=== Evaluation ===")
print(f"Accuracy: {accuracy:.2%}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

- **accuracy_score** — the fraction of test samples classified correctly.
- **classification_report** — per-class breakdown of precision, recall, and F1-score:
  - **Precision** — ratio of all samples predicted as class X, how many actually were X?
  - **Recall** — ratio of all actual class X samples, how many were correctly identified?
  - **F1-score** — harmonic mean of precision and recall; a single balanced metric.

</details>

---

## Step 5 — Understand Feature Importance

Extract the feature importances from the trained model and print them ranked from most to least important. In 2–3 sentences, explain what feature importance means in the context of a decision tree and which features you would expect to rank highest for the Iris dataset.

<details>
<summary>Solution</summary>

```python
# Feature importance = how much each feature reduced impurity across all splits
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank}. {iris.feature_names[idx]:<30} {importances[idx]:.4f}")
print()
```

Feature importance measures how much each feature reduced impurity (Gini) across all splits in the tree. Values sum to 1.0. A higher score means the feature was more influential in the model's decisions.

For the Iris dataset, petal features typically dominate over sepal features.

</details>

---

## Step 6 — Visualise

Display the following three plots:

- **6a** — The decision tree diagram, with nodes filled by majority class and feature/class names labelled.
- **6b** — A confusion matrix heatmap for the test set predictions using `ConfusionMatrixDisplay`.
- **6c** — A bar chart of feature importances.

Do not save figures to disk — display them inline.

<details>
<summary>Solution</summary>

### 6a: Decision Tree Diagram

```python
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
```

Renders the full tree structure. Each node shows:
- The split condition (e.g. `petal width (cm) <= 0.75`)
- The Gini impurity of that node
- The sample count
- The majority class (colour-coded when `filled=True`)

### 6b: Confusion Matrix

```python
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
```

A grid showing actual vs. predicted classes. The diagonal contains correct predictions; off-diagonal cells reveal which classes are confused with each other.

### 6c: Feature Importance Bar Chart

A bar chart ranking features by their importance score, making it easy to see which features the tree relied on most.

```python
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
```

</details>

---

## Step 7 — Overfitting Demo: Depth vs. Accuracy

Train 10 decision trees with `max_depth` values from 1 to 10. Record train and test accuracy for each depth and plot both curves on the same graph. In 2–3 sentences, describe what the plot reveals about the relationship between tree depth and overfitting.

<details>
<summary>Solution</summary>

```python
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
```

This loop trains 10 trees with increasing depth and records both **training** and **test** accuracy.

- At low depths: the model is too simple (underfitting) — both accuracies are modest.
- At the right depth: test accuracy peaks.
- At high depths: training accuracy reaches 100% but test accuracy plateaus or drops — this is **overfitting**. The model has memorised the training data and generalises poorly.

The resulting plot makes the overfitting effect visually obvious.

</details>

---

## Step 8 — Make a Single Prediction (Inference)

Using the trained classifier, predict the class and class probabilities for a new sample with the following measurements: sepal length = 5.1, sepal width = 3.5, petal length = 1.4, petal width = 0.2. Print both the predicted class label and the probability for each class.

<details>
<summary>Solution</summary>

```python
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # a new, unseen flower
prediction = clf.predict(sample)[0]
probabilities = clf.predict_proba(sample)[0]

print("\n=== Single Prediction ===")
print(f"Input features: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Predicted class: {iris.target_names[prediction]}")
print("Class probabilities:")
for cls, prob in zip(iris.target_names, probabilities):
    print(f"  {cls:<12} {prob:.2%}")
```

- `predict` returns the most likely class label.
- `predict_proba` returns the probability distribution across all classes — useful when you need a confidence score, not just a hard label.

This sample (short petals, narrow petals) is characteristic of *Iris setosa* and the model should predict it with high confidence.

</details>
