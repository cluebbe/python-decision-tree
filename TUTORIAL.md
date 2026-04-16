# Decision Trees with scikit-learn — Step-by-Step Tutorial

A decision tree is a supervised learning algorithm that splits data into branches based on feature values, forming a tree-like structure of decisions.

| Component | Role |
|---|---|
| Internal node | A feature/threshold test |
| Branch | The outcome of the test |
| Leaf | A predicted class (classification) or value (regression) |

---

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
X = iris.data   # shape: (150, 4)
y = iris.target # 0=setosa, 1=versicolor, 2=virginica
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
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

- **test_size=0.2** — 20% of samples (30) are held out for evaluation; 80% (120) are used for training.
- **random_state=42** — fixes the random seed so results are reproducible.
- **stratify=y** — ensures each split contains the same class proportions as the full dataset, preventing an imbalanced split by chance.

</details>

---

## Step 3 — Train a Decision Tree

Train a `DecisionTreeClassifier` with a maximum depth of 3 and Gini impurity as the splitting criterion. Use a fixed random state for reproducibility, then print a text representation of the learned tree rules labelled with feature names.

<details>
<summary>Solution</summary>

```python
clf = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
clf.fit(X_train, y_train)
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
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

- **accuracy_score** — the fraction of test samples classified correctly.
- **classification_report** — per-class breakdown of precision, recall, and F1-score:
  - **Precision** — of all samples predicted as class X, how many actually were X?
  - **Recall** — of all actual class X samples, how many were correctly identified?
  - **F1-score** — harmonic mean of precision and recall; a single balanced metric.

</details>

---

## Step 5 — Understand Feature Importance

Extract the feature importances from the trained model and print them ranked from most to least important. In 2–3 sentences, explain what feature importance means in the context of a decision tree and which features you would expect to rank highest for the Iris dataset.

<details>
<summary>Solution</summary>

```python
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
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
plot_tree(clf, feature_names=..., class_names=..., filled=True, rounded=True, ax=ax)
```

Renders the full tree structure. Each node shows:
- The split condition (e.g. `petal width (cm) <= 0.75`)
- The Gini impurity of that node
- The sample count
- The majority class (colour-coded when `filled=True`)

### 6b: Confusion Matrix

```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ...)
```

A grid showing actual vs. predicted classes. The diagonal contains correct predictions; off-diagonal cells reveal which classes are confused with each other.

### 6c: Feature Importance Bar Chart

A bar chart ranking features by their importance score, making it easy to see which features the tree relied on most.

</details>

---

## Step 7 — Overfitting Demo: Depth vs. Accuracy

Train 10 decision trees with `max_depth` values from 1 to 10. Record train and test accuracy for each depth and plot both curves on the same graph. In 2–3 sentences, describe what the plot reveals about the relationship between tree depth and overfitting.

<details>
<summary>Solution</summary>

```python
for d in range(1, 11):
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    # record train and test accuracy
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
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = clf.predict(sample)[0]
probabilities = clf.predict_proba(sample)[0]
```

- `predict` returns the most likely class label.
- `predict_proba` returns the probability distribution across all classes — useful when you need a confidence score, not just a hard label.

This sample (short petals, narrow petals) is characteristic of *Iris setosa* and the model should predict it with high confidence.

</details>
