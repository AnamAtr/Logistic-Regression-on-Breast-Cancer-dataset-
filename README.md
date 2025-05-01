# Logistic-Regression-on-Breast-Cancer-dataset-Task 4

This project uses **Logistic Regression** to classify tumors as **benign (0)** or **malignant (1)** using the Breast Cancer dataset. This project demonstrates that even with a relatively simple model like **Logistic Regression**, we can achieve high accuracy in distinguishing between **malignant** and **benign** tumors using structured diagnostic data.

---

##  Dataset Overview

**Attribute information** 

**1) ID number**

**2) Diagnosis** (M = malignant, B = benign)


Ten real-valued features are computed for each cell nucleus:

**a) radius** (mean of distances from center to points on the perimeter)

**b) texture** (standard deviation of gray-scale values)

**c) perimeter**

**d) area**

**e) smoothness** (local variation in radius lengths)

**f) compactness** (perimeter^2 / area - 1.0)

**g) concavity** (severity of concave portions of the contour)

**h) concave point**s (number of concave portions of the contour)

**i) symmetry**

**j) fractal dimension** ("coastline approximation" - 1)

- **Source**: UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- 
- **Shape**: 569 rows × 32 columns
- 
- **Target**: `diagnosis`  
  - `M` (Malignant) → 1  
  - `B` (Benign) → 0

---

###   Binary Classification Dataset
- We used `data.csv` containing tumor features and a binary target (`diagnosis`).
- Preprocessed: Removed `id`, cleaned missing values, and converted `M/B` to 1/0.

### Train/Test Split and Feature Standardization
- Split the dataset into **70% training** and **30% testing**.
- Used `StandardScaler()` to normalize feature values.
- Why: Standardization improves model convergence and accuracy in distance-based models like Logistic Regression.

###  Fit Logistic Regression Model
- Used `sklearn.linear_model.LogisticRegression`.
- Model trained on scaled features.
- Logistic Regression outputs probabilities between 0 and 1 using the **sigmoid function**.

###  Evaluate the Model
- **Confusion Matrix**:
- TP = 110, TN = 54, FP = 5, FN = 2

- **Accuracy**: 95.91%  
- **Precision/Recall/F1-Score** via `classification_report()`  
- **ROC Curve**: Visualized performance across thresholds (AUC close to 1)

###  Threshold Tuning and Sigmoid Function

- **Sigmoid Function** (used internally by Logistic Regression):
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]
It converts linear output `z` to a probability between 0 and 1.

- **Default threshold**: 0.5  
- We explored tuning this threshold to improve **Recall** (catch more true positives) or **Precision** (reduce false positives).

---

## Key Takeaways

| Metric         | Value         | What It Tells Us                              |
|----------------|---------------|-----------------------------------------------|
| Accuracy       | 95.9%         | High overall performance                      |
| Confusion Matrix | [[110, 5], [2, 54]] | Very few false positives/negatives      |
| Precision      | High          | Model is confident when predicting positive  |
| Recall         | High          | Very few malignant cases are missed           |
| ROC-AUC        | 0.96         | Model strongly separates the two classes      |

---


### Conclusion :
This project demonstrates that even with a relatively simple model like **Logistic Regression**, we can achieve high accuracy in distinguishing between **malignant** and **benign** tumors using structured diagnostic data.

The model's high **recall** (sensitivity) is particularly important in healthcare, where failing to detect a malignant case (false negative) can have severe consequences. By tuning thresholds and understanding the **sigmoid-based probability outputs**, medical professionals can prioritize **sensitivity over precision** — catching more true cases even at the cost of a few false positives.
- The model performed **exceptionally well**, with very few false predictions.
- **StandardScaler** improved performance by ensuring all features had equal weight.
- Removing highly correlated features (multicollinearity) improved model stability and interpretability.
- **Sigmoid function and threshold analysis** allowed exploring how model confidence affects classification decisions — important in medical diagnosis where **recall (not missing malignant cases)** is critical.


---
## Tools & Libraries Used

- Python (pandas, numpy)
- Scikit-learn (LogisticRegression, train_test_split, metrics)
- Matplotlib & Seaborn (visualizations)

---
