# Diabetes Risk Prediction Using Support Vector Machines (SVM)

## Overview
This repository implements Support Vector Machines (SVM) for binary classification of diabetes risk using clinical health data. It demonstrates a complete machine learning pipeline from exploratory analysis to model optimization and visualization of decision boundaries.


## Objectives
- Preprocess diabetes health data (handling missing values, feature scaling)
- Train and compare linear vs. non-linear (RBF kernel) SVM models
- Evaluate clinical performance metrics (accuracy, precision, recall)
- Optimize model performance through hyperparameter tuning
- Visualize decision boundaries using PCA dimensionality reduction


## Tools & Technologies
- **Language:** Python  
- **Environment:** Google Colab / Jupyter Notebook  
- **Key Libraries:**  
  - `pandas`, `numpy` – Data handling  
  - `matplotlib`, `seaborn` – Visualization  
  - `scikit-learn` – ML modeling (SVC, StandardScaler, GridSearchCV)  
  - `PCA` – Dimensionality reduction for visualization


## Workflow & Highlights

### Data Preprocessing
- Verified no missing values in dataset  
- Standardized all numerical features using `StandardScaler`  
- Retained all 8 clinical features for modeling  
- Stratified 70-30 train-test split to maintain class balance  

### Model Building
- **Linear SVM**: Baseline implementation  
- **RBF Kernel SVM**: Non-linear classification  
- **Hyperparameter Tuning**: Used `GridSearchCV` for optimal `C` and `gamma`  
- **Decision Boundary Visualization**: Used 2D PCA projection


## Evaluation Metrics

| Model            | Accuracy | Precision (0/1)  | Recall (0/1)  | F1-Score (0/1) |
|------------------|----------|------------------|---------------|----------------|
| Linear SVM       | 73.77%   | 0.76 / 0.65      | 0.85 / 0.40   | 0.80 / 0.56    |
| RBF Kernel SVM   | 75.28%   | 0.78 / 0.69      | 0.87 / 0.54   | 0.82 / 0.61    |
| Tuned RBF SVM    | 74.89%   | 0.76 / 0.71      | 0.89 / 0.48   | 0.82 / 0.57    |


## Cross-Validation Results
- **Best Parameters Found**: `C = 1`, `gamma = 0.01`  
- **Best Cross-Validation Accuracy**: **77.83%**


## Advanced Analysis
- Comparative performance analysis of linear vs. non-linear kernels  
- Impact of hyperparameter tuning on model generalization  
- Decision boundary visualization through PCA projection  
- Confusion matrix analysis for clinical interpretability


## Visual Outputs
- Confusion matrices for all model variants  
- Decision boundary visualization (PCA-reduced 2D projection)  
- Performance comparison between model types  
- Classification reports with precision-recall metrics


## Key Insights
- **RBF kernel outperformed linear SVM** (75.28% vs 73.77% test accuracy)
- Model shows **better performance identifying non-diabetic cases** (higher recall for class 0)
- Hyperparameter tuning **improved generalization** and reduced overfitting
- Clear class **separation visible in PCA decision boundary**
- **Glucose** and **BMI** emerged as the most influential features


## Learning Outcomes
- Practical implementation of SVM for medical classification
- Experience with kernel selection and hyperparameter tuning
- Techniques for visualizing high-dimensional decision boundaries
- Clinical interpretation of machine learning results
- Balancing model complexity with generalization performance
