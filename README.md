<div align="center">

# 🏦 Loan Approval Prediction using PyTorch

### End-to-End Deep Learning Project for Binary Classification

An Artificial Neural Network (ANN) built with **PyTorch** to predict whether a loan application should be approved based on applicant information.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

# 📌 Overview

This project demonstrates an **end-to-end Deep Learning pipeline** for Loan Approval Prediction using **PyTorch**.

Instead of focusing only on building an ANN, this project emphasizes **production-style project structure**, **data preprocessing**, **modular code organization**, and **proper model evaluation**.

---

# 🚀 Features

- ✅ End-to-End Deep Learning Pipeline
- ✅ Modular Project Structure
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Cleaning & Feature Engineering
- ✅ One-Hot Encoding
- ✅ Ordinal Encoding
- ✅ StandardScaler
- ✅ ColumnTransformer
- ✅ Custom PyTorch Dataset
- ✅ DataLoader
- ✅ Artificial Neural Network (ANN)
- ✅ Batch Normalization
- ✅ Early Stopping
- ✅ Model Evaluation
- ✅ Prediction Pipeline

---

# 📂 Project Structure

```text
Loan-Approval-Prediction/
│
├── data/
│
├── images/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── roc_curve.png
│
├── models/
│   ├── loan_approval_model.pth
│   └── preprocessor.pkl
│
├── results/
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

**Dataset:** Loan Approval Prediction Dataset

**Target Variable**

- `loan_status`

---

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Profiling | YData Profiling |
| Model Serialization | Joblib |

---

# 🧠 Model Architecture

```text
Input Layer
      │
      ▼
Linear (Input → 128)
      │
      ▼
ReLU
      │
      ▼
BatchNorm1d
      │
      ▼
Linear (128 → 64)
      │
      ▼
ReLU
      │
      ▼
BatchNorm1d
      │
      ▼
Linear (64 → 32)
      │
      ▼
ReLU
      │
      ▼
Linear (32 → 1)
```

> **Note:** The final layer does **not** use a Sigmoid activation because the model is trained using **BCEWithLogitsLoss**, which internally applies the Sigmoid function in a numerically stable way.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Removal of invalid age records
- Binary Feature Mapping
- One-Hot Encoding
- Ordinal Encoding
- Standard Scaling
- ColumnTransformer Pipeline
- Train-Test Split
- Saving Preprocessor using Joblib

---

# 📈 Final Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **89.48%** |
| Precision | **70.64%** |
| Recall | **90.10%** |
| F1-Score | **79.19%** |
| ROC-AUC | **96.97%** |

---

# 📊 Training Results

> Add your screenshots inside the **images/** folder.

### 📉 Training & Validation Loss

```markdown
![Loss Curve](images/loss_curve.png)
```

### 📊 Confusion Matrix

```markdown
![Confusion Matrix](images/confusion_matrix.png)
```

### 📈 ROC Curve

```markdown
![ROC Curve](images/roc_curve.png)
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/sakshamengineer/Loan-Approval-Prediction.git
```

### Navigate to Project

```bash
cd Loan-Approval-Prediction
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python main.py
```

---

# 💡 Key Learnings

Through this project, I explored:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding vs Ordinal Encoding
- Binary Feature Mapping
- StandardScaler
- ColumnTransformer
- Building Custom Dataset & DataLoader
- Artificial Neural Networks in PyTorch
- BCEWithLogitsLoss
- Batch Normalization
- Early Stopping
- Model Evaluation using Precision, Recall, F1-Score & ROC-AUC

More importantly, this project helped me transition from simply implementing deep learning models to understanding the reasoning behind each design decision in a production-style ML workflow.

---

# 🎯 Future Improvements

- Learning Rate Scheduler
- Hyperparameter Optimization using Optuna
- SHAP for Explainability
- K-Fold Cross Validation
- TensorBoard Integration
- Docker Support
- FastAPI Deployment

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

# 👨‍💻 Author

## Saksham Gupta

🔗 GitHub: https://github.com/sakshamengineer

🔗 LinkedIn: *(Add Your LinkedIn Profile)*

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Happy Coding! 🚀**

</div>