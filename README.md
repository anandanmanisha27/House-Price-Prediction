readme_content = """
# 🏠 House Price Prediction using Machine Learning

This project aims to predict house prices based on various features using machine learning regression algorithms. It includes data exploration, preprocessing, visualization, and model evaluation.

---

## 📊 Dataset

- Format: `.xlsx`
- Source: Kaggle or similar housing dataset
- Target Variable: `SalePrice`
- Features include:
  - Categorical: MSZoning, LotConfig, BldgType, Exterior1st, etc.
  - Numerical: LotArea, YearBuilt, TotalBsmtSF, OverallCond, etc.

---

## 🔍 Exploratory Data Analysis

- Identified categorical and numerical columns.
- Plotted:
  - Correlation heatmap
  - Unique value counts for categorical features
  - Distribution of each categorical variable

---

## 🛠️ Preprocessing

- Dropped `Id` column
- Handled missing values using mean for `SalePrice` and dropped remaining null rows
- One-Hot Encoding for categorical features using `OneHotEncoder`
- Final dataset created by combining numerical and encoded categorical features

---

## 🧠 Models Trained

- **Support Vector Regression (SVR)**
- **Random Forest Regressor**
- **Linear Regression**

### 🧪 Model Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**

Each model was evaluated using an 80-20 train-test split.

---

## 📈 Prediction on New Data

A new house entry is manually input as a DataFrame. Preprocessing steps (One-Hot Encoding and column alignment) are applied to ensure it matches training data.

