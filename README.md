# Telco Customer Churn: Exploratory Data Analysis & Deep Learning Modeling

**Course:** Exploratory Data Analysis and Visualization (Course Code: U21ADP05)  
**Academic Year:** 2026–2027 (Odd Sem)  
**Assignment:** Assignment II Case Study  

---

## 📌 Project Overview
Customer churn poses a major challenge in the telecommunications industry. Retaining existing subscribers is significantly more cost-effective than acquiring new ones. This project focuses on performing comprehensive Exploratory Data Analysis (EDA), statistical visualization, and applying a Deep Learning Multi-Layer Perceptron (MLP) architecture to predict customer churn based on subscription patterns, demographics, and payment metrics.

---

## 📊 Dataset Information
* **Dataset Name:** Telco Customer Churn
* **Source:** IBM Sample Data / Kaggle
* **Data Dimensions:** 7,043 rows × 21 attributes (exceeds the 15-feature minimum requirement)
* **Dataset Type:** Tabular (Numerical & Categorical features)
* **Target Feature:** `Churn` (Binary: `Yes` / `No`)

### Key Attributes:
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## 🛠️ Project Structure & Workflow

1. **Data Preprocessing & Cleaning:**
   - Handled missing and whitespace values in `TotalCharges`.
   - Converted data types and dropped non-predictive identifiers (`customerID`).
   - Categorical feature encoding via `LabelEncoder`.
   - Data partitioning into **70% Training**, **15% Validation**, and **15% Testing** sets.
   - Numerical feature standardization using `StandardScaler`.

2. **Exploratory Data Analysis (Minimum 5 Visualizations):**
   - **Visual 1 (Univariate Distribution):** Histogram and KDE plot of `MonthlyCharges`.
   - **Visual 2 (Categorical Distribution):** Bar plot showing Churn distribution across `Contract` types.
   - **Visual 3 (Multivariate Analysis):** Correlation Heatmap of encoded attributes against the target variable `Churn`.
   - **Visual 4 (Distribution Exploration):** Boxplot showing customer `tenure` distribution by Churn status.
   - **Visual 5 (Text/Category Visual):** WordCloud visualization of customer `PaymentMethod` preferences.

3. **Deep Learning Model (MLP Architecture):**
   - Built a 3-layer Sequential Neural Network (Dense 64 $\rightarrow$ Dense 32 $\rightarrow$ Dense 1).
   - Applied **ReLU** activation functions in hidden layers and **Sigmoid** activation for output prediction.
   - Integrated **Dropout regularization** (30% and 20%) and **EarlyStopping** to prevent overfitting.
   - Optimized using **Adam** optimizer and **Binary Crossentropy** loss.

4. **Model Performance Evaluation (4 Visual Plots):**
   - Loss vs. Epoch Curve (Train vs. Validation loss)
   - Accuracy vs. Epoch Curve (Train vs. Validation accuracy)
   - Confusion Matrix Heatmap
   - Receiver Operating Characteristic (ROC) Curve & AUC Score

---

## 📁 Repository Contents

```text
.
├── EDA_Assignment_II.ipynb   # Executable Google Colab / Jupyter notebook containing code & plots
├── Project_Report.pdf         # Comprehensive assignment report following required structure
├── README.md                  # Project overview and instructions (this file)
└── dataset_link.txt           # Direct download link for the raw CSV dataset
