# Loan Approval Prediction

A machine learning case study for automating loan decisioning — predicting whether a loan application should be approved or rejected based on demographic, credit, financial, and behavioral factors.

**[Open the notebook in Google Colab →](https://colab.research.google.com/drive/1lGqJRBMMNFrHyu1GpLoa2bhAjWK43ZIR?usp=sharing)**

---

## Problem Statement

Loan approval has traditionally relied on manual assessment by loan officers, who review credit histories, income, and other personal factors to judge an applicant's eligibility. This approach is often slow, inconsistent, and susceptible to human bias, creating long waiting periods and unpredictable outcomes for applicants.

This project builds an automated, data-driven model to predict loan approval outcomes, aiming to improve efficiency, accuracy, fairness, and scalability relative to manual review.

**The original brief scoped four categories of factors** that can influence a loan decision:
- **Demographic:** age, marital status, education level, employment status
- **Credit-related:** credit score, outstanding debts, credit history
- **Financial:** monthly income, loan amount, loan term, debt-to-income ratio
- **Behavioral/socio-economic:** housing situation, employment type, lifestyle factors

The provided dataset covers the credit-related and financial categories in full, but not demographic or behavioral/socio-economic factors — see [Limitations](#limitations) below for how this affected scope.

**Dataset:** [Original raw data (Google Drive)](https://drive.google.com/file/d/1n1I3hEcgN-YKycu174QRVXcqW2xmQk99/view)

## Dataset

4,269 loan applications, 13 fields — number of dependents, education, employment status, annual income, loan amount, loan term, CIBIL (credit) score, and four categories of asset value (residential, commercial, luxury, bank). Target: `loan_status` (Approved / Rejected).

Class balance: 2,656 Approved (62%) vs. 1,613 Rejected (38%) — moderately imbalanced but workable without resampling.

## Data Cleaning & Preparation

- **No missing values** in the raw data.
- **Whitespace bug**: column names and several categorical values (`education`, `self_employed`, `loan_status`) had leading whitespace from the source CSV, which silently broke dictionary-based encoding until stripped.
- **Data quality flag**: one record showed a negative `residential_assets_value` (-100,000) — not physically meaningful, flagged rather than silently dropped.
- **Feature engineering**: created `loan_to_income_ratio` (loan amount ÷ annual income) as a debt-to-income proxy, since it wasn't directly present in the data.
- **Encoding**: mapped `education`, `self_employed`, and `loan_status` to binary numeric values for modeling.

## Exploratory Data Analysis

- **CIBIL score** showed a striking, near-complete separation between classes: approved applicants clustered around 700–800, rejected applicants around 400–500, with minimal overlap.
- A correlation heatmap showed income, loan amount, and all four asset-value columns are highly correlated with each other (0.59–0.93) — wealthier applicants score higher across all of them simultaneously.
- **CIBIL score was uncorrelated with every other feature** — it carries independent predictive signal rather than duplicating information already present elsewhere.

## Modeling Approach

Five classification models were trained and compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | 83.3% |
| Decision Tree | 99.6% |
| **Random Forest** | **99.9%** |
| KNN (scaled) | 90.5% |
| SVM (scaled) | 94.0% |

KNN and SVM initially underperformed (57.8% and 62.8%) because they're distance-based algorithms sensitive to feature scale — income and asset values (in millions) dominated the distance calculation over CIBIL score (300–900). After applying `StandardScaler`, both improved substantially, illustrating why feature scaling matters for distance-based methods.

## Best Model: Random Forest

**99.9% accuracy**, precision/recall/F1 of 1.00 for both classes. Confusion matrix showed only **one misclassification out of 854 test cases**.

Feature importance:

| Feature | Importance |
|---|---|
| `cibil_score` | 80.1% |
| `loan_term` | 7.1% |
| `loan_to_income_ratio` | 4.5% |
| `loan_amount` | 1.5% |
| `residential_assets_value` | 1.4% |
| `luxury_assets_value` | 1.3% |
| `bank_asset_value` | 1.1% |
| `commercial_assets_value` | 1.1% |
| `income_annum` | 1.0% |
| `no_of_dependents` | 0.5% |
| `education` | 0.2% |
| `self_employed` | 0.1% |

CIBIL score alone accounts for roughly 80% of the model's decision-making, far outweighing every other feature combined.

## Missing Data Handling (Demonstration)

Since the original dataset had no missing values, missing-value handling was demonstrated separately: 5% of values in `self_employed`, `loan_amount`, and `cibil_score` were randomly removed and imputed (median for numeric, mode for categorical) — confirming the pipeline can handle incomplete real-world data even though it wasn't required here.

## Limitations

- The original brief scoped four factor categories (demographic, credit-related, financial, behavioral/socio-economic); the provided dataset only covers the **credit-related** and **financial** categories in full. **Demographic** factors (age, marital status) and **behavioral/socio-economic** factors (housing situation, lifestyle, employment type) named in the brief were not present in the data and could not be modeled.
- The brief also calls for handling "the dynamic nature of financial conditions, adjusting predictions based on changes in an applicant's financial standing over time" — but the dataset is a static snapshot with no time dimension, so this requirement could not be met with the data available. A production system would need richer, longitudinal data to track applicants over time.

## Fairness Considerations

CIBIL score alone accounts for ~80% of the model's decisions. This mirrors real lending practice, where credit score is a legitimate, heavily-weighted factor — but it also means the model's behavior is close to a single-variable decision rule. Two considerations follow:

1. The model's apparent objectivity is really just the objectivity of the CIBIL scoring system it defers to.
2. If CIBIL score itself reflects historical biases (e.g., correlation with income level or geography), the model would inherit and automate those biases rather than eliminate them.

A responsible deployment would need to audit the fairness properties of the CIBIL score itself before treating this model's decisions as bias-free.

## Conclusion

Tree-based models, particularly Random Forest, substantially outperformed other approaches, driven almost entirely by CIBIL score. The near-perfect accuracy (99.9%) likely reflects how this dataset was constructed rather than how loan decisions work in practice — real-world lending data would show more noise and class overlap. A production deployment would need the missing demographic/behavioral data, a fairness audit of the credit score input, and validation against genuinely noisy, real-world applications before being trusted for actual loan decisioning.

---

**Tools:** Python, Pandas, Scikit-learn, Matplotlib/Seaborn, Google Colab
