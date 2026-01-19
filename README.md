# Advertising Sales Prediction using Machine Learning


---

## Project Overview

This project analyzes the relationship between **advertising expenditure** across multiple media channels and **product sales**.  
It follows a complete data science workflow, including exploratory data analysis (EDA), statistical modeling, outlier handling, and regression-based machine learning to identify the most effective predictors of sales.

The primary objective is to quantify the impact of **TV, Radio, and Newspaper advertising** and select the **best-performing regression model** based on predictive accuracy.

---

## Dataset Summary

The dataset contains **200 observations** with advertising spend recorded across three channels and the resulting sales.

| Feature | Description |
|------|------------|
| TV | Advertising spend on television |
| Radio | Advertising spend on radio |
| Newspaper | Advertising spend on newspapers |
| Sales | Product sales (target variable) |

The dataset is complete, numerically consistent, and contains no missing values.

---

## Data Preprocessing

The dataset was loaded from a CSV file and cleaned by removing an unnecessary index column.  
Data types were verified, and descriptive statistics were computed to assess central tendency and dispersion.

Outlier detection using the **IQR method** identified two extreme observations, which were removed to improve robustness without affecting overall trends.

| Dataset Version | Rows |
|---------------|------|
| Original | 200 |
| Cleaned | 198 |

---

## Exploratory Data Analysis (EDA)

Univariate analysis showed that **TV and Radio advertising are approximately symmetrically distributed**, while Newspaper spending exhibits right skewness and mild outliers.  
Sales data shows moderate positive skew but no extreme anomalies.

Bivariate analysis revealed strong linear relationships between advertising spend and sales, particularly for TV advertising.


<img width="614" height="451" alt="image" src="https://github.com/user-attachments/assets/0c64df66-229c-47ed-8d9f-1a259e2a41ff" />


<img width="683" height="447" alt="image" src="https://github.com/user-attachments/assets/670b837b-d517-40d2-ab38-93a1de9ce510" />

### Correlation with Sales

| Channel | Pearson Correlation |
|-------|--------------------|
| TV | 0.78 |
| Radio | 0.58 |
| Newspaper | 0.23 |

TV advertising shows the strongest association with sales, while Newspaper advertising has limited predictive value.

---

## Statistical Modeling (OLS Regression)

A multivariate **Ordinary Least Squares (OLS)** regression model was used to quantify the combined effect of advertising channels.

Key findings:
- TV and Radio coefficients are **statistically significant**
- Newspaper advertising is **not statistically significant**
- Model explains approximately **89.7% of the variance in sales**

This confirms that TV and Radio are the primary drivers of sales performance.
```

 OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.897
Model:                            OLS   Adj. R-squared:                  0.896
Method:                 Least Squares   F-statistic:                     570.3
Date:                Mon, 19 Jan 2026   Prob (F-statistic):           1.58e-96
Time:                        17:19:21   Log-Likelihood:                -386.18
No. Observations:                 200   AIC:                             780.4
Df Residuals:                     196   BIC:                             793.6
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.9389      0.312      9.422      0.000       2.324       3.554
TV             0.0458      0.001     32.809      0.000       0.043       0.049
Radio          0.1885      0.009     21.893      0.000       0.172       0.206
Newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
==============================================================================
Omnibus:                       60.414   Durbin-Watson:                   2.084
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
Skew:                          -1.327   Prob(JB):                     1.44e-33
Kurtosis:                       6.332   Cond. No.                         454.
==============================================================================

```
---

## Machine Learning Models

Several regression models were trained using a train–test split (75% / 25%).  
Feature scaling was applied where required using standardized pipelines.

### Models Evaluated
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  

---

## Model Performance Comparison (Test Set)

| Model | R² Score | MAE | RMSE |
|-----|---------|-----|------|
| **Random Forest** | **0.982** | **0.58** | **0.70** |
| Lasso Regression | 0.895 | 1.38 | 1.68 |
| Ridge Regression | 0.894 | 1.40 | 1.70 |
| Linear Regression | 0.894 | 1.40 | 1.70 |

The **Random Forest Regressor** significantly outperforms linear models, capturing nonlinear relationships between advertising channels and sales.

---

## Training vs Testing Performance

| Model | Train R² | Test R² |
|-----|---------|--------|
| Random Forest | 0.997 | 0.982 |
| Linear Regression | 0.897 | 0.894 |
| Ridge Regression | 0.897 | 0.894 |
| Lasso Regression | 0.896 | 0.895 |

Minimal performance gap indicates strong generalization with limited overfitting.

---

## Key Insights

- TV advertising has the highest impact on sales
- Radio advertising contributes significantly but less than TV
- Newspaper advertising has minimal predictive power
- Nonlinear models outperform linear regression
- Random Forest provides the best balance of accuracy and stability

---

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-learn

---



## How to Run the Project

Install dependencies and run the notebook:

```bash
pip install -r requirements.txt
jupyter notebook
The trained SVM pipeline was serialized and saved as:
