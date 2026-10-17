# MSCS_634_Lab_4: Regression Analysis with Regularization Techniques

**Name:** Sandeep Reddy Banda
**Course:** MSCS-634 – Advanced Big Data and Data Mining

## Purpose
The goal of this lab was to practice different regression techniques on the Diabetes dataset from `sklearn.datasets` and see how regularization helps with overfitting. I built Simple Linear, Multiple, and Polynomial Regression models, then applied Ridge and Lasso Regression. Every model was evaluated with MAE, MSE, RMSE, and R², and I made plots to compare them.

## Files
- `MSCS_634_Lab_4.ipynb` – the full notebook (code, outputs, plots, and written analysis)
- `README.md` – this file

## Key Insights
| Model | Test R² | Test RMSE |
|---|---|---|
| Simple Linear (bmi) | 0.233 | 63.73 |
| Multiple Linear (10 features) | 0.453 | 53.85 |
| Polynomial degree 2 | 0.416 | 55.64 |
| Polynomial degree 3 | -14.561 | 287.13 |
| Ridge (degree 2) | 0.500 | 51.49 |
| Lasso (degree 2) | 0.506 | 51.14 |

- One feature (`bmi`) is not enough. Using all 10 features improved R² from 0.23 to 0.45.
- Polynomial regression overfits fast on this small dataset (442 rows). Degree 3 had a train R² of 0.88 but a test R² of -14.6.
- Ridge and Lasso fixed the overfitting of the degree 2 model. Lasso did slightly better and kept only 14 of 65 features, but the difference from Ridge is small.
- `bmi`, `s5`, and `bp` were the most important predictors. `s1` and `s2` are highly correlated, and regularization stabilized their coefficients.
- Even the best model only explains about 50% of the variance, so these features alone are limited.

## Challenges and Decisions
- Used an 80/20 split with `random_state=42` for all models so the comparison is fair.
- Standardized features before Ridge/Lasso because the penalty depends on feature scale.
- Chose alpha using 5-fold cross-validation on the training set only, to avoid using the test data for tuning.
- Used degree 2 for regularization since degree 3+ has more features than training rows.
- The dataset had no missing values, so no cleaning was needed.

## How to Run
```
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook MSCS_634_Lab_4.ipynb
```
