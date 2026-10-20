# Employee Attrition Prediction

Is it possible to spot who's going to quit before they actually resign?

That's what this project set out to answer. It started as a group assignment for "Statistical Thinking for Data Science" at UTS in 2026, and my part was the data preprocessing and the Logistic Regression model.

![SHAP beeswarm](figures/shap_beeswarm.png)

## Why bother?

Losing an employee costs a lot more than people expect. Between recruiting, training someone new and the productivity lost in between, the bill is often somewhere from half to twice the person's yearly salary. And most companies only learn the reasons in an exit interview, by which point nothing can be done.

So the goal here was to work backwards: take the HR records a company already keeps, find the people most likely to leave, and figure out what's pushing them.

## Data

The dataset is IBM's [HR Analytics Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) on Kaggle. It covers 1,470 employees, with details like age, role, salary, overtime and a handful of satisfaction ratings. Only around 16% of them left, which makes the two classes pretty lopsided.

Worth knowing: this data isn't real. IBM generated it, so while the patterns look believable, they don't describe any actual company.

## Preprocessing (my part)

Before any modelling, the raw data needed some work:

- Dropped four columns that carried no useful information: `EmployeeCount`, `StandardHours` and `Over18` have the same value for everyone, and `EmployeeNumber` is just an ID.
- Converted the seven text columns (things like department, job role and overtime) into numbers.
- Split the data 80/20 into training and test sets, keeping the same share of leavers in each.
- Used SMOTE on the training set only, generating synthetic leavers until both classes were the same size. The test set kept its real 16/84 split so the results would reflect reality.
- Scaled the features for Logistic Regression, since it's sensitive to features being on different scales.

## What the data showed

A few things stood out straight away, before any model was involved:

- Overtime: Employees doing overtime left 30.5% of the time. Without overtime, it was 10.4%, about a third as often.
- Sales Reps: At 39.8%, they had the worst attrition of any job.
- Younger staff: Among 18 to 25 year-olds, 35.8% left. That's close to four times the rate for people aged 36 to 45.
- Salary On average, leavers were paid roughly $2,046 less per month than people who stayed.

![ROC curves](figures/roc_all_models.png)

## Models

Three models went head to head: Logistic Regression, Random Forest and XGBoost.

I built the Logistic Regression one. It was meant as the baseline, and its big advantage is that you can read it directly. Each feature gets a coefficient, so you can tell HR something like "higher job satisfaction lowers the chance someone leaves" without explaining how a pile of decision trees works. Stock option level and job satisfaction had the largest coefficients. Its cross-validated AUC was 0.833 (± 0.018).

The metric we watched most closely was **recall**. Wrongly flagging someone who was going to stay just means an extra conversation. Missing someone who's about to leave means hiring a replacement.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.799 | 0.400 | 0.511 | 0.449 | 0.718 |
| Random Forest | 0.789 | 0.359 | 0.404 | 0.380 | 0.735 |
| XGBoost | 0.765 | 0.355 | 0.575 | 0.439 | 0.741 |

XGBoost found the most real leavers, so the team picked it as the final model. SHAP was used to explain its predictions, and the top drivers matched what Logistic Regression had suggested: stock options, job satisfaction and environment satisfaction, with overtime not far behind.

Employees were also grouped into low, medium and high risk based on XGBoost's predicted probabilities, and most of the people who really left landed in the higher groups.

## Suggestions for HR

1. Keep a close eye on overtime, particularly in roles that already lose a lot of people.
2. Check whether Sales Reps are paid fairly and have somewhere to grow.
3. Put proper onboarding and mentoring in place for staff under 25.
4. Extend stock options beyond senior employees.

## What I'd change

Since preprocessing was my area, most of these are on me:

- SMOTE should go inside cross-validation: I ran SMOTE first and cross-validated afterwards, which meant synthetic copies of the same employees could end up in both the training and validation folds. It made the cross-validation scores look better than they really were. Random Forest, for example, hit 0.967 in cross-validation but only 0.735 on the test set. Wrapping SMOTE in an `imblearn` Pipeline fixes that.
- One-hot encoding instead of label encoding: Columns like job role have no order, but label encoding gives them one (0, 1, 2...). That's harmless for tree models, but it can mislead Logistic Regression.
- Only one fix for imbalance: XGBoost got both SMOTE and `scale_pos_weight`, which double-corrects. Choosing one would likely help precision.
- Recall is still too low: Even XGBoost misses around 4 out of every 10 leavers.

If anyone used something like this for real, it would need a fairness check first. A model that labels people as flight risks could be used against them.

## Running it

Clone this repo, then:

```bash
cd employee-attrition-prediction
pip install -r requirements.txt
jupyter notebook notebooks/attrition_analysis.ipynb
```

Grab the CSV from Kaggle and drop it into `data/` before running the notebook. The full report is at `report/report.pdf`.

Tools used: Python, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, matplotlib, seaborn
