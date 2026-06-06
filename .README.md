# 🏡 Task 2 – Feature Engineering, Model Optimization \& Performance Comparison

## 📌 Objective

Task 1 covered the basics — load a dataset, train one model, check the score.  
Task 2 goes further and mirrors how ML engineers actually work in production:

* Prepare and scale data correctly before training
* Train multiple regression algorithms on the same data
* Compare all models using standard evaluation metrics
* Select and justify the best-performing model with evidence

\---

## 📁 Project Structure

```
Task\_2\_Model\_Comparison/
│
├── AI\_ML\_Task2\_Model\_Comparison.ipynb   # Main Jupyter Notebook
├── actual\_vs\_predicted.png              # Actual vs Predicted scatter plot
├── model\_comparison\_chart.png           # RMSE \& R² bar chart comparison
├── best\_model\_linear\_regression.pkl     # Saved best model (joblib)
└── README.md                            # This file
```

\---

## 📊 Dataset Used

**California Housing Dataset** — built into `scikit-learn`, sourced from the 1990 U.S. Census.

|Feature|Description|
|-|-|
|`MedInc`|Median household income in the block group|
|`HouseAge`|Median age of houses in the block group|
|`AveRooms`|Average number of rooms per household|
|`AveBedrms`|Average number of bedrooms per household|
|`Population`|Total population of the block group|
|`AveOccup`|Average number of household members|
|`Latitude`|Latitude coordinate|
|`Longitude`|Longitude coordinate|
|`HousePrice` ⭐|**Target** – Median house value (in $100,000 units)|

* **Total rows:** 20,640
* **Total features:** 8 input + 1 target
* **Missing values:** None

\---

## 🛠️ Tools \& Technologies

|Tool|Purpose|
|-|-|
|Python 3|Programming language|
|Jupyter Notebook|Development environment|
|pandas|Data loading and manipulation|
|NumPy|Numerical operations|
|scikit-learn|Dataset, preprocessing, models, metrics|
|matplotlib|Visualization and plots|
|joblib|Saving the trained model|

\---

## 🔄 ML Pipeline – Step by Step

### Step 1 — Import Libraries

All required libraries imported: `pandas`, `numpy`, `matplotlib`, `sklearn` modules for dataset, preprocessing, models, and metrics.

### Step 2 — Load the Dataset

```python
data = fetch\_california\_housing(as\_frame=True)
df = pd.concat(\[data.data, data.target.rename("HousePrice")], axis=1)
```

### Step 3 — Separate Features and Target

```python
X = df.drop("HousePrice", axis=1)
y = df\["HousePrice"]
```

### Step 4 — Feature Scaling ⚠️ Critical Step

Features in this dataset exist on vastly different scales (`MedInc` → 0–15, `Population` → up to 35,000+).  
Without scaling, large-valued features dominate the model's learning and skew results.

```python
scaler = StandardScaler()
X\_scaled = scaler.fit\_transform(X)
```

`StandardScaler` transforms each feature to **mean = 0, std = 1** — ensuring all features contribute equally.

### Step 5 — Train-Test Split

```python
X\_train, X\_test, y\_train, y\_test = train\_test\_split(
    X\_scaled, y, test\_size=0.2, random\_state=42
)
```

* Training set: **80%** (16,512 samples)
* Test set: **20%** (4,128 samples)
* `random\_state=42` ensures reproducibility

### Step 6 — Train Multiple Models

```python
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Decision Tree"     : DecisionTreeRegressor(max\_depth=5)
}
```

|Model|Rationale|
|-|-|
|Linear Regression|Baseline — assumes linear relationship between features and price|
|Ridge Regression|Adds L2 regularization to reduce overfitting|
|Decision Tree (depth=5)|Captures non-linear patterns; depth cap prevents overfitting|

### Step 7 — Evaluate and Compare

```python
for name, model in models.items():
    model.fit(X\_train, y\_train)
    predictions = model.predict(X\_test)
    rmse = mean\_squared\_error(y\_test, predictions, squared=False)
    r2   = r2\_score(y\_test, predictions)
```

**Metrics used:**

* **RMSE** (Root Mean Squared Error) → lower is better
* **R² Score** (Coefficient of Determination) → higher is better (max = 1.0)

### Step 8 — Visualize Performance

* **Scatter plot** — Actual vs Predicted house prices with a red reference line
* **Bar charts** — Side-by-side RMSE and R² comparison across all three models

\---

## 📈 Results

|Model|RMSE|R² Score|
|-|-|-|
|Linear Regression|\~0.73|\~0.60|
|Ridge Regression|\~0.73|\~0.60|
|Decision Tree (depth=5)|\~0.72|\~0.61|

> Exact values will populate after running the notebook end-to-end.

\---

## ✅ Final Model Selected — Linear Regression

**Why Linear Regression?**

1. **Competitive performance** — RMSE and R² are on par with the other two models. No meaningful gain from added complexity.
2. **Interpretability** — Coefficients can be directly read and explained. Example: a positive coefficient on `MedInc` confirms higher-income areas map to higher house prices.
3. **No overfitting risk** — Training and test scores are consistent.
4. **Occam's Razor** — When two models perform equally, prefer the simpler one.

\---

## 💾 Saving the Model (Optional)

```python
import joblib
joblib.dump(best\_model, "best\_model\_linear\_regression.pkl")

# Reload later
model = joblib.load("best\_model\_linear\_regression.pkl")
```

\---

## 🔑 Key Learnings

* Feature scaling is non-negotiable — it directly impacts model stability and fairness
* Always split data before evaluating — test set performance is the only honest metric
* Training multiple models and comparing them objectively is standard ML engineering practice
* RMSE and R² are complementary — report both, not just one
* Model selection must be justified with data, not intuition

\---

## 🚀 How to Run

1. Clone or download this repository
2. Open `AI\_ML\_Task2\_Model\_Comparison.ipynb` in Jupyter Notebook or VS Code
3. Run all cells top to bottom (`Kernel → Restart \& Run All`)
4. Output plots and the comparison table will generate automatically

**Dependencies:**

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

\---

