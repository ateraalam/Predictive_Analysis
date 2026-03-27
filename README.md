# Predicting Future Sales Patterns: A Data Mining Project

> Exploring how machine learning can forecast purchase quantities using real e-commerce transaction data.

---

## What This Project Is About

This project digs into a real-world e-commerce dataset and asks: **given what we know about a product's price, total transaction value, and where the customer is located — can we predict how much of it they'll buy?**

The dataset comes from the [UCI Machine Learning Repository's Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail), which contains over 540,000 actual transactions from a UK-based online retailer between December 2010 and December 2011. The company mainly sells unique all-occasion gifts, and a good chunk of their customers are wholesalers.

The notebook walks through the full data science lifecycle — from raw data all the way to trained models — comparing Linear Regression against a Decision Tree Regressor to see which one actually holds up.

**Spoiler:** the Decision Tree wins by a landslide (R² of 0.85 vs 0.25).

---

## Why This Matters

Predicting purchase quantities might sound simple, but it has real implications. If a retailer knows how much of a product a customer is likely to buy, they can plan inventory smarter, run better promotions, and avoid both stockouts and overstock situations. This project is a hands-on exploration of how data mining techniques can be applied to that kind of problem using real transactional data.

---

## What the Notebook Covers

📓 **`Data Mining Project Predicting Future Sales.ipynb`**

### 1. Data Loading
The dataset is pulled directly from the UCI repository — no manual download needed. It contains 541,909 transactions across 8 columns: invoice number, stock code, item description, quantity, invoice date, unit price, customer ID, and country.

### 2. Sampling
Since the full dataset is massive, the project samples 10,000 entries (with a fixed random state for reproducibility) to keep things manageable.

### 3. Data Cleaning & Preprocessing
This is where the messy parts get handled:
- Dropping rows with missing values (especially CustomerID, which has a lot of gaps)
- Removing negative quantities (returns/cancellations) and zero-price entries
- Ensuring datetime consistency on the InvoiceDate column
- After cleaning: **7,268 usable records**

### 4. Exploratory Data Analysis (EDA)
The notebook visualizes the data to understand what we're working with:
- **Unit Price distribution** — most items are low-priced, with a long right tail
- **Quantity distribution** — most purchases are small, but some bulk orders skew things
- **Top 10 countries by transactions** — the UK dominates (this is a UK retailer after all), with Germany, France, and EIRE trailing behind
- **Monthly sales over time** — clear spike in October/November, likely driven by holiday shopping season

### 5. Outlier Removal
The raw distributions were heavily skewed, making the visualizations hard to read. The notebook uses the **IQR method** (interquartile range) to filter out extreme values in both UnitPrice and Quantity, then re-plots for much cleaner visuals.

### 6. Feature Engineering & Selection
- **New feature created:** `TotalPrice` = Quantity × UnitPrice (captures the full transaction value)
- **Categorical encoding:** Country names are label-encoded into numeric values
- **Feature selection:** The model uses `UnitPrice`, `TotalPrice`, and `Country` as inputs to predict `Quantity`
- **Standardization:** Features are scaled using `StandardScaler` so no single variable dominates due to its range

### 7. KMeans Clustering
Before jumping into prediction, the notebook runs KMeans clustering to segment customers:
- **Elbow method** is used to find the optimal cluster count → **4 clusters**
- The clusters reveal distinct purchasing behaviors:
  - **Low price, low quantity** buyers (budget shoppers)
  - **Higher price, lower quantity** buyers (selective spenders)
  - **Higher value items in larger quantities** (premium bulk buyers)
  - **Low price, high quantity** buyers (wholesale/bulk discount shoppers)

### 8. Model 1 — Linear Regression
- 80/20 train-test split
- 5-fold cross-validation performed
- **Results:** MSE = 2,642 | R² = 0.25
- The actual-vs-predicted scatter plot shows heavy misalignment — Linear Regression just doesn't capture the non-linear relationships in this data

### 9. Model 2 — Decision Tree Regressor
- Same 80/20 split
- **Hyperparameter tuning** via GridSearchCV across max_depth, min_samples_split, and min_samples_leaf
- Best parameters found: `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=2`
- **Results:** MSE = 523 | R² = 0.85
- The actual-vs-predicted plot shows strong alignment along the diagonal — this model is performing well

---

## Results at a Glance

| Model | MSE | R² Score | Verdict |
|-------|-----|----------|---------|
| Linear Regression | 2,642 | 0.25 | Poor — doesn't capture the patterns |
| Decision Tree Regressor | 523 | 0.85 | Strong — reliable predictions |

The Decision Tree Regressor outperforms Linear Regression by a wide margin, reducing error by ~80% and explaining 85% of the variance in purchase quantities.

---

## The Dataset

**Source:** [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)

| Column | Description |
|--------|-------------|
| InvoiceNo | Unique invoice number (prefix 'C' indicates a cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Number of units per transaction |
| InvoiceDate | Date and time of the transaction |
| UnitPrice | Price per unit in GBP (£) |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

The dataset is loaded directly from the UCI repository URL in the notebook, so no manual download is needed.

---

## How to Run This

### Prerequisites

Python 3.7+ with the standard data science libraries:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn openpyxl
```

> **Note:** `openpyxl` is needed because the source data is an Excel file (.xlsx).

### Steps

1. **Clone the repo**
   ```bash
   git clone https://github.com/ateraalam/Predictive_Analysis.git
   cd Predictive_Analysis
   ```

2. **Launch the notebook**
   ```bash
   jupyter notebook "Data Mining Project Predicting Future Sales.ipynb"
   ```

3. **Run all cells** from top to bottom. The data loads directly from the UCI repository, so you just need an internet connection.

---

## Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data wrangling and numerical operations
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — preprocessing (LabelEncoder, StandardScaler), clustering (KMeans), models (LinearRegression, DecisionTreeRegressor), evaluation (MSE, R²), hyperparameter tuning (GridSearchCV, cross_val_score)

---

## Key Takeaways

- **Linear Regression isn't always the answer.** It's tempting to start with it because it's simple, but when the underlying relationships are non-linear (as they are here), it'll underperform significantly.
- **Decision Trees handle messy, non-linear data well.** With proper hyperparameter tuning, the Decision Tree Regressor captured patterns that Linear Regression completely missed.
- **Outlier removal makes a real difference.** The raw data was heavily skewed by a handful of extreme transactions. Cleaning those out with IQR filtering made both the visualizations and the models more meaningful.
- **Clustering adds context.** Even though the primary goal was prediction, the KMeans clustering step revealed distinct customer segments that could inform business strategy beyond just forecasting.

---

## Limitations

- Only two models were tested. There's a whole world of algorithms (Random Forest, XGBoost, Gradient Boosting) that could potentially do even better.
- The project uses a 10,000-entry sample from a 540K+ record dataset. Training on the full dataset might yield different (and possibly better) results.
- Feature engineering was kept relatively simple. There's room to extract more signal from the data (time-based features, customer purchase history, product category groupings, etc.).

---

## Ideas for Future Work

- Try ensemble methods like **Random Forest** or **Gradient Boosting** to see if they outperform the single Decision Tree
- Incorporate **time-series features** (month, day of week, days until Christmas) to capture seasonal patterns
- Train on the **full dataset** instead of a sample
- Experiment with **deep learning** approaches (neural networks) for comparison
- Build a proper **customer segmentation pipeline** on top of the clustering work
- Add **cross-validation to the Decision Tree** evaluation for a more robust performance estimate

---

## Project Structure

```
Predictive_Analysis/
├── Data Mining Project Predicting Future Sales.ipynb   # The full analysis notebook
└── README.md                                           # This file
```

---

## Author

**ateraalam** — [GitHub Profile](https://github.com/ateraalam)

---

*Built as a data mining course project. If you found this helpful or have feedback, feel free to open an issue or reach out!*
