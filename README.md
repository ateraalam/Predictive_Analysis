# Predicting Future Sales — A Data Mining Project

> Using machine learning to forecast sales from historical transaction data.

---

## What This Project Is About

This project tackles a classic (and genuinely tricky) problem: **can we look at past sales data and reliably predict what's going to sell in the future?**

The short answer is yes — with the right data wrangling and the right models. This notebook walks through the entire journey, from messy raw data all the way to trained predictive models, step by step.

It's built around the [Kaggle "Predict Future Sales"](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) challenge, which provides daily sales history from a large Russian software company (1C Company). The goal is to predict the total sales for every product in every store for the following month.

---

## Why This Matters

Sales forecasting isn't just an academic exercise. Businesses live and die by their ability to anticipate demand. Get it right and you can optimize inventory, plan marketing spend, and allocate resources intelligently. Get it wrong and you're either sitting on unsold stock or scrambling to fill orders you didn't see coming.

This project explores how data mining and machine learning techniques can be applied to real-world sales data to build models that actually generalize — not just memorize the past, but learn patterns that hold up going forward.

---

## What's Inside

The core of this repo is a single Jupyter notebook:

📓 **`Data Mining Project Predicting Future Sales.ipynb`**

Here's a rough roadmap of what the notebook covers:

### 1. Data Loading & Exploration
- Importing the raw datasets (sales history, item metadata, shop info, etc.)
- Getting a feel for the shape, size, and quirks of the data
- Identifying missing values, outliers, and anything that looks off

### 2. Data Cleaning & Preprocessing
- Handling missing or corrupted entries
- Removing outliers that would throw off the models
- Encoding categorical variables so the algorithms can work with them
- Aggregating daily sales into monthly totals (since the prediction target is monthly)

### 3. Feature Engineering
- Creating lag features (what did sales look like 1, 2, 3 months ago?)
- Rolling averages and trend indicators
- Extracting useful signals from item names, categories, and shop metadata
- Building time-based features (month, year, seasonal indicators)

### 4. Model Building & Training
- Splitting data into training and validation sets
- Training one or more predictive models (likely including XGBoost, which is the go-to for this kind of tabular data)
- Tuning hyperparameters to squeeze out better performance

### 5. Evaluation & Results
- Measuring model performance using appropriate metrics (RMSE is the standard for this competition)
- Comparing how different approaches stack up
- Visualizing predictions vs. actuals to see where the model shines and where it struggles

---

## The Dataset

This project uses data from the [Kaggle Predict Future Sales competition](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data). The key files are:

| File | What It Contains |
|------|-----------------|
| `sales_train.csv` | Daily sales transactions from Jan 2013 to Oct 2015 |
| `items.csv` | Item names and the category each item belongs to |
| `item_categories.csv` | Names of all item categories |
| `shops.csv` | Shop names and identifiers |
| `test.csv` | Shop-item pairs to predict sales for (Nov 2015) |

**Note:** The datasets aren't included in this repo. You'll need to download them from the Kaggle competition page (link above). Drop them in the same directory as the notebook and you're good to go.

---

## How to Run This

### Prerequisites

You'll need Python 3.7+ and the usual data science stack. Here's what to install if you don't already have it:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Steps

1. **Clone this repo**
   ```bash
   git clone https://github.com/ateraalam/Predictive_Analysis.git
   cd Predictive_Analysis
   ```

2. **Download the data** from [Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data) and place the CSV files in the project directory.

3. **Fire up the notebook**
   ```bash
   jupyter notebook "Data Mining Project Predicting Future Sales.ipynb"
   ```

4. **Run the cells** top to bottom. The notebook is designed to be followed sequentially.

---

## Tech Stack

- **Python** — the backbone of the whole thing
- **Pandas & NumPy** — data manipulation and number crunching
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — preprocessing, model evaluation, and baseline models
- **XGBoost** — the heavy-hitter for gradient-boosted predictions

---

## Key Takeaways

A few things that come up when you work through this kind of project:

- **Feature engineering is where the magic happens.** The raw data by itself isn't enough — lag features, rolling statistics, and clever aggregations are what give the model something meaningful to learn from.
- **Outlier handling matters more than you'd think.** A few extreme sales values can completely distort your model if you don't deal with them early.
- **Time series data needs careful validation.** You can't just randomly split your data like you would with a typical ML problem. The training set has to come *before* the validation set chronologically, or you're leaking future information into your model.

---

## Project Structure

```
Predictive_Analysis/
├── Data Mining Project Predicting Future Sales.ipynb   # The main notebook
├── README.md                                           # You're reading it
└── (data files — download separately from Kaggle)
```

---

## Possible Next Steps

If you want to take this further, here are some ideas:

- **Try different models** — LightGBM, CatBoost, or even a simple neural network could be interesting to compare
- **Ensemble methods** — blend predictions from multiple models for potentially better results
- **More aggressive feature engineering** — text features from item names, city extraction from shop names, holiday calendars
- **Deploy as an API** — wrap the trained model in a Flask/FastAPI service so it can make predictions on demand

---

## License

This is a personal/academic project. Feel free to use it as a reference or learning resource.

---

## Author

**ateraalam** — [GitHub Profile](https://github.com/ateraalam)

---

*Built as a data mining course project. If you found this useful or have suggestions, feel free to open an issue or reach out!*
