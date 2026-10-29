# California Housing Prices Prediction

A machine learning project that predicts the **median house value** of California districts from census data, such as income, location, population and housing details.

The project follows a full end-to-end machine learning workflow: exploring the data, preparing it, training and comparing several models, tuning the best one, and evaluating it on a held-out test set.

---

## Dataset

- **Source:** [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices), based on the 1990 California census
- **Size:** 20,640 districts, 10 attributes
- **Target:** `median_house_value`

| Attribute | Description |
|---|---|
| `longitude`, `latitude` | Location of the district |
| `housing_median_age` | Median age of the houses |
| `total_rooms`, `total_bedrooms` | Total rooms and bedrooms in the district |
| `population`, `households` | Number of people and households |
| `median_income` | Median income, in tens of thousands of dollars (3 = about $30,000) |
| `ocean_proximity` | Categorical: `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND` |
| `median_house_value` | Median house value (**target**) |

**Data notes:**
- `total_bedrooms` has 207 missing values.
- `median_house_value` is **capped at $500,000**, and `median_income` is also capped. Districts above the cap are all recorded with the same value.

---

## Project Workflow

### 1. Data Exploration
- Summary statistics and histograms of every numerical attribute
- Geographical scatter plot of the districts, coloured by house value and sized by population
- Correlation analysis and scatter matrix. `median_income` is the attribute most strongly correlated with house value.

### 2. Creating the Test Set
- Compared random sampling with **stratified sampling** based on income categories
- Used `StratifiedShuffleSplit` so that the 20% test set represents the full range of incomes

### 3. Data Preparation
- **Missing values:** filled with the median using `SimpleImputer`
- **Categorical data:** `ocean_proximity` converted with `OneHotEncoder`
- **New features:** a custom transformer (`CombinedAttributesAdder`) adds `rooms_per_household`, `population_per_household` and `bedrooms_per_room`
- **Feature scaling:** `StandardScaler`
- All steps combined into one `ColumnTransformer` pipeline, so the same preparation is applied to training, test and new data

### 4. Model Selection
Three models were compared with **10-fold cross-validation**:

| Model | Cross-validation RMSE | Std. Dev. | Notes |
|---|---:|---:|---|
| Linear Regression | ~$69,104 | ~$2,880 | Underfitting: too simple for the data |
| Decision Tree | ~$71,269 | ~$2,692 | Overfitting: $0 training error but poor on unseen data |
| **Random Forest** | **~$50,436** | **~$2,203** | Best and most consistent model |

### 5. Fine-Tuning
- The Random Forest hyperparameters (`n_estimators`, `max_features`, `bootstrap`) were tuned with `GridSearchCV` using 5-fold cross-validation.

### 6. Feature Importance
The most important features for the final model:

| Feature | Importance |
|---|---:|
| `median_income` | 0.335 |
| `INLAND` | 0.171 |
| `pop_per_hhold` | 0.115 |
| `longitude` | 0.071 |
| `bedrooms_per_room` | 0.071 |
| `latitude` | 0.065 |

Income and location are the strongest predictors of house prices. Some ocean proximity categories (`NEAR BAY`, `ISLAND`) contribute very little.

### 7. Final Evaluation
- **Test set RMSE: ~$48,110**
- The test score is close to the cross-validation score, which shows that the model generalizes well and is not overfitting.
- A **95% confidence interval** on the test RMSE is used to judge how reliable this score is and to compare the model fairly with an existing system.

### 8. Saving the Model
The preprocessing pipeline and the trained model are saved together with `joblib`, so the model can be reloaded and used on raw data without retraining:

```python
import joblib

final_model = Pipeline([
    ('preparation', full_pipeline),
    ('forest', forest_reg),
])
joblib.dump(final_model, 'california_housing_model.pkl')

loaded_model = joblib.load('california_housing_model.pkl')
```

> **Note:** the pipeline uses the custom `CombinedAttributesAdder` class, which must be defined before calling `joblib.load()` in another notebook or script.

---

## Results Summary

- The **Random Forest Regressor** was the best model, reducing the error by about **$20,000** compared with Linear Regression and the Decision Tree.
- On unseen test data, the model's predictions are off by about **$48,000** on average.
- **Median income** is by far the most important predictor, followed by whether a district is **inland**.

## Limitations

- **Price cap:** all houses worth more than $500,000 are recorded as $500,000, so the model cannot predict expensive districts accurately.
- **Remaining error:** an average error of about $48,000 is still large compared with typical district values ($120,000 to $265,000). The model is best used to support human estimates, not replace them.
- **Old data:** the dataset comes from the 1990 census, so the model reflects the patterns of that time, not today's housing market.

## Possible Improvements

- Remove or handle the capped districts, or obtain their real values
- Add new features, such as distance to the coast or to major cities
- Try other models, such as Gradient Boosting
- Drop low-importance features and analyse the model's largest errors

---

## How to Run

1. Clone this repository and download `housing.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn scipy joblib jupyter
   ```
3. Open the notebook:
   ```bash
   jupyter notebook "California Housing Prices Project.ipynb"
   ```
4. In the first code cell, change the file path to where you saved `housing.csv`:
   ```python
   housing_data = pd.read_csv("housing.csv")
   ```
5. Run all cells from top to bottom.

## Tools and Libraries

- **Python**
- **pandas** and **NumPy** for data handling
- **Matplotlib** for visualization
- **Scikit-Learn** for preprocessing, models, cross-validation and grid search
- **SciPy** for the confidence interval
- **joblib** for saving the model

## Acknowledgements

This project follows the end-to-end machine learning project in *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.
