# Precision Agriculture - Data Analysis & Machine Learning

A practical machine learning project for precision agriculture, covering crop recommendation, yield prediction, and data cleaning on real-world agricultural datasets.

## Project Structure

```
crop_analysis/
├── explore.ipynb              # Data analysis and model training
├── app.py                     # Interactive Streamlit dashboard
├── crop_model.pkl             # Trained crop recommendation model
├── Crop_recommendation.csv    # Crop recommendation dataset
├── yield_df.csv               # Raw crop yield dataset
└── yield_clean.csv            # Cleaned crop yield dataset
```

## What This Project Covers

### 1. Crop Recommendation System
- Dataset: 2,200 samples across 22 crop types
- Features: Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall
- Models: Random Forest Classifier (99.3% accuracy) vs XGBoost (98.6% accuracy)
- Output: Interactive Streamlit dashboard for real-time crop recommendation

### 2. Data Cleaning (Real-World Dataset)
- Dataset: 28,242 global crop yield records across multiple countries
- Issues handled: Extreme outliers in pesticide usage and yield values
- Method: IQR-based outlier removal
- Result: Cleaned dataset of 24,785 records (removed 12% outliers)

### 3. Crop Yield Prediction
- Task: Regression (predicting actual yield in hg/ha)
- Model: Random Forest Regressor
- Results: R² = 0.98, MAE = 3,076 hg/ha (~5.3% of mean yield)
- Preprocessing: OneHotEncoding for categorical features (country, crop type)

### 4. Feature Importance Analysis
- Identified crop type (especially Potatoes, Sweet potatoes, Cassava) as the dominant predictor
- Environmental factors: pesticide usage > rainfall > temperature

## Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Classification vs Regression | RandomForestClassifier vs RandomForestRegressor |
| Outlier Detection | IQR method with before/after visualization |
| Categorical Encoding | LabelEncoder vs OneHotEncoder |
| Model Evaluation | Accuracy, MAE, R² Score, Confusion Matrix |
| Feature Importance | Built-in Random Forest feature_importances_ |
| Model Persistence | joblib for saving/loading trained models |

## How to Run

### Install dependencies
```bash
pip install pandas matplotlib scikit-learn streamlit xgboost joblib
```

### Run the interactive dashboard
```bash
streamlit run app.py
```

### Open the analysis notebook
Open `explore.ipynb` in PyCharm or Jupyter Notebook and run cells sequentially.

## Datasets

- **Crop Recommendation Dataset** – [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Crop Yield Prediction Dataset** – [Kaggle](https://www.kaggle.com/datasets/patelris603/crop-yield-prediction-dataset)

## Environment

- Python 3.10+
- pandas, scikit-learn, xgboost, matplotlib, streamlit, joblib
