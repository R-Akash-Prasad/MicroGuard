import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, 
                             precision_recall_curve)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

filterwarnings('ignore')

def load_and_preprocess_data(input_csv='WB_FSI.csv'):
    """
    Loads raw World Bank FSI data and pivots it into a country-year format.
    """
    print(f"Loading data from {input_csv}...")
    data = pd.read_csv(input_csv)
    
    # Pivot the dataset
    print("Pivoting dataset...")
    df = data.pivot(
        index=['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'],
        columns='INDICATOR_LABEL',
        values='OBS_VALUE'
    ).reset_index()
    
    # Filter for G20 countries
    countries = ["ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN", "ITA",
                 "JPN", "MEX", "RUS", "SAU", "ZAF", "KOR", "TUR", "GBR", "USA", "EUU"]
    df = df[df['REF_AREA'].isin(countries)].reset_index(drop=True)
    
    # Select meaningful indicators (based on notebook exploration)
    meaningful_indicators = [
        'REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD',
        'Domestic credit to private sector (% of GDP)',
        'Domestic credit to private sector by banks (% of GDP)',
        'Foreign direct investment, net inflows (BoP, current US$)',
        'Foreign direct investment, net outflows (% of GDP)',
        'Market capitalization of listed domestic companies (% of GDP)',
        'Monetary Sector credit to private sector (% GDP)',
        'Portfolio equity, net inflows (BoP, current US$)',
        'Stocks traded, total value (% of GDP)',
        'Stocks traded, turnover ratio of domestic shares (%)'
    ]
    df = df[meaningful_indicators]
    
    # Simple interpolation/filling for missing values
    df = df.groupby('REF_AREA_LABEL').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    df.fillna(0, inplace=True)
    
    return df

def perform_feature_engineering(df):
    """
    Calculates HP Filter gaps, targets, and lags.
    """
    print("Performing feature engineering...")
    indicators = df.columns.difference(['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'])
    lamb = 100 # Annual frequency lambda
    
    # HP Filter Gap Calculation
    for col in indicators:
        def get_gap(group):
            if len(group) < 2:
                return pd.Series(0, index=group.index)
            cycle, _ = hpfilter(group, lamb=lamb)
            return cycle
        df[f'{col}_Gap'] = df.groupby('REF_AREA_LABEL')[col].transform(get_gap)
        
    # Target Variable: 2-year forward Credit Gap
    credit_col = 'Domestic credit to private sector by banks (% of GDP)_Gap'
    df['target_credit_gap2'] = df.groupby('REF_AREA_LABEL')[credit_col].shift(-2)
    df = df.dropna(subset=['target_credit_gap2'])
    
    # Winsorization
    lower_limit = df['target_credit_gap2'].quantile(0.01)
    upper_limit = df['target_credit_gap2'].quantile(0.99)
    df['target_credit_gap2'] = df['target_credit_gap2'].clip(lower_limit, upper_limit)
    
    # 1-Year Lagged Features
    gap_cols = [col for col in df.columns if '_Gap' in col]
    for col in gap_cols:
        df[f'{col}_Lag1'] = df.groupby('REF_AREA_LABEL')[col].shift(1)
        
    # Handle NaNs from lagging
    df = df.groupby('REF_AREA_LABEL').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    # Scaling
    scaler = StandardScaler()
    metadata = ['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD', 'target_credit_gap2']
    feature_cols = df.columns.difference(metadata)
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols

def run_modeling_and_optimization(df, feature_cols):
    """
    Trains models, performs Time Series Cross-Validation, and optimizes thresholds.
    """
    print("Running modeling and optimization...")
    CRISIS_THRESHOLD = -2
    df['crisis_label'] = (df['target_credit_gap2'] < CRISIS_THRESHOLD).astype(int)
    
    # Split by year (e.g., test on 2018)
    train_df = df[df['TIME_PERIOD'] <= 2017]
    test_df = df[df['TIME_PERIOD'] > 2017]
    
    X_train, y_train = train_df[feature_cols], train_df['crisis_label']
    X_test, y_test = test_df[feature_cols], test_df['crisis_label']
    
    if not XGBOOST_AVAILABLE:
        print("XGBoost not found. Using Random Forest only.")
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
    else:
        print("Training XGBoost Classifier...")
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [1, scale_pos_weight]
        }
    
    grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    # Probabilities for testing
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Time Series Cross-Validation
    print("\nStarting Time Series Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    df_sorted = df.sort_values('TIME_PERIOD')
    X_tscv = df_sorted[feature_cols]
    y_tscv = df_sorted['crisis_label']
    
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_tscv)):
        X_tr, X_te = X_tscv.iloc[train_idx], X_tscv.iloc[test_idx]
        y_tr, y_te = y_tscv.iloc[train_idx], y_tscv.iloc[test_idx]
        
        m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                          scale_pos_weight=len(y_tr[y_tr==0])/len(y_tr[y_tr==1]),
                          random_state=42, eval_metric='logloss', use_label_encoder=False) if XGBOOST_AVAILABLE \
            else RandomForestClassifier(class_weight='balanced', random_state=42)
            
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        cv_scores.append(roc_auc_score(y_te, p))
    print(f"Mean TSCV ROC-AUC: {np.mean(cv_scores):.4f}")
    
    # Threshold Optimization
    print("\nOptimizing threshold...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = [2*p*r/(p+r) if (p+r)>0 else 0 for p, r in zip(precisions, recalls)]
    best_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[min(best_idx, len(thresholds)-1)]
    
    print(f"Optimal Threshold: {opt_threshold:.4f}")
    print(f"Recall at Opt Threshold: {recalls[best_idx]:.4f}")
    
    return best_model, opt_threshold

def main():
    print("=== MacroGuard Early Warning System Pipeline ===")
    
    # 1. Preprocess
    df = load_and_preprocess_data()
    
    # 2. Feature Engineering
    df, feature_cols = perform_feature_engineering(df)
    df.to_csv('processed_features.csv', index=False)
    
    # 3. Modeling
    model, opt_threshold = run_modeling_and_optimization(df, feature_cols)
    
    print("\nPipeline execution complete.")
    print("Final Model ready for crisis prediction.")

if __name__ == "__main__":
    main()
