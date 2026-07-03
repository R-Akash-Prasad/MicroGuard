import pandas as pd
import numpy as np
import requests
import json
from warnings import filterwarnings
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier

filterwarnings('ignore')

# G20 Country codes
G20_CODES = ["AR", "AU", "BR", "CA", "CN", "FR", "DE", "IN", "ID", "IT",
             "JP", "MX", "RU", "SA", "ZA", "KR", "TR", "GB", "US", "EU"]

# Our indicators mapping (WB API codes)
INDICATORS = {
    'Domestic credit to private sector (% of GDP)': 'GFDEBTN',
    'Domestic credit to private sector by banks (% of GDP)': 'GFDEBTN',
    'Monetary Sector credit to private sector (% GDP)': 'MONEYL',
    'Market capitalization of listed domestic companies (% of GDP)': 'GFDXIND',
    'Stocks traded, total value (% of GDP)': 'GFDXVTN',
    'Stocks traded, turnover ratio of domestic shares (%)': 'GFDXSTO',
    'Foreign direct investment, net inflows (BoP, current US$)': 'BX.KLT.DINV.CD.WD',
    'Foreign direct investment, net outflows (% of GDP)': 'CI.XOV.UND.CD',
    'Portfolio equity, net inflows (BoP, current US$)': 'BX.PEF.TOTL.CD.WD'
}

def fetch_wb_data():
    print("Fetching data from World Bank API...")
    all_data = []
    
    for code, indicator in INDICATORS.items():
        print(f"  Fetching: {code}...")
        for country in G20_CODES:
            try:
                url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&date=2000:2024&per_page=500"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        for record in data[1]:
                            all_data.append({
                                'INDICATOR_LABEL': code,
                                'REF_AREA': country,
                                'TIME_PERIOD': int(record['date']),
                                'OBS_VALUE': record['value']
                            })
            except Exception as e:
                print(f"    Error fetching {country} {code}: {e}")
    
    df = pd.DataFrame(all_data)
    print(f"Fetched {len(df)} records from API")
    return df

def update_final_data(wb_data):
    print("\nMerging with existing final_data.csv...")
    existing = pd.read_csv('final_data.csv')
    
    # Pivot WB data
    wb_pivot = wb_data.pivot_table(
        index=['REF_AREA', 'TIME_PERIOD'],
        columns='INDICATOR_LABEL',
        values='OBS_VALUE'
    ).reset_index()
    
    # Map country codes to match existing
    code_map = {
        'AR': 'ARG', 'AU': 'AUS', 'BR': 'BRA', 'CA': 'CAN', 'CN': 'CHN',
        'FR': 'FRA', 'DE': 'DEU', 'IN': 'IND', 'ID': 'IDN', 'IT': 'ITA',
        'JP': 'JPN', 'MX': 'MEX', 'RU': 'RUS', 'SA': 'SAU', 'ZA': 'ZAF',
        'KR': 'KOR', 'TR': 'TUR', 'GB': 'GBR', 'US': 'USA'
    }
    wb_pivot['REF_AREA'] = wb_pivot['REF_AREA'].map(code_map)
    
    # Filter existing for years > 2000 and G20
    existing_filtered = existing[existing['TIME_PERIOD'] > 2000]
    
    # Combine
    combined = pd.concat([existing_filtered, wb_pivot], ignore_index=True)
    combined = combined.drop_duplicates(subset=['REF_AREA', 'TIME_PERIOD'], keep='last')
    combined = combined.sort_values(['REF_AREA', 'TIME_PERIOD'])
    
    # Add EU back (it's aggregated)
    eu_data = existing[existing['REF_AREA'] == 'EUU']
    
    final = pd.concat([combined, eu_data], ignore_index=True)
    final = final.sort_values(['REF_AREA', 'TIME_PERIOD'])
    
    print(f"Combined dataset has {len(final)} records")
    return final

def process_features(df):
    print("\nProcessing features...")
    indicators = df.columns.difference(['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'])
    lamb = 100
    
    # HP Filter
    for col in indicators:
        def get_gap(group):
            if len(group) < 2:
                return pd.Series(0, index=group.index)
            cycle, _ = hpfilter(group, lamb=lamb)
            return cycle
        df[f'{col}_Gap'] = df.groupby('REF_AREA')[col].transform(get_gap)
    
    # Target
    credit_col = 'Domestic credit to private sector by banks (% of GDP)_Gap'
    df['target_credit_gap2'] = df.groupby('REF_AREA')[credit_col].shift(-2)
    df = df.dropna(subset=['target_credit_gap2'])
    
    # Winsorize
    lower = df['target_credit_gap2'].quantile(0.01)
    upper = df['target_credit_gap2'].quantile(0.99)
    df['target_credit_gap2'] = df['target_credit_gap2'].clip(lower, upper)
    
    # Lags
    gap_cols = [col for col in df.columns if '_Gap' in col]
    for col in gap_cols:
        df[f'{col}_Lag1'] = df.groupby('REF_AREA')[col].shift(1)
    
    df = df.groupby('REF_AREA').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    # Scale
    scaler = StandardScaler()
    metadata = ['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD', 'target_credit_gap2']
    feature_cols = df.columns.difference(metadata)
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols

def retrain_model(df, feature_cols):
    print("\nRetraining model...")
    CRISIS_THRESHOLD = -2
    df['crisis_label'] = (df['target_credit_gap2'] < CRISIS_THRESHOLD).astype(int)
    
    train_df = df[df['TIME_PERIOD'] <= 2021]
    test_df = df[df['TIME_PERIOD'] > 2021]
    
    X_train, y_train = train_df[feature_cols], train_df['crisis_label']
    X_test, y_test = test_df[feature_cols], test_df['crisis_label']
    
    scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
    
    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric='logloss', use_label_encoder=False
        )
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba) if len(y_test[y_test==1]) > 0 else 0.5
    
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test set size: {len(test_df)}")
    print(f"Latest year in test: {test_df['TIME_PERIOD'].max()}")
    
    return model, scaler

def main():
    print("=" * 50)
    print("MacroGuard: Updating Dataset from World Bank API")
    print("=" * 50)
    
    try:
        wb_data = fetch_wb_data()
        df = update_final_data(wb_data)
    except Exception as e:
        print(f"API fetch failed: {e}")
        print("Using existing final_data.csv instead...")
        df = pd.read_csv('final_data.csv')
    
    df, feature_cols = process_features(df)
    df.to_csv('processed_features.csv', index=False)
    print(f"\nSaved processed_features.csv with {len(df)} records")
    
    # Show latest years
    print(f"\nLatest years available per country:")
    latest = df.groupby('REF_AREA')['TIME_PERIOD'].max().sort_values(ascending=False)
    print(latest.head(10))
    
    model, scaler = retrain_model(df, feature_cols)
    
    # Save artifacts
    import joblib
    joblib.dump(model, 'macroguard_model.pkl')
    joblib.dump(scaler, 'macroguard_scaler.pkl')
    joblib.dump(list(feature_cols), 'feature_columns.pkl')
    
    print("\nModel and artifacts updated!")

if __name__ == "__main__":
    main()
