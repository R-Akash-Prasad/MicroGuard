import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.filters.hp_filter import hpfilter
from warnings import filterwarnings

filterwarnings('ignore')

def process_and_predict():
    print("=" * 60)
    print("MacroGuard: Generating 2025 Crisis Predictions")
    print("=" * 60)
    
    print("\n1. Loading model artifacts...")
    model = joblib.load('macroguard_model.pkl')
    scaler = joblib.load('macroguard_scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    
    print("2. Loading final_data.csv...")
    df = pd.read_csv('final_data.csv')
    
    print("3. Processing data with HP filters...")
    indicators = df.columns.difference(['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'])
    lamb = 100
    
    for col in indicators:
        def get_gap(group):
            if len(group) < 2:
                return pd.Series(0, index=group.index)
            cycle, _ = hpfilter(group, lamb=lamb)
            return cycle
        df[f'{col}_Gap'] = df.groupby('REF_AREA')[col].transform(get_gap)
    
    gap_cols = [col for col in df.columns if '_Gap' in col]
    
    for col in gap_cols:
        df[f'{col}_Lag1'] = df.groupby('REF_AREA')[col].shift(1)
    
    df = df.groupby('REF_AREA').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    print("4. Generating predictions for 2025...")
    country_list = df['REF_AREA'].unique()
    predictions = []
    
    for country in country_list:
        country_df = df[df['REF_AREA'] == country].sort_values('TIME_PERIOD')
        
        if len(country_df) < 2:
            continue
        
        row_2023 = country_df[country_df['TIME_PERIOD'] == 2023]
        row_2022 = country_df[country_df['TIME_PERIOD'] == 2022]
        
        if len(row_2023) == 0 or len(row_2022) == 0:
            row_2023 = country_df.iloc[-1]
            row_2022 = country_df.iloc[-2] if len(country_df) > 1 else country_df.iloc[-1]
        else:
            row_2023 = row_2023.iloc[0]
            row_2022 = row_2022.iloc[0]
        
        features = {}
        for col in gap_cols:
            features[col] = row_2023[col] if not pd.isna(row_2023[col]) else 0
            features[f'{col}_Lag1'] = row_2022[col] if not pd.isna(row_2022[col]) else 0
        
        feature_df = pd.DataFrame([features])
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[feature_cols]
        
        feature_scaled = scaler.transform(feature_df)
        prob = model.predict_proba(feature_scaled)[0][1]
        
        predictions.append({
            'Country_Code': country,
            'Country_Name': row_2023['REF_AREA_LABEL'],
            'Crisis_Probability_2025': round(prob * 100, 2),
            'Risk_Level': 'High' if prob > 0.5 else ('Medium' if prob > 0.3 else 'Low')
        })
    
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('Crisis_Probability_2025', ascending=False)
    
    print("\n" + "=" * 60)
    print("2025 CRISIS PROBABILITY PREDICTIONS FOR G20 COUNTRIES")
    print("=" * 60)
    print(f"\nModel uses 2023 data to predict 2025 crisis probability")
    print(f"Crisis defined as: Credit-to-GDP Gap < -2")
    print(f"\nHigh Risk:   Probability > 50%")
    print(f"Medium Risk: Probability 30% - 50%")
    print(f"Low Risk:    Probability < 30%")
    print()
    print(results_df.to_string(index=False))
    
    results_df.to_csv('crisis_predictions_2025.csv', index=False)
    print(f"\nSaved to 'crisis_predictions_2025.csv'")
    
    print("\n5. Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    ax1 = axes[0]
    colors = results_df['Risk_Level'].map({'High': 'red', 'Medium': 'orange', 'Low': 'green'})
    bars = ax1.barh(results_df['Country_Name'], results_df['Crisis_Probability_2025'], color=colors)
    ax1.set_xlabel('Crisis Probability (%)', fontsize=12)
    ax1.set_title('G20 Countries: 2025 Crisis Probability Prediction', fontsize=14)
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='High Risk (50%)')
    ax1.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (30%)')
    ax1.legend()
    ax1.set_xlim(0, 100)
    
    for bar, prob in zip(bars, results_df['Crisis_Probability_2025']):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontsize=9)
    
    ax2 = axes[1]
    risk_counts = results_df['Risk_Level'].value_counts()
    colors_pie = {'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44aa44'}
    ax2.pie([risk_counts.get('High', 0), risk_counts.get('Medium', 0), risk_counts.get('Low', 0)],
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            colors=[colors_pie['High'], colors_pie['Medium'], colors_pie['Low']],
            autopct='%1.0f%%', startangle=90, explode=(0.05, 0, 0))
    ax2.set_title('Risk Distribution Across G20', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('crisis_predictions_2025_chart.png', dpi=150, bbox_inches='tight')
    print("Saved chart to 'crisis_predictions_2025_chart.png'")
    plt.close()
    
    high_risk = results_df[results_df['Risk_Level'] == 'High']
    medium_risk = results_df[results_df['Risk_Level'] == 'Medium']
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal Countries Analyzed: {len(results_df)}")
    print(f"High Risk (2025):    {len(high_risk)} countries")
    print(f"Medium Risk (2025):  {len(medium_risk)} countries")
    print(f"Low Risk (2025):     {len(results_df) - len(high_risk) - len(medium_risk)} countries")
    
    if len(high_risk) > 0:
        print(f"\n🚨 HIGH RISK COUNTRIES (Immediate Attention Required):")
        for _, row in high_risk.iterrows():
            print(f"   - {row['Country_Name']}: {row['Crisis_Probability_2025']}%")
    
    return results_df

if __name__ == "__main__":
    results = process_and_predict()
