import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.filters.hp_filter import hpfilter
from warnings import filterwarnings

filterwarnings('ignore')

st.set_page_config(
    page_title="MacroGuard: Early Warning System",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('macroguard_model.pkl')
    scaler = joblib.load('macroguard_scaler.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, scaler, features

@st.cache_data
def generate_predictions():
    model, scaler, feature_cols = load_artifacts()
    df = pd.read_csv('final_data.csv')
    
    # Fill missing values within each country group first
    df = df.sort_values(['REF_AREA', 'TIME_PERIOD'])
    df = df.set_index(['REF_AREA', 'TIME_PERIOD']).groupby(level=0).ffill().bfill().reset_index()
    
    indicators = df.columns.difference(['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'])
    lamb = 100
    
    for col in indicators:
        def get_gap(group):
            if len(group) < 2:
                return pd.Series(0, index=group.index)
            try:
                cycle, _ = hpfilter(group, lamb=lamb)
                return cycle
            except:
                return pd.Series(0, index=group.index)
        df[f'{col}_Gap'] = df.groupby('REF_AREA')[col].transform(get_gap)
    
    gap_cols = [col for col in df.columns if '_Gap' in col]
    
    for col in gap_cols:
        df[f'{col}_Lag1'] = df.groupby('REF_AREA')[col].shift(1)
    
    # Fill missing values in newly created gap columns
    df = df.set_index(['REF_AREA', 'TIME_PERIOD']).groupby(level=0).ffill().bfill().reset_index()
    
    country_codes_map = {
        'ARG': 'Argentina', 'AUS': 'Australia', 'BRA': 'Brazil', 'CAN': 'Canada',
        'CHN': 'China', 'FRA': 'France', 'DEU': 'Germany', 'IND': 'India',
        'IDN': 'Indonesia', 'ITA': 'Italy', 'JPN': 'Japan', 'MEX': 'Mexico',
        'RUS': 'Russia', 'SAU': 'Saudi Arabia', 'ZAF': 'South Africa',
        'KOR': 'South Korea', 'TUR': 'Turkey', 'GBR': 'United Kingdom',
        'USA': 'United States of America', 'EUU': 'European Union'
    }
    
    predictions = []
    for country in df['REF_AREA'].unique():
        country_df = df[df['REF_AREA'] == country].sort_values('TIME_PERIOD')
        if len(country_df) < 2:
            continue
        
        row_2023 = country_df[country_df['TIME_PERIOD'] == 2023]
        row_2022 = country_df[country_df['TIME_PERIOD'] == 2022]
        
        if len(row_2023) == 0 or len(row_2022) == 0:
            continue
        
        row_2023 = row_2023.iloc[-1]
        row_2022 = row_2022.iloc[-1]
        
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
        
        country_name = country_codes_map.get(country, country)
        
        predictions.append({
            'REF_AREA': country,
            'Country': country_name,
            'Crisis_Probability': round(prob * 100, 1),
            'Risk_Level': 'High' if prob > 0.5 else ('Medium' if prob > 0.3 else 'Low')
        })
    
    return pd.DataFrame(predictions)

try:
    model, scaler, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

st.title("MacroGuard: Financial Stability Early Warning System")
st.markdown("""
This dashboard predicts the probability of a **Financial Crisis** (Credit-to-GDP Gap < -2) 
for G20 countries using machine learning (XGBoost), based on 2023 data to predict 2025.
""")

predictions_df = generate_predictions()
predictions_df = predictions_df.sort_values('Crisis_Probability', ascending=False)

st.divider()

tab1, tab2 = st.tabs(["World Map", "Country Details"])

with tab1:
    st.subheader("Global Crisis Risk Map - 2025 Prediction")
    
    color_map = {'High': '#ff4444', 'Medium': '#ffaa44', 'Low': '#44aa44'}
    
    fig_map = px.choropleth(
        predictions_df,
        locations='Country',
        locationmode='country names',
        color='Crisis_Probability',
        hover_name='Country',
        hover_data={'Crisis_Probability': True, 'Risk_Level': True, 'REF_AREA': False},
        color_continuous_scale='RdYlGn_r', # Red-Yellow-Green reversed
        range_color=[0, 100],
        title='G20 Countries: Crisis Probability (%) for 2025'
    )
    
    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth' # Clean 2D view
        ),
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Probability %")
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        high_count = len(predictions_df[predictions_df['Risk_Level'] == 'High'])
        st.metric("High Risk Countries", high_count)
    with col2:
        med_count = len(predictions_df[predictions_df['Risk_Level'] == 'Medium'])
        st.metric("Medium Risk Countries", med_count)
    with col3:
        low_count = len(predictions_df[predictions_df['Risk_Level'] == 'Low'])
        st.metric("Low Risk Countries", low_count)
    
    st.subheader("Risk Summary Table")
    summary_table = predictions_df[['Country', 'Crisis_Probability', 'Risk_Level']].sort_values('Country')
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

with tab2:
    st.sidebar.title("Select Country")
    countries = sorted(predictions_df['Country'].unique())
    selected_country = st.sidebar.selectbox("Choose a Country", countries)
    
    country_data = predictions_df[predictions_df['Country'] == selected_country].iloc[0]
    prob = country_data['Crisis_Probability'] / 100
    risk_level = country_data['Risk_Level']
    risk_color = "red" if risk_level == "High" else ("orange" if risk_level == "Medium" else "green")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Crisis Probability (2025)", f"{country_data['Crisis_Probability']:.1f}%")
    with col2:
        st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
    
    full_df = pd.read_csv('final_data.csv')
    full_df_g20 = full_df[full_df['REF_AREA'].isin(['ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'IDN', 'ITA', 'JPN', 'MEX', 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA'])]
    
    country_codes_inv = {
        'Argentina': 'ARG', 'Australia': 'AUS', 'Brazil': 'BRA', 'Canada': 'CAN',
        'China': 'CHN', 'France': 'FRA', 'Germany': 'DEU', 'India': 'IND',
        'Indonesia': 'IDN', 'Italy': 'ITA', 'Japan': 'JPN', 'Mexico': 'MEX',
        'Russia': 'RUS', 'Saudi Arabia': 'SAU', 'South Africa': 'ZAF',
        'South Korea': 'KOR', 'Turkey': 'TUR', 'United Kingdom': 'GBR',
        'United States of America': 'USA', 'European Union': 'EUU'
    }
    
    code = country_codes_inv.get(selected_country, selected_country)
    country_df = full_df_g20[full_df_g20['REF_AREA'] == code].sort_values('TIME_PERIOD')
    
    indicators = country_df.columns.difference(['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD'])
    lamb = 100
    
    for col in indicators:
        def get_gap(group):
            if len(group) < 2:
                return pd.Series(0, index=group.index)
            cycle, _ = hpfilter(group, lamb=lamb)
            return cycle
        country_df[f'{col}_Gap'] = country_df.groupby('REF_AREA')[col].transform(get_gap)
    
    gap_col = 'Domestic credit to private sector by banks (% of GDP)_Gap'
    
    st.subheader(f"Historical Credit-to-GDP Gap: {selected_country}")
    fig_line = px.line(
        country_df, 
        x='TIME_PERIOD', 
        y=gap_col,
        title="Credit Gap Trend (HP Filtered)",
        labels={'TIME_PERIOD': 'Year', gap_col: 'Credit Gap'}
    )
    fig_line.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Crisis Threshold (-2)")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("Top 10 Model Features")
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig_imp = px.bar(
            importances, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Top 10 Model Features (Highest on Top)",
            color='Importance', 
            color_continuous_scale='Reds'
        )
        fig_imp.update_yaxes(autorange="reversed") # Ensure highest importance is at the top
        st.plotly_chart(fig_imp, use_container_width=True)

st.sidebar.divider()
st.sidebar.info("""
**Model Info:**
- Algorithm: XGBoost Classifier
- Target: 2-Year Forward Credit Gap
- Data Used: 2023 data to predict 2025
- Optimized Threshold: 0.5679
""")
