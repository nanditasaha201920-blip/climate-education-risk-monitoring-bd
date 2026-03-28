import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SafeSchool-BD | Risk Monitor", layout="wide", page_icon="🏫")

# --- CUSTOM CSS FOR NATIONAL DASHBOARD LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- TITLE & HEADER ---
st.title("🏛️ SafeSchool-BD: National Climate-Education Risk Framework")
st.markdown("### *Decision Support System for Smart Education Planning*")
st.divider()

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    # আপনার দেওয়া CSV ফাইলটি লোড করা হচ্ছে
    try:
        df = pd.read_csv("bangladesh_integrated_data.csv")
        
        # ১. Climate Score (Normalized 0-1)
        df["climate_score"] = (df["flood_risk"] + df["cyclone_risk"] + df["salinity_risk"]) / 3
        
        # ২. Education Risk (Normalized 0-1)
        df["education_risk"] = ((1 - df["attendance_rate"]) + df["dropout_rate"] + (df["school_closure_days"] / 60)) / 3
        
        # ৩. Socioeconomic Risk (Normalized 0-1)
        df["socio_risk"] = (df["poverty_rate"] + df["child_labor_rate"] + df["displacement_rate"]) / 3
        
        # ৪. Combined National Risk Score (Weighted)
        df["total_risk_score"] = (0.4 * df["climate_score"] + 0.3 * df["education_risk"] + 0.3 * df["socio_risk"])
        
        return df
    except FileNotFoundError:
        st.error("Error: 'bangladesh_integrated_data.csv' not found!")
        return None

df = load_data()

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("📍 Region Selection")
    divisions = st.sidebar.multiselect("Select Division(s):", options=df["division"].unique(), default=df["division"].unique())
    filtered_df = df[df["division"].isin(divisions)]

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Risk District", df.loc[df['total_risk_score'].idxmax(), 'district'])
    m2.metric("Avg National Risk", f"{df['total_risk_score'].mean():.2f}")
    m3.metric("Max Dropout Rate", f"{df['dropout_rate'].max():.1%}")
    m4.metric("Avg School Closure", f"{int(df['school_closure_days'].mean())} Days")

    # --- VISUALIZATION SECTION ---
    col1, col2 = st.columns([6, 4])

    with col1:
        st.subheader("📊 Comparative Risk Analysis (District-wise)")
        fig_bar = px.bar(filtered_df.sort_values("total_risk_score", ascending=False), 
                         x="district", y="total_risk_score", color="total_risk_score",
                         color_continuous_scale="RdYlGn_r", labels={'total_risk_score':'Risk Level'},
                         hover_data=["climate_score", "education_risk"])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🎯 Risk Correlation Matrix")
        fig_scatter = px.scatter(filtered_df, x="climate_score", y="education_risk", 
                                 size="poverty_rate", color="division", hover_name="district",
                                 trendline="ols", title="Climate vs Education Impact")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- MACHINE LEARNING PREDICTION ---
    st.divider()
    st.subheader("🤖 AI Predictive Insight (Policy Simulator)")
    
    # Train a simple model to predict Dropout based on Climate Risk
    X = df[['climate_score']].values
    y = df['dropout_rate'].values
    model = LinearRegression().fit(X, y)

    c1, c2 = st.columns([1, 2])
    with c1:
        sim_climate = st.slider("Simulate Future Climate Risk Level:", 0.0, 1.0, 0.5)
        pred_dropout = model.predict(np.array([[sim_climate]]))[0]
        st.info(f"**Predicted Dropout Rate:** {pred_dropout:.2%}")
    
    with c2:
        st.write("💡 **Policy Suggestion:**")
        if pred_dropout > 0.30:
            st.error("HIGH ALERT: Urgent need for 'Climate-Resilient Schools' and financial grants in this scenario.")
        else:
            st.success("STABLE: Focus on digital education and standard infrastructure maintenance.")

    # --- DATA TABLE ---
    with st.expander("📥 View Full National Dataset"):
        st.dataframe(filtered_df.style.background_gradient(cmap='YlOrRd', subset=['total_risk_score']), use_container_width=True)

    st.caption("Developed for National Policy Planning | Data Source: IPCC/BBS Integrated Reports")
