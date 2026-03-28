import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# ১. পেজ কনফিগারেশন (সবার আগে থাকতে হবে)
st.set_page_config(page_title="CERMS-BD Dashboard", layout="wide", page_icon="🌏")

# ২. টাইটেল ও হেডার
st.title("🌏 Climate-Education Risk Monitoring System (CERMS-BD)")
st.markdown("### 📊 A Data-Driven Early Warning Dashboard for Bangladesh")

# ৩. ডাটা লোডিং ও প্রসেসিং ফাংশন
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv("bangladesh_integrated_data.csv")
        
        # রিস্ক স্কোর ক্যালকুলেশন (Normalization সহ)
        df["climate_score"] = (df["flood_risk"] + df["cyclone_risk"] + df["salinity_risk"]) / 3
        df["education_risk"] = ((1 - df["attendance_rate"]) + df["dropout_rate"] + (df["school_closure_days"] / 60)) / 3
        df["socio_risk"] = (df["poverty_rate"] + df["child_labor_rate"] + df["displacement_rate"]) / 3
        
        # ফাইনাল ন্যাশনাল রিস্ক স্কোর (Weighted)
        df["total_risk_score"] = (0.4 * df["climate_score"] + 0.3 * df["education_risk"] + 0.3 * df["socio_risk"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_and_process_data()

if df is not None:
    # ৪. টপ মেট্রিক্স (Metric Cards)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📍 Top Risk District", df.loc[df["total_risk_score"].idxmax(), "district"])
    col2.metric("📊 Avg National Risk", round(df["total_risk_score"].mean(), 2))
    col3.metric("🎓 Max Dropout Rate", f"{round(df['dropout_rate'].max()*100,1)}%")
    col4.metric("🏫 Avg School Closure", f"{int(df['school_closure_days'].mean())} Days")

    # ৫. সাইডবার ফিল্টার
    st.sidebar.header("📍 Region Selection")
    selected_division = st.sidebar.multiselect(
        "Select Division(s):",
        options=df["division"].unique(),
        default=df["division"].unique()
    )
    filtered_df = df[df["division"].isin(selected_division)]

    # ৬. ভিজুয়ালাইজেশন - রো ১ (Bar Chart)
    st.markdown("---")
    fig_bar = px.bar(
        filtered_df.sort_values("total_risk_score", ascending=False),
        x="district",
        y="total_risk_score",
        color="total_risk_score",
        color_continuous_scale="RdYlGn_r",
        title="📊 District-wise Risk Levels (Higher is More Vulnerable)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ৭. ভিজুয়ালাইজেশন - রো ২ (Scatter Plot)
    fig_scatter = px.scatter(
        filtered_df,
        x="climate_score",
        y="education_risk",
        size="poverty_rate",
        color="division",
        hover_name="district",
        trendline="ols",
        title="🎯 Relationship: Climate Impact vs. Education Risk"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ৮. এআই প্রেডিকশন (বিপ্লবী ফিচার)
    st.subheader("🤖 Policy Simulator (Machine Learning)")
    X = df[['climate_score']].values
    y = df['dropout_rate'].values
    model = LinearRegression().fit(X, y)

    sim_col, text_col = st.columns([1, 2])
    with sim_col:
        input_val = st.slider("Simulate Climate Risk Rise:", 0.0, 1.0, 0.5)
        prediction = model.predict([[input_val]])[0]
        st.warning(f"Predicted Dropout: {prediction:.1%}")

    with text_col:
        st.info("💡 **AI Insight:** এই সিমুলেশনটি নীতিনির্ধারকদের আগাম সিদ্ধান্ত নিতে সাহায্য করবে। যদি জলবায়ু ঝুঁকি বাড়ে, তবে কোন জেলায় শিক্ষার ওপর প্রভাব বেশি পড়বে তা এটি থেকে বোঝা যায়।")

    # ৯. ডাটা টেবিল
    st.subheader("📋 Detailed Risk Data")
    st.dataframe(filtered_df.sort_values("total_risk_score", ascending=False), use_container_width=True)

    # ১০. ফুটার
    st.markdown("---")
    st.markdown("🔍 *This dashboard is a prototype for a national Climate-Education Risk Monitoring System integrating climate, education, and socioeconomic data.*")

else:
    st.warning("Please ensure 'bangladesh_integrated_data.csv' is in the same folder.")
