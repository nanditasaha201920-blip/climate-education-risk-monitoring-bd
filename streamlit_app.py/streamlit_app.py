import pandas as pd

# ডেটা লোড (আপনার দেওয়া CSV ফরম্যাট অনুযায়ী)
df = pd.read_csv("bangladesh_integrated_data.csv")

# ১. Climate Score (০ থেকে ১ এর মধ্যে গড়)
df["climate_score"] = (df["flood_risk"] + df["cyclone_risk"] + df["salinity_risk"]) / 3

# ২. Education Risk (Normalization: সব প্যারামিটারকে ০-১ এর স্কেলে আনা)
# attendance_rate যেহেতু শতাংশে (০.৬৫), তাই (১ - attendance) দিলে অনুপস্থিতির হার পাওয়া যায়
df["education_risk"] = (
    (1 - df["attendance_rate"]) + 
    df["dropout_rate"] + 
    (df["school_closure_days"] / 60)  # ৬০ দিনকে সর্বোচ্চ ধরে ১-এর স্কেলে আনা
) / 3

# ৩. Socioeconomic Risk (০ থেকে ১ এর মধ্যে গড়)
df["socio_risk"] = (df["poverty_rate"] + df["child_labor_rate"] + df["displacement_rate"]) / 3

# ৪. Final Score (Weighted Average)
df["total_risk_score"] = (
    0.4 * df["climate_score"] +
    0.3 * df["education_risk"] + 
    0.3 * df["socio_risk"]
)

# ৫. Risk Classification (স্কোর ডিস্ট্রিবিউশন অনুযায়ী থ্রেশহোল্ড সমন্বয়)
def classify(score):
    if score >= 0.60:    # বাস্তবসম্মত ডেটায় ০.৭৫ পাওয়া কঠিন হতে পারে, তাই ০.৬০ দেওয়া হয়েছে
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_level"] = df["total_risk_score"].apply(classify)

# ৬. Alerts Generation
df["alert"] = df.apply(lambda x:
    f"⚠ {x['district']} HIGH RISK" if x["risk_level"]=="HIGH"
    else f"⚡ {x['district']} MEDIUM RISK" if x["risk_level"]=="MEDIUM"
    else f"✅ {x['district']} LOW RISK", axis=1)

# ফলাফল প্রদর্শন এবং সেভ করা
print(df[["district", "total_risk_score", "risk_level", "alert"]])
df.to_csv("analyzed_risk_data.csv", index=False)
