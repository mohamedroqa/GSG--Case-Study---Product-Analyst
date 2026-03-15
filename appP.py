import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="GSG Product Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

plt.style.use("seaborn-v0_8")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #0f172a; }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .small-note { color: #475569; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 GSG Product Analytics Dashboard")
st.markdown(
    "<div class='small-note'>لوحة معلومات احترافية لعرض نتائج تحليل بيانات التطبيق بشكل واضح وقابل للمشاركة.</div>",
    unsafe_allow_html=True
)

@st.cache_data
def load_dashboard_data():
    kpis = pd.read_csv("dashboard_data/kpis.csv")
    revenue_by_ad = pd.read_csv("dashboard_data/revenue_by_ad.csv")
    revenue_by_country = pd.read_csv("dashboard_data/revenue_by_country.csv")
    retention = pd.read_csv("dashboard_data/retention.csv")
    anova = pd.read_csv("dashboard_data/anova_results.csv")
    variation_counts = pd.read_csv("dashboard_data/variation_counts.csv")
    sample_preview = pd.read_csv("dashboard_data/sample_preview.csv")
    return kpis, revenue_by_ad, revenue_by_country, retention, anova, variation_counts, sample_preview

kpis, revenue_by_ad, revenue_by_country, retention_df, anova_df, variation_counts, sample_preview = load_dashboard_data()

total_revenue = float(kpis.loc[0, "total_revenue"])
total_users = int(kpis.loc[0, "total_users"])
cashout_conversion = float(kpis.loc[0, "cashout_conversion"])
transaction_conversion = float(kpis.loc[0, "transaction_conversion"])
avg_revenue_per_cashout = float(kpis.loc[0, "avg_revenue_per_cashout"])

f_stat = anova_df.loc[0, "f_stat"] if "f_stat" in anova_df.columns else None
p_value = anova_df.loc[0, "p_value"] if "p_value" in anova_df.columns else None

st.markdown("## ملخص تنفيذي")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("إجمالي الإيرادات الصافية", f"{total_revenue:,.2f}")
with col2:
    day1 = retention_df.loc[retention_df["Day"] == 1, "Retention"].iloc[0]
    st.metric("Retention Day 1", f"{day1:.2f}%")
with col3:
    st.metric("Cashout Conversion", f"{cashout_conversion:.2f}%")
with col4:
    st.metric("عدد المستخدمين", f"{total_users:,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 الإيرادات",
    "📉 الاحتفاظ",
    "🧪 ANOVA",
    "💸 التحويل",
    "🗂️ البيانات والتوصيات"
])

with tab1:
    st.markdown("### تحليل الإيرادات حسب نوع الإعلان")
    st.dataframe(revenue_by_ad, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(revenue_by_ad["ad_unit_format"].astype(str), revenue_by_ad["revenue"], color=colors[:len(revenue_by_ad)])
        ax.set_title("Net Revenue by Ad Format", fontsize=14, fontweight="bold")
        ax.set_xlabel("Ad Format")
        ax.set_ylabel("Net Revenue")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.xticks(rotation=20)
        st.pyplot(fig)

    with c2:
        colors = ["#0ea5e9", "#f97316", "#22c55e", "#e11d48", "#a855f7", "#06b6d4"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(revenue_by_ad["ad_unit_format"].astype(str), revenue_by_ad["transactions"], color=colors[:len(revenue_by_ad)])
        ax.set_title("Transactions by Ad Format", fontsize=14, fontweight="bold")
        ax.set_xlabel("Ad Format")
        ax.set_ylabel("Transactions")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.xticks(rotation=20)
        st.pyplot(fig)

    st.markdown("### الإيرادات حسب الدولة")
    top_countries = revenue_by_country.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_countries["country_code"].astype(str), top_countries["net_revenue"], color="#10b981")
    ax.set_title("Top 10 Countries by Net Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Net Revenue")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig)

with tab2:
    st.markdown("### معدل الاحتفاظ")
    st.dataframe(retention_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(retention_df["Day"], retention_df["Retention"], marker="o", linewidth=3, markersize=8, color="#2563eb")
    ax.fill_between(retention_df["Day"], retention_df["Retention"], color="#93c5fd", alpha=0.25)
    ax.set_title("Retention Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Retention %")
    ax.set_xticks([1, 3, 7])
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)

with tab3:
    st.markdown("### اختبار ANOVA")
    if pd.notna(f_stat) and pd.notna(p_value):
        c1, c2 = st.columns(2)
        c1.metric("F-statistic", f"{float(f_stat):.2f}")
        c2.metric("p-value", f"{float(p_value):.4f}")

        if float(p_value) > 0.05:
            st.info("لا يوجد فرق ذو دلالة إحصائية بين المجموعات بناءً على قيمة p-value.")
        else:
            st.success("يوجد فرق ذو دلالة إحصائية بين المجموعات.")
    else:
        st.warning("تعذر حساب ANOVA.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(variation_counts["variation"], variation_counts["count"], color="#8b5cf6")
    ax.set_title("Records by Variation Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Variation")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=20)
    st.pyplot(fig)

with tab4:
    st.markdown("### تحليل معدل التحويل")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cashout Conversion", f"{cashout_conversion:.2f}%")
    c2.metric("Transaction Conversion", f"{transaction_conversion:.2f}%")
    c3.metric("Avg Revenue per Cashout", f"{avg_revenue_per_cashout:.2f}")

with tab5:
    st.markdown("### معاينة البيانات")
    st.dataframe(sample_preview, use_container_width=True)

    st.markdown("### توصيات عملية")
    st.markdown(
        """
        1. **التركيز على أنواع الإعلانات الأعلى أداءً** مثل Interstitial.  
        2. **مراجعة القيم Unknown** لتحسين جودة التتبع.  
        3. **تحسين الاحتفاظ بالمستخدمين** بعد الأيام الأولى.  
        4. **تبسيط رحلة الـ Cashout** لرفع معدل التحويل.  
        5. **التركيز على الدول الأعلى تحقيقًا للإيرادات** وتحسين الاستهداف فيها.  
        """
    )

st.markdown("---")
st.caption("Built with Streamlit • Product Analytics Case Study")