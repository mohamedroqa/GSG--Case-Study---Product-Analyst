import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="GSG Product Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

plt.style.use("seaborn-v0_8")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #0f172a;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .small-note {
        color: #475569;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📊 GSG Product Analytics Dashboard")
st.markdown(
    "<div class='small-note'>لوحة معلومات احترافية لعرض نتائج تحليل بيانات التطبيق بشكل واضح وقابل للمشاركة.</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
def normalize_ad_format(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"

    value = str(value).strip().lower()
    mapping = {
        "inter": "Interstitial",
        "banner": "Banner",
        "reward": "Rewarded Video",
        "rewarded": "Rewarded Video",
        "rewarded video": "Rewarded Video",
        "rewarded_video": "Rewarded Video",
        "offerwall": "Offerwall",
        "missing": "Unknown"
    }
    return mapping.get(value, value.title())


@st.cache_data(show_spinner=True)
def load_data(file_source, file_name=""):
    important_columns = [
        "user_id",
        "variation",
        "assignment_dt",
        "install_dt",
        "created_at",
        "country_code",
        "platform",
        "os_version",
        "channel",
        "assigned",
        "activated",
        "is_ad_tracking_limited",
        "revenue_amount",
        "cashout_amount",
        "earning_amount",
        "cashout_transactions",
        "ad_unit_format",
        "revenue_type"
    ]

    # تحديد نوع الملف
    source_name = file_name.lower() if file_name else str(file_source).lower()

    if source_name.endswith(".parquet"):
        df = pd.read_parquet(file_source)
    else:
        df = pd.read_csv(file_source, low_memory=True)

    # الاحتفاظ فقط بالأعمدة المهمة الموجودة فعليًا
    existing_cols = [col for col in important_columns if col in df.columns]
    df = df[existing_cols].copy()

    # تحويل التواريخ
    for col in ["assignment_dt", "install_dt", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # تنظيف النصوص
    if "os_version" in df.columns:
        df["os_version"] = df["os_version"].fillna("unknown")

    if "ad_unit_format" in df.columns:
        df["ad_unit_format"] = df["ad_unit_format"].apply(normalize_ad_format)
    else:
        df["ad_unit_format"] = "Unknown"

    # تحويل الأعمدة الرقمية
    for col in ["revenue_amount", "cashout_amount", "earning_amount", "cashout_transactions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # صافي الإيراد
    if "revenue_amount" in df.columns and "cashout_amount" in df.columns:
        df["net_revenue"] = df["revenue_amount"] - df["cashout_amount"]
    else:
        df["net_revenue"] = 0

    return df


def calculate_retention(df):
    required_cols = ["user_id", "install_dt", "created_at"]
    if not all(col in df.columns for col in required_cols):
        return {1: 0, 3: 0, 7: 0}

    user_installs = df.groupby("user_id", as_index=False)["install_dt"].min()
    events = df[["user_id", "created_at"]].dropna().drop_duplicates()
    merged = events.merge(user_installs, on="user_id", how="left")
    merged["days_since_install"] = (merged["created_at"] - merged["install_dt"]).dt.days

    total_users = user_installs["user_id"].nunique()
    if total_users == 0:
        return {1: 0, 3: 0, 7: 0}

    retention = {}
    for day in [1, 3, 7]:
        retained_users = merged.loc[merged["days_since_install"] >= day, "user_id"].nunique()
        retention[day] = round((retained_users / total_users) * 100, 2)

    return retention


def calculate_anova(df):
    if "variation" not in df.columns or "net_revenue" not in df.columns:
        return None, None

    grouped = []
    for _, group in df.groupby("variation"):
        series = group["net_revenue"].dropna()
        if len(series) > 1:
            grouped.append(series)

    if len(grouped) < 2:
        return None, None

    f_stat, p_value = f_oneway(*grouped)
    return float(f_stat), float(p_value)


def calculate_conversion(df):
    total_users = df["user_id"].nunique() if "user_id" in df.columns else 0
    users_with_cashout = df.loc[df["cashout_amount"] > 0, "user_id"].nunique() if "user_id" in df.columns else 0
    cashout_conversion = round((users_with_cashout / total_users) * 100, 2) if total_users else 0

    total_transactions = len(df)
    cashout_transactions = len(df[df["revenue_type"] == "cashout"]) if "revenue_type" in df.columns else 0
    transaction_conversion = round((cashout_transactions / total_transactions) * 100, 2) if total_transactions else 0

    total_revenue = df["net_revenue"].sum() if "net_revenue" in df.columns else 0
    avg_revenue_per_cashout = round(total_revenue / users_with_cashout, 2) if users_with_cashout else 0

    return cashout_conversion, transaction_conversion, avg_revenue_per_cashout, total_revenue


# -----------------------------
# Sidebar: upload file
# -----------------------------
st.sidebar.header("⚙️ إعدادات الداشبورد")
uploaded_file = st.sidebar.file_uploader(
    "ارفع ملف البيانات",
    type=["csv", "parquet"]
)

if uploaded_file is None:
    st.info("ارفع ملف CSV أو Parquet من الشريط الجانبي لعرض البيانات.")
    st.stop()

st.sidebar.caption("الملف المرفوع:")
st.sidebar.code(uploaded_file.name, language="text")

with st.spinner("جاري تحميل البيانات..."):
    df = load_data(uploaded_file, uploaded_file.name)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 الفلاتر")

country_options = ["All"] + sorted(df["country_code"].dropna().astype(str).unique().tolist()) if "country_code" in df.columns else ["All"]
ad_options = ["All"] + sorted(df["ad_unit_format"].dropna().astype(str).unique().tolist()) if "ad_unit_format" in df.columns else ["All"]
variation_options = ["All"] + sorted(df["variation"].dropna().astype(str).unique().tolist()) if "variation" in df.columns else ["All"]

selected_country = st.sidebar.selectbox("الدولة", country_options)
selected_ad = st.sidebar.selectbox("نوع الإعلان", ad_options)
selected_variation = st.sidebar.selectbox("المجموعة التجريبية", variation_options)

filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country_code"].astype(str) == selected_country]

if selected_ad != "All":
    filtered_df = filtered_df[filtered_df["ad_unit_format"].astype(str) == selected_ad]

if selected_variation != "All":
    filtered_df = filtered_df[filtered_df["variation"].astype(str) == selected_variation]

# -----------------------------
# Data calculations
# -----------------------------
retention = calculate_retention(filtered_df)
f_stat, p_value = calculate_anova(filtered_df)
cashout_conversion, transaction_conversion, avg_revenue_per_cashout, total_revenue = calculate_conversion(filtered_df)

revenue_by_ad = (
    filtered_df.groupby("ad_unit_format", as_index=False)
    .agg(
        revenue=("net_revenue", "sum"),
        transactions=("revenue_type", "count")
    )
    .sort_values("revenue", ascending=False)
) if "ad_unit_format" in filtered_df.columns else pd.DataFrame()

revenue_by_country = (
    filtered_df.groupby("country_code", as_index=False)["net_revenue"]
    .sum()
    .sort_values("net_revenue", ascending=False)
) if "country_code" in filtered_df.columns else pd.DataFrame()

retention_df = pd.DataFrame({
    "Day": [1, 3, 7],
    "Retention": [retention[1], retention[3], retention[7]]
})

# -----------------------------
# KPI cards
# -----------------------------
st.markdown("## ملخص تنفيذي")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("إجمالي الإيرادات الصافية", f"{total_revenue:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Retention Day 1", f"{retention[1]:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Cashout Conversion", f"{cashout_conversion:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("عدد المستخدمين", f"{filtered_df['user_id'].nunique():,}" if "user_id" in filtered_df.columns else "0")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 الإيرادات",
    "📉 الاحتفاظ",
    "🧪 ANOVA",
    "💸 التحويل",
    "🗂️ البيانات والتوصيات"
])

# -----------------------------
# Tab 1: Revenue
# -----------------------------
with tab1:
    st.markdown("### تحليل الإيرادات حسب نوع الإعلان")
    if not revenue_by_ad.empty:
        st.dataframe(revenue_by_ad, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(
                revenue_by_ad["ad_unit_format"].astype(str),
                revenue_by_ad["revenue"],
                color=colors[:len(revenue_by_ad)]
            )
            ax.set_title("Net Revenue by Ad Format", fontsize=14, fontweight="bold")
            ax.set_xlabel("Ad Format")
            ax.set_ylabel("Net Revenue")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            plt.xticks(rotation=20)
            st.pyplot(fig)

        with c2:
            colors = ["#0ea5e9", "#f97316", "#22c55e", "#e11d48", "#a855f7", "#06b6d4"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(
                revenue_by_ad["ad_unit_format"].astype(str),
                revenue_by_ad["transactions"],
                color=colors[:len(revenue_by_ad)]
            )
            ax.set_title("Transactions by Ad Format", fontsize=14, fontweight="bold")
            ax.set_xlabel("Ad Format")
            ax.set_ylabel("Transactions")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            plt.xticks(rotation=20)
            st.pyplot(fig)

    st.markdown("### الإيرادات حسب الدولة")
    if not revenue_by_country.empty:
        top_countries = revenue_by_country.head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            top_countries["country_code"].astype(str),
            top_countries["net_revenue"],
            color="#10b981"
        )
        ax.set_title("Top 10 Countries by Net Revenue", fontsize=14, fontweight="bold")
        ax.set_xlabel("Country")
        ax.set_ylabel("Net Revenue")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        st.pyplot(fig)

# -----------------------------
# Tab 2: Retention
# -----------------------------
with tab2:
    st.markdown("### معدل الاحتفاظ")
    st.dataframe(retention_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        retention_df["Day"],
        retention_df["Retention"],
        marker="o",
        linewidth=3,
        markersize=8,
        color="#2563eb"
    )
    ax.fill_between(retention_df["Day"], retention_df["Retention"], color="#93c5fd", alpha=0.25)
    ax.set_title("Retention Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Retention %")
    ax.set_xticks([1, 3, 7])
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)

# -----------------------------
# Tab 3: ANOVA
# -----------------------------
with tab3:
    st.markdown("### اختبار ANOVA")
    if f_stat is not None and p_value is not None:
        c1, c2 = st.columns(2)
        c1.metric("F-statistic", f"{f_stat:.2f}")
        c2.metric("p-value", f"{p_value:.4f}")

        if p_value > 0.05:
            st.info("لا يوجد فرق ذو دلالة إحصائية بين المجموعات بناءً على قيمة p-value.")
        else:
            st.success("يوجد فرق ذو دلالة إحصائية بين المجموعات.")
    else:
        st.warning("تعذر حساب ANOVA لأن البيانات لا تحتوي على عدد كافٍ من المجموعات أو القيم.")

    if "variation" in filtered_df.columns:
        variation_counts = filtered_df["variation"].astype(str).value_counts().reset_index()
        variation_counts.columns = ["variation", "count"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            variation_counts["variation"],
            variation_counts["count"],
            color="#8b5cf6"
        )
        ax.set_title("Records by Variation Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Variation")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.xticks(rotation=20)
        st.pyplot(fig)

# -----------------------------
# Tab 4: Conversion
# -----------------------------
with tab4:
    st.markdown("### تحليل معدل التحويل")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cashout Conversion", f"{cashout_conversion:.2f}%")
    c2.metric("Transaction Conversion", f"{transaction_conversion:.2f}%")
    c3.metric("Avg Revenue per Cashout", f"{avg_revenue_per_cashout:.2f}")

    conversion_df = pd.DataFrame({
        "Metric": ["Cashout Conversion", "Transaction Conversion"],
        "Value": [cashout_conversion, transaction_conversion]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(conversion_df["Metric"], conversion_df["Value"], color=["#f59e0b", "#06b6d4"])
    ax.set_title("Conversion Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig)

# -----------------------------
# Tab 5: Data + Recommendations
# -----------------------------
with tab5:
    st.markdown("### معاينة البيانات")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.markdown("### توصيات عملية")
    st.markdown(
        """
        1. **التركيز على أنواع الإعلانات الأعلى أداءً** مثل Interstitial إذا أكدت الفلاتر والبيانات ذلك.  
        2. **مراجعة القيم Unknown** أو القيم الفارغة في نوع الإعلان لتحسين جودة التتبع.  
        3. **تحسين الاحتفاظ بالمستخدمين** بعد الأيام الأولى عبر الحوافز والإشعارات وتجربة onboarding أقوى.  
        4. **تبسيط رحلة الـ Cashout** لرفع معدل التحويل وتحسين تجربة المستخدم.  
        5. **التركيز على الدول الأعلى تحقيقًا للإيرادات** وتحسين الاستهداف فيها.  
        """
    )

    st.markdown("### ملاحظات")
    st.caption("الداشبورد يقرأ ملف CSV مباشرة من الجهاز ويطبّق الفلاتر بشكل حي على المؤشرات والرسوم.")

st.markdown("---")
st.caption("Built with Streamlit • Product Analytics Case Study")