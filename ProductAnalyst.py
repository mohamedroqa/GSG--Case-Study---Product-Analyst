import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

print("----------------------------------------")

# تحميل بيانات المستخدمين
users_df = pd.read_csv(r"C:\Users\M.S.I\Desktop\File folders for storing software projects\GSG- Case Study - Product Analyst\users_data_final_case_study.csv")

# تحميل بيانات الإيرادات
revenue_df = pd.read_csv(r"C:\Users\M.S.I\Desktop\File folders for storing software projects\GSG- Case Study - Product Analyst\revenue_data_final_case_study.csv")
print("----------------------------------------")

# عرض أول 5 صفوف من بيانات المستخدمين
print(users_df.head())
# عرض أول 5 صفوف من بيانات الإيرادات
print(revenue_df.head())

print("----------------------------------------")


# فحص التفاصيل مثل نوع البيانات وعدد الصفوف والأعمدة
print(users_df.info())
print(revenue_df.info())
print("----------------------------------------")

# معرفة أنواع البيانات لكل عمود
print(users_df.dtypes)  # لبيانات المستخدمين
print(revenue_df.dtypes)  # لبيانات الإيرادات
print("----------------------------------------")

# فحص القيم المفقودة في البيانات
print(users_df.isnull().sum())
print(revenue_df.isnull().sum())
print("----------------------------------------")

# تحويل الأعمدة إلى نوع datetime
users_df['assignment_dt'] = pd.to_datetime(users_df['assignment_dt'])
users_df['install_dt'] = pd.to_datetime(users_df['install_dt'])
revenue_df['created_at'] = pd.to_datetime(revenue_df['created_at'])
print("----------------------------------------")


# إحصائيات لبيانات رقمية مثل الإيرادات
print(users_df.describe())  # لبيانات المستخدمين
print(revenue_df.describe())  # لبيانات الإيرادات
print("----------------------------------------")

# حساب تكرار القيم في عمود معين
print(users_df['platform'].value_counts())  # لبيانات المستخدمين
print(revenue_df['ad_unit_format'].value_counts())  # لبيانات الإيرادات
print("----------------------------------------")

# عرض القيم المتطرفة في عمود الإيرادات
print(revenue_df['revenue_amount'].describe())
print("----------------------------------------")

# فحص التوزيع الزمني للبيانات
users_df['install_dt'] = pd.to_datetime(users_df['install_dt'])
print(users_df['install_dt'].min(), users_df['install_dt'].max())
print("----------------------------------------")
# دمج بيانات المستخدمين مع بيانات الإيرادات باستخدام user_id
merged_df = pd.merge(users_df, revenue_df, on='user_id', how='inner')

# التحقق من حجم البيانات المدمجة
print(merged_df.shape)  # إظهار عدد الصفوف والأعمدة بعد الدمج

# عرض أول 5 صفوف من البيانات المدمجة للتحقق
print(merged_df.head())
print("----------------------------------------")
 
# عرض أول 5 صفوف من البيانات المدمجة
print(merged_df.head())

# فحص معلومات البيانات: عدد الصفوف والأعمدة ونوع البيانات لكل عمود
print(merged_df.info())

# فحص أنواع البيانات لكل عمود
print(merged_df.dtypes)

# إحصائيات عامة للأعمدة الرقمية
print(merged_df.describe())

# فحص عدد القيم المفقودة لكل عمود
print(merged_df.isnull().sum())
print("----------------------------------------")


# استبدال القيم المفقودة في 'os_version' بـ "unknown"
users_df['os_version'] = users_df['os_version'].fillna('unknown')

# استبدال القيم المفقودة في 'ad_unit_format' بـ "unknown"
revenue_df['ad_unit_format'] = revenue_df['ad_unit_format'].fillna('unknown')

print("----------------------------------------")

# فحص القيم المفقودة بعد الاستبدال
print(users_df.isnull().sum())
print(revenue_df.isnull().sum())

print(users_df['os_version'].isnull().sum())  # للتحقق من عدم وجود قيم مفقودة في عمود os_version
print(revenue_df['ad_unit_format'].isnull().sum())  # للتحقق من عدم وجود قيم مفقودة في عمود ad_unit_format
print("----------------------------------------")

print(merged_df.info())

print("----------------------------------------")
# استبدال القيم المفقودة في الجدول المدمج
merged_df['os_version'] = merged_df['os_version'].fillna('unknown')
merged_df['ad_unit_format'] = merged_df['ad_unit_format'].fillna('unknown')

# التحقق من القيم المفقودة بعد الاستبدال في الجدول المدمج
print(merged_df.isnull().sum())

print("----------------------------------------")

merged_df['net_revenue'] = merged_df['revenue_amount'] - merged_df['cashout_amount']
print("----------------------------------------")


revenue_by_ad_format = merged_df.groupby('ad_unit_format')['revenue_amount'].sum()
print(revenue_by_ad_format)

print("----------------------------------------")

impressions_by_ad_format = merged_df.groupby('ad_unit_format')['revenue_type'].count()
print(impressions_by_ad_format)

print("----------------------------------------")

merged_df['retention_day_1'] = merged_df['install_dt'] + pd.Timedelta(days=1)
merged_df['retention_day_3'] = merged_df['install_dt'] + pd.Timedelta(days=3)
merged_df['retention_day_7'] = merged_df['install_dt'] + pd.Timedelta(days=7)

retention_day_1 = merged_df[merged_df['retention_day_1'] <= merged_df['created_at']].shape[0] / merged_df.shape[0]
retention_day_3 = merged_df[merged_df['retention_day_3'] <= merged_df['created_at']].shape[0] / merged_df.shape[0]
retention_day_7 = merged_df[merged_df['retention_day_7'] <= merged_df['created_at']].shape[0] / merged_df.shape[0]

print(f"Retention Day 1: {retention_day_1}")
print(f"Retention Day 3: {retention_day_3}")
print(f"Retention Day 7: {retention_day_7}")

print("----------------------------------------")

# تحضير البيانات لكل مجموعة
# تحضير البيانات لكل مجموعة بعد تعديل الأسماء
group1 = merged_df[merged_df['variation'] == 'androidndorder']['net_revenue']
group2 = merged_df[merged_df['variation'] == 'androidnd1order']['net_revenue']
group3 = merged_df[merged_df['variation'] == 'androidnd7order']['net_revenue']
group4 = merged_df[merged_df['variation'] == 'androidnd30order']['net_revenue']


# إجراء اختبار ANOVA
f_stat, p_value = f_oneway(group1, group2, group3, group4)

print(f"F-statistic: {f_stat}, p-value: {p_value}")
if p_value < 0.05:
    print("هناك فرق ذو دلالة إحصائية بين المجموعات")
else:
    print("لا يوجد فرق ذو دلالة إحصائية بين المجموعات")

print(merged_df['variation'].unique())
 
print(f"Group 1 (androidNdOrder): {group1.count()}")
print(f"Group 2 (androidNd1Order): {group2.count()}")
print(f"Group 3 (androidNd7Order): {group3.count()}")
print(f"Group 4 (androidNd30Order): {group4.count()}")

print("----------------------------------------")

# حساب عدد المستخدمين الذين قاموا بعملية Cashout
users_with_cashout = merged_df[merged_df['cashout_amount'] > 0]

# حساب إجمالي عدد المستخدمين الذين ثبتوا التطبيق
total_users = merged_df.shape[0]

# حساب النسبة المئوية للمستخدمين الذين قاموا بعملية Cashout
conversion_rate_cashout = users_with_cashout.shape[0] / total_users * 100

# حساب عدد المعاملات (transactions) التي أجراها المستخدمون
total_transactions = merged_df['revenue_type'].count()

# حساب إجمالي المعاملات التي تحتوي على Cashout
cashout_transactions = merged_df[merged_df['revenue_type'] == 'cashout'].shape[0]

# حساب النسبة المئوية للمعاملات (transactions)
transaction_conversion_rate = cashout_transactions / total_transactions * 100

# ربط هذه النسب بالإيرادات
total_revenue = merged_df['net_revenue'].sum()
average_revenue_per_conversion = total_revenue / users_with_cashout.shape[0]

# عرض النتائج
print(f"Conversion rate for Cashout: {conversion_rate_cashout:.2f}%")
print(f"Conversion rate for Transactions: {transaction_conversion_rate:.2f}%")
print(f"Total Revenue: {total_revenue}")
print(f"Average Revenue per Cashout Conversion: {average_revenue_per_conversion:.2f}")

print("----------------------------------------")


# رسم توزيع الإيرادات
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['net_revenue'], kde=True, color='skyblue', bins=50)
plt.title('Distribution of Net Revenue')
plt.xlabel('Net Revenue')
plt.ylabel('Frequency')
plt.show()

# رسم عدد التفاعلات حسب نوع الإعلان
plt.figure(figsize=(10, 6))
sns.countplot(x='ad_unit_format', data=merged_df, palette='Set2')
plt.title('Distribution of Transactions by Ad Unit Format')
plt.xlabel('Ad Unit Format')
plt.ylabel('Number of Transactions')
plt.show()

# رسم الإيرادات حسب البلد
plt.figure(figsize=(12, 6))
revenue_by_country = merged_df.groupby('country_code')['net_revenue'].sum().reset_index()
sns.barplot(x='country_code', y='net_revenue', data=revenue_by_country, palette='viridis')
plt.title('Total Revenue by Country')
plt.xlabel('Country')
plt.ylabel('Total Net Revenue')
plt.show()

# رسم بياني لعدد المستخدمين النشطين عبر الأيام
plt.figure(figsize=(10, 6))
sns.lineplot(data=merged_df.groupby(merged_df['install_dt'].dt.date)['user_id'].count())
plt.title('Active Users per Day')
plt.xlabel('Date')
plt.ylabel('Number of Active Users')
plt.xticks(rotation=45)
plt.show()


# رسم العلاقة بين الإيرادات وعدد المعاملات
plt.figure(figsize=(10, 6))
sns.scatterplot(x='net_revenue', y='cashout_transactions', data=merged_df)
plt.title('Revenue vs Number of Transactions')
plt.xlabel('Net Revenue')
plt.ylabel('Number of Transactions')
plt.show()
