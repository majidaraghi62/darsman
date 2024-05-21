import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# -----------1
# خواندن دیتاست
df = pd.read_csv("Loans.csv")

# نمایش چند ردیف اول از داده‌ه
print(df.head())

# -----------2
# تشخیص و پردازش مقادیر گم‌شده
missing_values = df.isnull().sum() 
print(missing_values)

# تابع برای پر کردن مقادیر عددی خالی با میانگین
def fill_numeric_with_mean(df, column):
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)

# تابع برای پر کردن مقادیر متنی خالی با مقدار پر کاربرد
def fill_categorical_with_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)

# پیمایش ستون‌های دیتاست و پر کردن مقادیر خالی
for column in df.columns:
    if df[column].dtype in [np.float64, np.int64]:
        fill_numeric_with_mean(df, column)
    elif df[column].dtype == object:
        fill_categorical_with_mode(df, column)

# ------------3
numeric_columns = df.select_dtypes(include=['number']).columns  # دریافت نام ستون‌های عددی

# تشخیص و پردازش مقادیر پرت
for col in numeric_columns:
    # محاسبه میانگین و انحراف معیار ستون
    mean = df[col].mean()
    std = df[col].std()
    
    # محدوده نقاط مرزی  برای تشخیص مقادیر پرت ( 3 انحراف معیاری)
    threshold = 3
    lower_threshold = mean - threshold * std
    upper_threshold = mean + threshold * std
    
    # تشخیص مقادیر پرت و پردازش آن‌ها ( حذف)
    outliers = df[(df[col] < lower_threshold) | (df[col] > upper_threshold)]
    if not outliers.empty:
        print(f"maghadir part:{col}")
        print(outliers)
        # حذف داده پرت
        df.drop(outliers.index, inplace=True)


# ----------------4

# تبدیل متغیرها
# اینجا فرض می‌کنیم که متغیرهای عددی مثبت هستند
for col in numeric_columns:
    # تبدیل لگاریتمی به متغیرهایی که مقادیر آن‌ها مثبت هستند
    if df[col].min() > 0:
        df[col] = np.log(df[col])


# مقیاس بندی متغیرهای عددی با استفاده از استاندارد سازی
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# تبدیل متغیرهای دسته‌ای با استفاده از Label Encoding
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns  # دریافت نام ستون‌های دسته‌ای
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])



# -------------------5
# مقیاس بندی متغیرهای عددی با استفاده از استاندارد سازی
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# ----------------------6
# تبدیل متغیرهای دسته‌ای با استفاده از Label Encoding
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns  # دریافت نام ستون‌های دسته‌ای
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    

# ---------------تمرین 2


# تعریف فیلدها
features =["client_id","loan_type","repaid","loan_id","loan_start","loan_end","rate"]
X = df[features]
y = df["loan_amount"]
# تقسیم داده به تست و آموزش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# ایجاد مدل رگرسیون خطی
linear_reg_model = LinearRegression()

# آموزش مدل بر روی داده‌های آموزش
linear_reg_model.fit(X_train, y_train)

# پیش‌بینی مقدار وام (loan_amount) برای داده‌های آزمون
y_pred = linear_reg_model.predict(X_test)

# ارزیابی عملکرد مدل با استفاده از معیار Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)




# انتخاب ویژگی‌های موثرترین با استفاده از روش SelectKBest و f_regression
selector = SelectKBest(score_func=f_regression, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# ایجاد مدل رگرسیون خطی چند متغیره
multi_linear_reg_model = LinearRegression()

# آموزش مدل بر روی داده‌های انتخاب شده
multi_linear_reg_model.fit(X_train_selected, y_train)

# پیش‌بینی مقدار وام برای داده‌های آزمون
y_pred_multi = multi_linear_reg_model.predict(X_test_selected)

# ارزیابی عملکرد مدل با استفاده از معیار Mean Squared Error (MSE)
mse_multi = mean_squared_error(y_test, y_pred_multi)
print("Mean Squared Error (MSE) for multi-variable linear regression model:", mse_multi)


# --



# ایجاد مدل رگرسیون چند جمله‌ای با درجه 2
degree_2_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
degree_2_model.fit(X_train, y_train)

# پیش‌بینی مقادیر وام برای داده‌های آزمون با مدل رگرسیون چند جمله‌ای درجه 2
y_pred_degree_2 = degree_2_model.predict(X_test)

# ارزیابی عملکرد مدل با درجه 2 با استفاده از معیار Mean Squared Error (MSE)
mse_degree_2 = mean_squared_error(y_test, y_pred_degree_2)
print("Mean Squared Error (MSE) for polynomial regression model with degree 2:", mse_degree_2)

# ایجاد مدل رگرسیون چند جمله‌ای با درجه 3
degree_3_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
degree_3_model.fit(X_train, y_train)

# پیش‌بینی مقادیر وام برای داده‌های آزمون با مدل رگرسیون چند جمله‌ای درجه 3
y_pred_degree_3 = degree_3_model.predict(X_test)

# ارزیابی عملکرد مدل با درجه 3 با استفاده از معیار Mean Squared Error (MSE)
mse_degree_3 = mean_squared_error(y_test, y_pred_degree_3)
print("Mean Squared Error (MSE) for polynomial regression model with degree 3:", mse_degree_3)
