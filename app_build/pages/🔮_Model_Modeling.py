import pandas as pd
import re
import numpy as np
import streamlit as st

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Config trang -----------------------------------------
st.set_page_config(
    page_title="Data Modeling",
    page_icon="🔮",
    layout="wide"
)

df = pd.read_csv('app_build/analysis_df_employee.csv')
df_non_tech = pd.read_csv('app_build/analysis_df.csv')
df_title_job = pd.read_csv('app_build/analysis_title_salary.csv')


# Tạo tiêu đề -----------------------------------------
col1, col2, col3 = st.columns([1,5,1])

with col1:
    st.write("")
with col2:
    st.image('https://i.pinimg.com/564x/b4/10/1e/b4101eb5bc27a62b8f681bd03da9ffff.jpg')
    st.markdown("<h1 style='text-align: center; color: #B799FF;'>MODEL AND LINEAR REGRESSION</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
with col3:
    st.write("")

# Tiền xử lý dữ liệu -----------------------------------
df_train = pd.read_csv("app_build/analysis_df_model.csv")
new_df = df_train[["Age", "Gender", "Salary", "Title", "Formal Education", "Coding Experience", "ML Experience", "Country", "Year"]]

new_df = new_df.dropna()

# Xử lý cột tuổi:- --------------------------------------
new_df["Age"] = new_df["Age"].apply(lambda x: x.replace("+", "")).apply(lambda x: sum(map(lambda i: int(i), x.split("-"))) / len(x.split("-")))

# Xử lý cột Gender:--------------------------------------
gender_list = new_df["Gender"].unique()
gender_map = {gender: idx for gender, idx in zip(list(gender_list), range(1, len(list(gender_list)) + 1))}

new_df["Gender"] = new_df["Gender"].map(gender_map)

# Xử lý cột Title:---------------------------------------
title_list = new_df["Title"].unique()
title_map = {title: idx for title, idx in zip(list(title_list), range(1, len(list(title_list)) + 1))}
new_df["Title"] = new_df["Title"].map(title_map)

# Xử lý cột Formal Education:----------------------------
education_list = new_df["Formal Education"].unique()
education_map = {education: idx for education, idx in zip(list(education_list), [10, 7, 20, 30, 3, 15, 8])}
new_df["Formal Education"] = new_df["Formal Education"].map(education_map)

# Xử lý cột Coding Experience:---------------------------  
pattern = r'\b([0-9]+)\b'
exp_list = new_df["Coding Experience"].unique()

exp_numeric = []

for exp in exp_list:
    matches = re.findall(pattern, exp)
    numbers = [int(match[:2]) for match in matches]
    if not numbers:
        numbers = [0]
    exp_numeric.append(numbers)

exp_map = {k: sum(v) / len(v) for k, v in zip(list(exp_list), exp_numeric)}

new_df["Coding Experience"] = new_df["Coding Experience"].map(exp_map)

# Xử lý cột ML Experience:-------------------------------
ml_list = new_df["ML Experience"].unique()
ml_numeric = []

for ml in ml_list:
    matches = re.findall(pattern, ml)
    numbers = [int(match[:2]) for match in matches]
    if not numbers:
        numbers = [0]
    ml_numeric.append(numbers)

ml_map = {k: sum(v) / len(v) for k, v in zip(list(ml_list), ml_numeric)}
new_df["ML Experience"] = new_df["ML Experience"].map(ml_map)

# Xử lý cột Country:-------------------------------------
country_list = new_df["Country"].unique()
country_map = {k: v for k, v in zip(list(country_list), [5, 20, 18, 15])}
new_df["Country"] = new_df["Country"].map(country_map)

# Chia tập dữ liệu thành train và test:------------------
X = new_df.drop("Salary", axis=1)
y = new_df["Salary"]

from sklearn.preprocessing import MinMaxScaler

X = X.to_numpy()
y = y.to_numpy()
mean = np.mean(y)
std = np.std(y)
scaler = MinMaxScaler()
scaler.fit(y.reshape(-1, 1))
y = scaler.transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----
st.markdown("<h3 style='text-align: center; color:green'>1. BÀI TOÁN HỒI QUY</h3>", unsafe_allow_html=True)
st.write("Dự đoán mức lương của 1 người tham gia trả lời dựa vào 1 số thông tin cá nhân của họ.")
st.info(""" **Các cột dữ liệu được sử dụng làm feature:**
- Age: tuổi.
- Gender: giới tính.
- Title: vị trí/vai trò.
- Formal Education: bằng cấp.
- Coding Experience: số năm kinh nghiệm lập trình.
- ML Experience: số năm kinh nghiệm về machine learning.
- Country: quốc gia (gồm 4 quốc gia chính là India, China, USA, VietNam).
- Year: năm người tham gia trả lời câu hỏi.

**Cột dữ liệu được sử dụng làm target:**
- Salary: mức lương.

**Mô hình:** Linear Regression.

**Độ đo:** RMSE và MAE.
""", icon="ℹ️")

# Chọn mô hình ------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predict = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predict))
linear_mae = mean_absolute_error(y_test, linear_predict)
print("Linear Regression RMSE:", linear_rmse)
print("Linear Regression MAE", linear_mae)

# ----
col1, col2 = st.columns(2)

col1.metric(
    label="RMSE",
    value=round(linear_rmse, 3),
)
col2.metric(
    label="MAE",
    value=round(linear_mae, 3),
)

# Thử tạo mẫu dữ liệu -----------------------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>TẠO MẪU DỮ LIỆU</h4>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# ----------------------------------------------
col1, col2, col3 = st.columns(3) 

# ---- Age
age_choice = col1.selectbox("Tuổi", (df_train["Age"].unique()))

lst_age = df_train["Age"].unique()
lst_true_model_age = new_df["Age"].unique()
index = np.where(lst_age == age_choice)[0][0]
age_choice = lst_true_model_age[index]

# ---- Gender
gender_choice = col2.radio("Giới tính", ("Man", "Woman")) 
if gender_choice == 'Man':
    gender_choice = 1
else:
    gender_choice = 2

# ---- Title
lst_title = df_train["Title"].unique()
lst_title = np.delete(lst_title, 13)

title_choice = col3.selectbox("Vị trí/vai trò", lst_title)
lst_true_model_title = new_df["Title"].unique()

index_title = np.where(lst_title == title_choice)[0][0]
title_choice = lst_true_model_title[index_title]
  
# ---- Education
lst_education = df_train["Formal Education"].unique()
lst_education = np.delete(lst_education, 6)
education_choice = col1.selectbox("Bằng cấp", lst_education)

lst_true_model_education = new_df["Formal Education"].unique()
index_education = np.where(lst_education == education_choice)[0][0]

education_choice = lst_true_model_education[index_education]

# ---- Country
lst_country = df_train["Country"].unique()
country_choice = col2.selectbox("Quốc gia", lst_country)

lst_true_model_country = new_df["Country"].unique()
index_country = np.where(lst_country == country_choice)[0][0]

country_choice = lst_true_model_country[index_country]

# ---- Coding Experience
lst_coding_exp = df_train["Coding Experience"].unique()
lst_coding_exp = np.delete(lst_coding_exp, 4)

coding_exp_choice = col3.selectbox("Số năm kinh nghiệm lập trình", lst_coding_exp)

lst_true_model_coding_exp = new_df["Coding Experience"].unique()
index_coding_exp = np.where(lst_coding_exp == coding_exp_choice)[0][0]

coding_exp_choice = lst_true_model_coding_exp[index_coding_exp]

# ---- ML Experience
lst_ml_exp = df_train["ML Experience"].unique()
lst_ml_exp = np.delete(lst_ml_exp, 5)

ml_exp_choice = col1.selectbox("Số năm kinh nghiệm ML", lst_ml_exp)

lst_true_model_ml_exp = new_df["ML Experience"].unique()
index_ml_exp = np.where(lst_ml_exp == ml_exp_choice)[0][0]

ml_exp_choice = lst_true_model_ml_exp[index_ml_exp]

# ---- Year

year_choice = col2.selectbox("Năm", (df_train["Year"].unique()))

#---- Lấy dữ liệu từ tất cả lựa chọn:

data_point = np.array([age_choice, gender_choice, title_choice, education_choice, coding_exp_choice, ml_exp_choice, country_choice, year_choice])
data_point = data_point.reshape(1, -1)
pred = linear_model.predict(data_point)

value_predict = pred[0] * std + mean
#----------------------------------------------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>DỰ ĐOÁN</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Note: Bạn phải chọn các thông số từ điểm dữ liệu trên. </h5>", unsafe_allow_html=True)
col1_value, col2_value = st.columns(2)

col1_value.markdown("#### Mức lương người đó nhận được theo sự lựa chọn của bạn: ", unsafe_allow_html=True)

col2_value.metric(
    label="Mức lương:",
    value=np.around(value_predict, 2),
)

# ----
st.markdown("---", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;color:green'>2. BÀI TOÁN PHÂN LỚP KNN</h3>", unsafe_allow_html=True)

st.markdown("Dự đoán các khoảng lương dựa trên mô hình phân lớp KNN", unsafe_allow_html=True)
st.info(""" **Các cột dữ liệu sử dụng làm feature:**
- Age: Tuổi
- Title: Vị trí/vai trò
- Formal Education: Bằng cấp
- Coding Experience: Số năm kinh nghiệm lập trình
- ML Experience: Số năm kinh nghiệm ML
- Country: Quốc gia

**Cột dữ liệu sử dụng làm target:**
- Salary Range: Khoảng lương.

**Mô hình:** K-Nearest Neighbors (KNN) Classifier.

**Độ đo đánh giá mô hình:** Accuracy Score.

**Số lượng neighbor:** Sau khi dùng GridSearchCV, số lượng neighbor tốt nhất là 24.
""", icon="ℹ️")


col1_KNN, col2_KNN = st.columns(2)