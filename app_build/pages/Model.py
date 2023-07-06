import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold, GridSearchCV
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import streamlit as st

# Config trang -----------------------------------------
st.set_page_config(
    page_title="Model: Linear Regression, K-Means và LSTM",
    page_icon="🔮",
    layout="wide"
)

df = pd.read_excel("app_build/MDSInc_sales.xlsx")
df_2 = pd.read_csv("app_build/Q12_2014.csv", encoding_errors='ignore')
df_2014 = pd.read_csv("app_build/Q12_2014.csv", encoding_errors='ignore')
df_2014["order_date"] = pd.to_datetime(df["order_date"])
df["order_date"] = pd.to_datetime(df["order_date"])


# Tạo tiêu đề -----------------------------------------
col1, col2, col3 = st.columns([1,5,1])

with col1:
    st.write("")
with col2:
    st.image('https://i.pinimg.com/564x/b4/10/1e/b4101eb5bc27a62b8f681bd03da9ffff.jpg')
    st.markdown("<h1 style='text-align: center; color: #B799FF;'>Model: Linear Regression, K-Means và LSTM</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
with col3:
    st.write("")

# Tiền xử lý dữ liệu -----------------------------------

cleaned_df = df.groupby(["customer_name"]).agg({'sales': 'sum', 'quantity': 'sum', 'shipping_cost': 'sum', 'profit': 'sum', 'discount': 'sum'}).reset_index()
new_df = cleaned_df.drop("customer_name", axis=1).to_numpy()

st.markdown("## Customer Segmentation")

st.info(""" **Phân loại khách hàng:**
- Ta cần phân loại khách hàng để tìm kiếm được các nhóm khách hàng tiềm năng để đánh mạnh vào.

**Cột dữ liệu được sử dụng:**
- Sales: mức lương.
- Profit: lợi nhuận
- Quantity: số lượng sản phẩm
- Shipping_cost: phí giao hàng

**Mô hình:** K-Means.

**Độ đo:** Inertia và Silhoutte score.
""", icon="ℹ️")

sum_distances = []
K = range(1,15)
for k in K:
  k_mean = KMeans(n_clusters=k)
  k_mean.fit(new_df)
  sum_distances.append(k_mean.inertia_)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(K), y=sum_distances, mode='lines+markers', marker=dict(color='blue'), name='Inertia value'))
fig.update_layout(
    title="Elbow method",
    xaxis_title="Number of clusters",
    yaxis_title="Inertia value",
    title_font=dict(size=14)
)
st.plotly_chart(fig)

st.info(""" **Elbow method:**
- Sử dụng phương pháp này để xác định số phân cụm tốt nhất cho mô hình (do K-Means cần xác định số cụm trước)
- Tìm điểm cùi chỏ (vị trí mà chỉ số inertia sẽ không còn giảm đáng kể)

""", icon="ℹ️")
st.write("Số phân cụm tốt nhất: `n = 2`")

k_mean_2 = KMeans(n_clusters=2)
model = k_mean_2.fit(new_df)
result = k_mean_2.labels_

st.write('Silhouette score:', metrics.silhouette_score(new_df, result, metric='euclidean'))

obj = {
    "sales": 0,
    "quantity": 1,
    "discount": 2,
    "profit": 3,
    "shipping_cost": 4,
}
column = obj["quantity"]

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=new_df[result == 0, 0], y=new_df[result == 0, column],
        mode='markers', marker=dict(color='lightgreen', symbol='square', size=8),
        name='Segmentation 1'
    ))
    fig.add_trace(go.Scatter(
        x=new_df[result == 1, 0], y=new_df[result == 1, column],
        mode='markers', marker=dict(color='orange', symbol='circle', size=8),
        name='Segmentation 2'
    ))
    fig.add_trace(go.Scatter(
        x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, column],
        mode='markers', marker=dict(color='red', symbol='star', size=12),
        name='Centroids'
    ))
    fig.update_layout(
        legend=dict(
            traceorder='normal',
        ),
        xaxis=dict(title='Sales'),
        yaxis=dict(title='Quantity'),
        title='Customer Segmentation with Sales and Quantity',
        title_font=dict(size=14),
    )
    st.plotly_chart(fig)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=new_df[result == 0, 0], y=new_df[result == 0, obj["shipping_cost"]],
        mode='markers', marker=dict(color='lightgreen', symbol='square', size=8),
        name='Segmentation 1'
    ))
    fig.add_trace(go.Scatter(
        x=new_df[result == 1, 0], y=new_df[result == 1, obj["shipping_cost"]],
        mode='markers', marker=dict(color='orange', symbol='circle', size=8),
        name='Segmentation 2'
    ))
    fig.add_trace(go.Scatter(
        x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, obj["shipping_cost"]],
        mode='markers', marker=dict(color='red', symbol='star', size=12),
        name='Centroids'
    ))
    fig.update_layout(
        legend=dict(
            traceorder='normal',
        ),
        xaxis=dict(title='Sales'),
        yaxis=dict(title='Shipping Cost'),
        title='Customer Segmentation with Sales and Shipping Cost',
        title_font=dict(size=14),
    )
    st.plotly_chart(fig)

st.info(""" **Đề xuất:**
- Phân khúc khách hàng tiêu dùng cao có lương doanh thu > 10k, ta có thể đánh mạnh vào yếu tố này.
- Ví dụ: Xây dựng hệ thống membership với nhiều ưu đãi cho khách hàng có doanh thu > 10k.

""", icon="ℹ️")

linear_df = df.groupby(["order_date"]).agg({'sales': 'sum', 'quantity': 'sum', 'shipping_cost': 'sum', 'profit': 'sum', 'discount': 'sum'}).reset_index()
linear_df = linear_df.drop("order_date", axis=1)

y = linear_df.iloc[:, 0]
X = linear_df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

st.markdown("## Find best feature (K-Fold cross validation)")

st.info(""" **Tìm yếu tố ảnh hưởng nhất đến doanh thu:**
- Trong việc kinh doanh ta luôn muốn biết yếu tố nào có tác động lớn nhất đến doanh thu của cửa hàng, từ đó khai thác được thế mạnh trong khía cạnh đó.
- Việc tìm được đặc trưng tốt nhất sẽ giúp ta đồng thời hiểu được sự tương quan giữa các thuộc tính.

**Phương pháp sử dụng:**
- K-Fold cross validation: kiểm tra chéo

**Độ đo đánh giá:**
- RMSE
""", icon="ℹ️")

def model_rmse(y_test, y_pred):
    return np.sqrt(np.mean((y_test.ravel() - y_pred.ravel())**2))

def train_each_feature_cross_validation(train, fold = 5):
    feature = {k: 0 for k in train.columns if k != 'sales'}

    for train_split, test_split in KFold(n_splits=fold, shuffle=True).split(train):
        for column in feature.keys():
            feature_train = np.array(train.iloc[train_split].loc[:,[column]])
            label_train = np.array(train.iloc[train_split].loc[:,['sales']])

            feature_test = np.array(train.iloc[test_split].loc[:,[column]])
            label_test = np.array(train.iloc[test_split].loc[:,['sales']])
                
            model = lr().fit(feature_train, label_train)
            pred = model.predict(feature_test)
            rmse = model_rmse(label_test, pred)

            feature[column] += rmse

    return {k: v/fold for k, v in feature.items()}

features = train_each_feature_cross_validation(pd.concat([X_train, y_train], axis=1))
best_feature = min(features, key=features.get)
st.write(f'The best feature in the dataset (most correlated feature to label): `{best_feature}`')
st.write(f"RMSE của đặc trưng tốt nhất: `{features[best_feature]}`")

st.info(""" **Nhận xét:**
- Chi phí giao hàng mới là yếu tố quan trọng nhất tác động đến doanh thu.
- Để làm rõ được nhận định này, ta cần trực quan bằng scatter plot giữa 2 thuộc tính này và biểu đồ tương quan

**Biểu đồ:**
- Scatter plot: 2 thuộc tính `sales` và `shipping_cost`
- Biểu đồ tương quan: giữa các đặc trưng với nhau + giữa 2 thuộc tính `sales` và `shipping_cost` (phương pháp spearman)

""", icon="ℹ️")

st.write("### Scatter plot")

fig = px.scatter(linear_df, x=linear_df[best_feature], y=linear_df['sales'], color=linear_df[best_feature])
fig.update_layout(xaxis_type='log', yaxis_type='log', title="Correlation between best feature and label")
st.plotly_chart(fig)

st.write("### Biều đồ tương quan")

col1, col2 = st.columns(2)

with col1:
    correlation_matrix = linear_df.corr(method='spearman')
    fig = px.imshow(correlation_matrix, text_auto = True, color_continuous_scale = 'RdYlBu', title="Correlation Heatmap of all features and label")
    st.plotly_chart(fig)
with col2:
    correlation_matrix = linear_df[["sales", "shipping_cost"]].corr(method='spearman')
    fig = px.imshow(correlation_matrix, text_auto = True, color_continuous_scale = 'RdYlBu', title="Correlation Heatmap of Sales and Shipping Cost")
    st.plotly_chart(fig)

st.markdown("## Sales prediction with Linear Regression")

st.info(""" **Dự đoán doanh thu:**
- Dự đoán trước được lượng doanh thu với những đặc trưng cho trước tưởng chừng như đơn giản những là 1 điều rất cần thiết trong việc phân tích dữ liệu.
- Cần ước lượng trước doanh thu để đưa ra các quyết định điều chỉnh mô hình kinh doanh kịp thời.

**Phương pháp sử dụng:**
- Linear Regression: hồi quy tuyến tính
- RepeatedStratifiedKFold và GridSearchCV: tìm ra bộ tham số tốt nhất cho mô hình LR

**Độ đo đánh giá:**
- MSE
- MAE
- R2 score (Accuracy score)

""", icon="ℹ️")

cleaned_df = df.groupby(df["order_date"]).agg({'sales': 'sum', 'quantity': 'sum', 'shipping_cost': 'sum', 'profit': 'sum', 'discount': 'sum'}).reset_index()
cleaned_df = cleaned_df.drop(["order_date", "discount", "profit"], axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=cleaned_df.index, y=cleaned_df['sales'], mode='lines', name='Sales in Time Series'))
fig.update_layout(title='Sales in Time Series', xaxis_title='Time', yaxis_title='Sales')
st.plotly_chart(fig)

# cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
# linear_grid = {'positive': [True,False], 
#                 'fit_intercept': [True, False], 
#                 'n_jobs': [1, -1]}
# linear = GridSearchCV( lr(), param_grid=linear_grid,
#                           cv=cv, 
#                           scoring='completeness_score', 
#                           verbose=0)
# linear.fit(X_train, y_train)

model2 = lr().fit(X_train, y_train)
pred = model2.predict(X_test)
result2 = pd.DataFrame(pred, columns=['sales'], index=X_test.index)
st.write('MSE of biased model:', metrics.mean_squared_error(y_test, pred))
st.write('MAE of biased model:', metrics.mean_absolute_error(y_test, pred))
st.write('Model score:', model2.score(X_test, y_test))

st.info(""" **Nhận xét:**
- Mô hình với bộ tham số tốt nhất cho ra accuracy score khá cao với khoảng 90%.
- Chúng ta cần kiểm tra với bộ dữ liệu doanh thu trong Q1 và Q2 năm 2014.

""", icon="ℹ️")

df_2 = df_2.groupby(["order_date"]).agg({'sales': 'sum', 'quantity': 'sum', 'shipping_cost': 'sum', 'profit': 'sum', 'discount': 'sum'}).reset_index()
test_df = df_2.drop("order_date", axis=1)

test_X = test_df.iloc[:, 1:]
y_true = test_df.iloc[:, 0]

y_test = model2.predict(test_X)
temp = y_true.to_numpy().reshape(-1, 1)
df_2["order_date"] = pd.to_datetime(df_2["order_date"], format="mixed", dayfirst=True).dt.strftime("%d-%b-%Y")

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df_2["order_date"].values), y=y_test, mode='lines', name='True Sales', line=dict(color='green')))
fig.add_trace(go.Scatter(x=list(df_2["order_date"].values), y=temp[:, 0], mode='lines', name='Predict Sales', line=dict(color='red')))
fig.update_layout(
    title='Sales prediction vs true values in Q1 and Q2 in 2014',
    xaxis_title='Date',
    yaxis_title='Total sales',
    showlegend=True
)

st.plotly_chart(fig)
st.write("R2 score (Accuracy):", metrics.r2_score(y_true.to_numpy(), y_test))

st.markdown("## Sales prediction with LSTM (RNN)")

st.info(""" **Dự đoán doanh thu với mô hình Deep Learning:**
- Sử dụng Deep Learning sẽ cho ra kết quả tốt hơn và đáng tin cậy hơn so với Linear Regression là mô hình có giám sát.
- Đồng thời, sử dụng Deep Learning để đối chiếu kết quả thực nghiệm với mô hình LR trước đó.

**Mô hình sử dụng:**
- LSTM (Long short-term memory): mô hình biến thể của RNN (Recurrent Neural Network)

**Lý do sử dụng:**
- RNN được biết đến với khả năng dự đoán dữ liệu dạng chuỗi, đặc biệt là Time Series (dữ liệu dòng thời gian). LSTM cho phép mô hình dự đoán tốt hơn khi khắc phục được nhược điểm phụ thuộc xa của RNN.

**Phương pháp đánh giá:**
- R2 score (Accuracy)

""", icon="ℹ️")

model_df = StandardScaler().fit_transform(cleaned_df)
train_X, train_y = model_df, model_df[2:, 0]

model = keras.Sequential()
model.add(layers.GRU(
    units = 128,
    input_shape =(5,3)
))
model.add(layers.Dense(units = 1))
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(0.001)
)

full_n = train_y.shape[0]
n = int(train_y.shape[0]*0.7)

train = keras.preprocessing.timeseries_dataset_from_array(train_X[:n],train_y[:n],sequence_length=5,batch_size=1) #train
test = keras.preprocessing.timeseries_dataset_from_array(train_X[n:],train_y[n:],sequence_length=5,batch_size=1) #test

test_x = np.array([i[0] for i in test])
test_x.resize(( full_n - n, 5, 3))

model.fit(train)
y_pred = model.predict(test_x)

fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(len(train_y[:n])), y=train_y[:n], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=np.arange(n, full_n), y=train_y[n:], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=np.arange(n, full_n), y=y_pred[:, 0], mode='lines', name='Predict'))

fig.update_layout(title='Sales prediction in 2011 to 2013',
                  xaxis_title='Time',
                  yaxis_title='Sales',
                  showlegend=True)

fig.update_xaxes(showgrid=True, zeroline=True)
fig.update_yaxes(showgrid=True, zeroline=True)
st.plotly_chart(fig)

st.write("R2 score:", r2_score( np.array(train_y)[n:],y_pred))

st.info(""" **Nhận xét:**
- LSTM cho ra độ chính xác cao hơn so với Linear Regression, khi dự đoán khớp khoảng 94% so với tập validation.
- Tiếp theo cần kiểm tra trên tập dữ liệu doanh thu Q1 và Q2 của năm 2014.

""", icon="ℹ️")

df_2014 = df_2014.groupby(["order_date"]).agg({'sales': 'sum', 'quantity': 'sum', 'shipping_cost': 'sum', 'profit': 'sum', 'discount': 'sum'}).reset_index()
dates = df_2014["order_date"].dt.strftime("%d-%b-%Y")

df_2014 = df_2014.drop(["order_date", "discount", "profit"], axis=1)

model_df_2014 = StandardScaler().fit_transform(df_2014)

X, y = model_df_2014, model_df_2014[5:, 0]

another_test = keras.preprocessing.timeseries_dataset_from_array(X,y,sequence_length=5,batch_size=1)

pred = model.predict(another_test)

fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(len(dates)-5), y=X[2:-3, 0], mode='lines', name='Test'))
fig.add_trace(go.Scatter(x=np.arange(len(dates)-5), y=pred[:, 0], mode='lines', name='Predict'))

fig.update_layout(title='Sales prediction in Q1 and Q2 of 2014',
                  xaxis_title='Time',
                  yaxis_title='Sales',
                  showlegend=True)

fig.update_xaxes(showgrid=True, zeroline=True)
fig.update_yaxes(showgrid=True, zeroline=True)
st.plotly_chart(fig)

st.write("R2 score:", r2_score(X[2:-3, 0], pred))
