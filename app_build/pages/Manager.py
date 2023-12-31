import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Config trang -----------------------------------------
st.set_page_config(
    page_title="Dashboard Data Analysis",
    page_icon="📊",
    layout="wide"
)

df = pd.read_excel("app_build/MDSInc_sales.xlsx")
df["order_date"] = pd.to_datetime(df["order_date"])

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")
with col2:
    st.markdown("<h1 style='text-align: center; color: #B799FF;'>MANAGER DASHBOARD</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
with col3:
    st.write("")
st.markdown("## 1. Tổng quan")
# plot 0
st.markdown("##### Bộ lọc năm toàn cục:", unsafe_allow_html=True)

year_filter = st.selectbox("Lọc theo năm:", df['order_date'].dt.year.unique())
df_year = df[df["order_date"].dt.year == year_filter]

# Tính năm trước với năm đã được chọn:
if year_filter == 2011:
    pre_year = 2011
else:
    pre_year = year_filter - 1

df_pre_year = df[df["order_date"].dt.year == pre_year]

# Lấy các độ đo thống kê:
df_describe = df_year.describe().round(2).T
df_describe_pre = df_pre_year.describe().round(2).T


tab1, tab2, tab3 = st.tabs(["Profit", "Sales", "Quantity"])
with tab1:
    stats_age_col1, _ = st.columns(2)
    stats_age_col1.metric(
        label="Sum",
        value=round(df_year["profit"].sum(), 2),
        delta=round(float(df_year["profit"].sum() - df_pre_year["profit"].sum()), 2),
    )
with tab2:
    stats_age_col1, _ = st.columns(2)
    stats_age_col1.metric(
        label="Sum",
        value=round(df_year["sales"].sum(), 2),
        delta=round(float(df_year["sales"].sum() - df_pre_year["sales"].sum()), 2),
    )
with tab3:
    stats_age_col1, _ = st.columns(2)
    stats_age_col1.metric(
        label="Sum",
        value=round(df_year["quantity"].sum(), 2),
        delta=round(float(df_year["quantity"].sum() - df_pre_year["quantity"].sum()), 2),
    )


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Profit", "Sales", "Quantity","Discount","Shipping Cost"])
# Tạo các cột để hiển thị thông số thống kê (P1):
with tab1:
    stats_age_col1, stats_age_col2, stats_age_col3, stats_age_col4 = st.columns(4)
    stats_age_col1.metric(
        label="Mean",
        value=df_describe.loc["profit"].loc["mean"],
        delta=round(df_describe.loc["profit"].loc["mean"] - df_describe_pre.loc["profit"].loc["mean"], 2),
    )
    stats_age_col2.metric(
        label="Median",
        value=df_describe.loc["profit"].loc["50%"],
        delta=round(df_describe.loc["profit"].loc["50%"] - df_describe_pre.loc["profit"].loc["50%"], 2),
    )
    stats_age_col3.metric(
        label="Std",
        value=df_describe.loc["profit"].loc["std"],
        delta=round(df_describe.loc["profit"].loc["std"] - df_describe_pre.loc["profit"].loc["std"], 2),
    )
    stats_age_col4.metric(
        label="Count",
        value = int(df_describe.loc["profit", "count"]),
        delta=round(df_describe.loc["profit"].loc["count"] - df_describe_pre.loc["profit"].loc["count"], 2),
    )

    stats_age_col5, stats_age_col6, stats_age_col7, stats_age_col8 = st.columns(4)
    stats_age_col5.metric(
        label="25%",
        value=df_describe.loc["profit"].loc["25%"],
        delta=round(df_describe.loc["profit"].loc["25%"] - df_describe_pre.loc["profit"].loc["25%"], 2),
    )
    stats_age_col6.metric(
        label="75%",
        value=df_describe.loc["profit"].loc["max"],
        delta=round(df_describe.loc["profit"].loc["max"] - df_describe_pre.loc["profit"].loc["max"], 2),
    )
    stats_age_col7.metric(
        label="Min",
        value=df_describe.loc["profit"].loc["min"],
        delta=round(df_describe.loc["profit"].loc["min"] - df_describe_pre.loc["profit"].loc["min"], 2),
    )
    stats_age_col8.metric(
        label="Max",
        value=df_describe.loc["profit"].loc["max"],
        delta=round(df_describe.loc["profit"].loc["max"] - df_describe_pre.loc["profit"].loc["max"], 2),
    )
with tab2:
    stats_age_col1, stats_age_col2, stats_age_col3, stats_age_col4 = st.columns(4)
    stats_age_col1.metric(
        label="Mean",
        value=df_describe.loc["sales"].loc["mean"],
        delta=round(df_describe.loc["sales"].loc["mean"] - df_describe_pre.loc["sales"].loc["mean"], 2),
    )
    stats_age_col2.metric(
        label="Median",
        value=df_describe.loc["sales"].loc["50%"],
        delta=round(df_describe.loc["sales"].loc["50%"] - df_describe_pre.loc["sales"].loc["50%"], 2),
    )
    stats_age_col3.metric(
        label="Std",
        value=df_describe.loc["sales"].loc["std"],
        delta=round(df_describe.loc["sales"].loc["std"] - df_describe_pre.loc["sales"].loc["std"], 2),
    )
    stats_age_col4.metric(
        label="Count",
        value=int(df_describe.loc["sales", "count"]),
        delta=round(df_describe.loc["sales"].loc["count"] - df_describe_pre.loc["sales"].loc["count"], 2),
    )

    stats_age_col5, stats_age_col6, stats_age_col7, stats_age_col8 = st.columns(4)
    stats_age_col5.metric(
        label="25%",
        value=df_describe.loc["sales"].loc["25%"],
        delta=round(df_describe.loc["sales"].loc["25%"] - df_describe_pre.loc["sales"].loc["25%"], 2),
    )
    stats_age_col6.metric(
        label="75%",
        value=df_describe.loc["sales"].loc["max"],
        delta=round(df_describe.loc["sales"].loc["max"] - df_describe_pre.loc["sales"].loc["max"], 2),
    )
    stats_age_col7.metric(
        label="Min",
        value=df_describe.loc["sales"].loc["min"],
        delta=round(df_describe.loc["sales"].loc["min"] - df_describe_pre.loc["sales"].loc["min"], 2),
    )
    stats_age_col8.metric(
        label="Max",
        value=df_describe.loc["sales"].loc["max"],
        delta=round(df_describe.loc["sales"].loc["max"] - df_describe_pre.loc["sales"].loc["max"], 2),
    )
with tab3:
    stats_age_col1, stats_age_col2, stats_age_col3, stats_age_col4 = st.columns(4)
    stats_age_col1.metric(
        label="Mean",
        value=df_describe.loc["quantity"].loc["mean"],
        delta=round(df_describe.loc["quantity"].loc["mean"] - df_describe_pre.loc["quantity"].loc["mean"], 2),
    )
    stats_age_col2.metric(
        label="Median",
        value=df_describe.loc["quantity"].loc["50%"],
        delta=round(df_describe.loc["quantity"].loc["50%"] - df_describe_pre.loc["quantity"].loc["50%"], 2),
    )
    stats_age_col3.metric(
        label="Std",
        value=df_describe.loc["quantity"].loc["std"],
        delta=round(df_describe.loc["quantity"].loc["std"] - df_describe_pre.loc["quantity"].loc["std"], 2),
    )
    stats_age_col4.metric(
        label="Count",
        value=int(df_describe.loc["quantity", "count"]),
        delta=round(df_describe.loc["quantity"].loc["count"] - df_describe_pre.loc["quantity"].loc["count"], 2),
    )

    stats_age_col5, stats_age_col6, stats_age_col7, stats_age_col8 = st.columns(4)
    stats_age_col5.metric(
        label="25%",
        value=df_describe.loc["quantity"].loc["25%"],
        delta=round(df_describe.loc["quantity"].loc["25%"] - df_describe_pre.loc["quantity"].loc["25%"], 2),
    )
    stats_age_col6.metric(
        label="75%",
        value=df_describe.loc["quantity"].loc["max"],
        delta=round(df_describe.loc["quantity"].loc["max"] - df_describe_pre.loc["quantity"].loc["max"], 2),
    )
    stats_age_col7.metric(
        label="Min",
        value=df_describe.loc["quantity"].loc["min"],
        delta=round(df_describe.loc["quantity"].loc["min"] - df_describe_pre.loc["quantity"].loc["min"], 2),
    )
    stats_age_col8.metric(
        label="Max",
        value=df_describe.loc["quantity"].loc["max"],
        delta=round(df_describe.loc["quantity"].loc["max"] - df_describe_pre.loc["quantity"].loc["max"], 2),
    )
with tab4:
    stats_age_col1, stats_age_col2, stats_age_col3, stats_age_col4 = st.columns(4)
    stats_age_col1.metric(
        label="Mean",
        value=df_describe.loc["discount"].loc["mean"],
        delta=round(df_describe.loc["discount"].loc["mean"] - df_describe_pre.loc["discount"].loc["mean"], 2),
    )
    stats_age_col2.metric(
        label="Median",
        value=df_describe.loc["discount"].loc["50%"],
        delta=round(df_describe.loc["discount"].loc["50%"] - df_describe_pre.loc["discount"].loc["50%"], 2),
    )
    stats_age_col3.metric(
        label="Std",
        value=df_describe.loc["discount"].loc["std"],
        delta=round(df_describe.loc["discount"].loc["std"] - df_describe_pre.loc["discount"].loc["std"], 2),
    )
    stats_age_col4.metric(
        label="Count",
        value=int(df_describe.loc["discount", "count"]),
        delta=round(df_describe.loc["discount"].loc["count"] - df_describe_pre.loc["discount"].loc["count"], 2),
    )

    stats_age_col5, stats_age_col6, stats_age_col7, stats_age_col8 = st.columns(4)
    stats_age_col5.metric(
        label="25%",
        value=df_describe.loc["discount"].loc["25%"],
        delta=round(df_describe.loc["discount"].loc["25%"] - df_describe_pre.loc["discount"].loc["25%"], 2),
    )
    stats_age_col6.metric(
        label="75%",
        value=df_describe.loc["discount"].loc["max"],
        delta=round(df_describe.loc["discount"].loc["max"] - df_describe_pre.loc["discount"].loc["max"], 2),
    )
    stats_age_col7.metric(
        label="Min",
        value=df_describe.loc["discount"].loc["min"],
        delta=round(df_describe.loc["discount"].loc["min"] - df_describe_pre.loc["discount"].loc["min"], 2),
    )
    stats_age_col8.metric(
        label="Max",
        value=df_describe.loc["discount"].loc["max"],
        delta=round(df_describe.loc["discount"].loc["max"] - df_describe_pre.loc["discount"].loc["max"], 2),
    )
with tab5:
    stats_age_col1, stats_age_col2, stats_age_col3, stats_age_col4 = st.columns(4)
    stats_age_col1.metric(
        label="Mean",
        value=df_describe.loc["shipping_cost"].loc["mean"],
        delta=round(df_describe.loc["shipping_cost"].loc["mean"] - df_describe_pre.loc["shipping_cost"].loc["mean"], 2),
    )
    stats_age_col2.metric(
        label="Median",
        value=df_describe.loc["shipping_cost"].loc["50%"],
        delta=round(df_describe.loc["shipping_cost"].loc["50%"] - df_describe_pre.loc["shipping_cost"].loc["50%"], 2),
    )
    stats_age_col3.metric(
        label="Std",
        value=df_describe.loc["shipping_cost"].loc["std"],
        delta=round(df_describe.loc["shipping_cost"].loc["std"] - df_describe_pre.loc["shipping_cost"].loc["std"], 2),
    )
    stats_age_col4.metric(
        label="Count",
        value=int(df_describe.loc["shipping_cost", "count"]),
        delta=round(df_describe.loc["shipping_cost"].loc["count"] - df_describe_pre.loc["shipping_cost"].loc["count"], 2),
    )

    stats_age_col5, stats_age_col6, stats_age_col7, stats_age_col8 = st.columns(4)
    stats_age_col5.metric(
        label="25%",
        value=df_describe.loc["shipping_cost"].loc["25%"],
        delta=round(df_describe.loc["shipping_cost"].loc["25%"] - df_describe_pre.loc["shipping_cost"].loc["25%"], 2),
    )
    stats_age_col6.metric(
        label="75%",
        value=df_describe.loc["shipping_cost"].loc["max"],
        delta=round(df_describe.loc["shipping_cost"].loc["max"] - df_describe_pre.loc["shipping_cost"].loc["max"], 2),
    )
    stats_age_col7.metric(
        label="Min",
        value=df_describe.loc["shipping_cost"].loc["min"],
        delta=round(df_describe.loc["shipping_cost"].loc["min"] - df_describe_pre.loc["shipping_cost"].loc["min"], 2),
    )
    stats_age_col8.metric(
        label="Max",
        value=df_describe.loc["shipping_cost"].loc["max"],
        delta=round(df_describe.loc["shipping_cost"].loc["max"] - df_describe_pre.loc["shipping_cost"].loc["max"], 2),
    )

# plot 2
tab1, tab2, tab3 = st.tabs(["Sales", "Quantity", "Profit"])
with tab1:
    df_sales_by_month = df[df["order_date"].dt.year == year_filter]
    df_sales_by_month['month_year'] = df_sales_by_month['order_date'].dt.strftime('%Y-%m')
    sales_by_month = df_sales_by_month.groupby('month_year')['sales'].sum().reset_index()
    fig = go.Figure(data=go.Scatter(x=sales_by_month['month_year'], y=sales_by_month['sales'], mode='lines', marker=dict(color='green')))
    fig.update_layout(
        title='Sales by Month',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Sales'),
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8)
    )
    st.plotly_chart(fig,use_container_width=True,height=800)
with tab2:
    df_sales_by_month = df[df["order_date"].dt.year == year_filter]
    df_sales_by_month['month_year'] = df_sales_by_month['order_date'].dt.strftime('%Y-%m')
    sales_by_month = df_sales_by_month.groupby('month_year')['quantity'].sum().reset_index()
    fig = go.Figure(data=go.Scatter(x=sales_by_month['month_year'], y=sales_by_month['quantity'], mode='lines', marker=dict(color='blue')))
    fig.update_layout(
        title='Quantity by Month',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Sales'),
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8)
    )
    st.plotly_chart(fig,use_container_width=True,height=800)
with tab3:
    df_sales_by_month = df[df["order_date"].dt.year == year_filter]
    df_sales_by_month['month_year'] = df_sales_by_month['order_date'].dt.strftime('%Y-%m')
    sales_by_month = df_sales_by_month.groupby('month_year')['profit'].sum().reset_index()
    fig = go.Figure(data=go.Scatter(x=sales_by_month['month_year'], y=sales_by_month['profit'], mode='lines', marker=dict(color='purple')))
    fig.update_layout(
        title='Profit by Month',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Sales'),
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8)
    )
    st.plotly_chart(fig,use_container_width=True,height=800)

# country
tab1, tab2, tab3 = st.tabs(["Sales", "Quantity", "Profit"])
with tab1:
    filtered_df = df[df['order_date'].dt.year == year_filter]
    grouped_df = filtered_df.groupby('country')['sales'].sum()
    top_10_countries = grouped_df.nlargest(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_10_countries.index,
        y=top_10_countries.values,
        marker=dict(color=px.colors.qualitative.Plotly),  # Using Plotly's qualitative color palette
    ))
    fig.update_layout(
        title='Top 10 Countries with Highest Sales',
        xaxis=dict(title='Country'),
        yaxis=dict(title='Profit'),
    )
    st.plotly_chart(fig,use_container_width=True,height=800)
with tab2:
    filtered_df = df[df['order_date'].dt.year == year_filter]
    grouped_df = filtered_df.groupby('country')['quantity'].sum()
    top_10_countries = grouped_df.nlargest(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_10_countries.index,
        y=top_10_countries.values,
        marker=dict(color=px.colors.qualitative.Plotly),  # Using Plotly's qualitative color palette
    ))
    fig.update_layout(
        title='Top 10 Countries with Highest Quantity',
        xaxis=dict(title='Country'),
        yaxis=dict(title='Quantity'),
    )
    st.plotly_chart(fig,use_container_width=True,height=800)
with tab3:
    filtered_df = df[df['order_date'].dt.year == year_filter]
    grouped_df = filtered_df.groupby('country')['profit'].sum()
    top_10_countries = grouped_df.nlargest(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_10_countries.index,
        y=top_10_countries.values,
        marker=dict(color=px.colors.qualitative.Plotly),  # Using Plotly's qualitative color palette
    ))
    fig.update_layout(
        title='Top 10 Countries with Highest Profit',
        xaxis=dict(title='Country'),
        yaxis=dict(title='Profit'),
    )
    st.plotly_chart(fig,use_container_width=True,height=800)

# plot 3

st.markdown("## 2. Phân bố số lượng đơn hàng")

df_market_pie = df.groupby(df['market'])['order_id'].count()
fig_col1, fig_col2 = st.columns([5, 5])
with fig_col1:
    category_by_profit = df.groupby(['category', 'sub_category'])['profit'].sum().reset_index()
    category_by_profit.sort_values('profit', ascending=False, inplace=True)
    fig = go.Figure()
    for category in category_by_profit['category'].unique():
        data = category_by_profit[category_by_profit['category'] == category]
        fig.add_trace(go.Bar(
            x=data['sub_category'],
            y=data['profit'],
            name=category
        ))
    fig.update_layout(
        title='Profit by Sub-Category within Each Category',
        xaxis=dict(title='Sub-Category'),
        yaxis=dict(title='Profit'),
        barmode='group',
        legend_title='Category'
    )
    st.plotly_chart(fig,use_container_width=True,height=800)
with fig_col2:
    category_by_count = df.groupby(['category', 'sub_category']).count()['order_id'].reset_index()
    category_by_count.sort_values('order_id', ascending=False, inplace=True)
    fig = go.Figure()
    for category in category_by_count['category'].unique():
        data = category_by_count[category_by_count['category'] == category]
        fig.add_trace(go.Bar(
            x=data['sub_category'],
            y=data['order_id'],
            name=category
        ))
    fig.update_layout(
        title='Order Count by Sub-Category within Each Category',
        xaxis=dict(title='Sub-Category'),
        yaxis=dict(title='Order Count'),
        barmode='group',
        legend_title='Category'
    )
    st.plotly_chart(fig,use_container_width=True,height=800)

# plot 5
st.markdown("## 3. Top 10")
tab1, tab2, tab3 = st.tabs(["Top 10 sales product", "Top 10 quantity product", "Top 10 customers"])
with tab1:
    products_sales = pd.DataFrame(df.groupby('product_name')["sales"].sum())
    products_sales = products_sales.nlargest(columns="sales", n=10)
    st.dataframe(products_sales)
with tab2:
    products_by_quantity = pd.DataFrame(df.groupby('product_name')['quantity'].sum())
    products_by_quantity_sorted = products_by_quantity.nlargest(columns="quantity", n=10)
    st.dataframe(products_by_quantity_sorted)
with tab3:
    top_10_customers = pd.DataFrame(df.groupby('customer_name')["order_id"].count())
    top_10_customers = top_10_customers.nlargest(columns="order_id", n=10)
    st.dataframe(top_10_customers)
