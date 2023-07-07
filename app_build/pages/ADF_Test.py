import streamlit as st

st.markdown("<h1 style='text-align: center; color: #B799FF;'>ADF TEST</h1>", unsafe_allow_html=True)
st.markdown("## ADF Test của từng loại Category")
tab1, tab2, tab3 = st.tabs(["Furniture", "Office Supplies", "Technology"])
with tab1:
    st.info("""**Augmented Dickey-Fuller Test:**
- ADF test statistic:       -0.340857
- p-value:                  0.919515
- lags used:                2.000000
- observations:             33.000000
- critical value (1%):      -3.646135
- critical value (5%):      -2.954127
- critical value (10%):     -2.615968
- Weak evidence against the null hypothesis.
- Fail to reject the null hypothesis.
- Data has a unit root and is non-stationary.
    """, icon="📝")
with tab2:
    st.info("""**Augmented Dickey-Fuller Test:**
- ADF test statistic:       -0.535314
- p-value:                  0.884939
- lags used:                2.000000
- observations:             33.000000
- critical value (1%):      -3.646135
- critical value (5%):      -2.954127
- critical value (10%):     -2.615968
- Weak evidence against the null hypothesis.
- Fail to reject the null hypothesis.
- Data has a unit root and is non-stationary.
    """, icon="📝")
with tab3:
    st.info("""**Augmented Dickey-Fuller Test:**
- ADF test statistic:       -0.146793
- p-value:                  0.944574
- lags used:                8.000000
- observations:             27.000000
- critical value (1%):      -3.699608
- critical value (5%):      -2.976430
- critical value (10%):     -2.627601
- Weak evidence against the null hypothesis.
- Fail to reject the null hypothesis.
- Data has a unit root and is non-stationary.
    """, icon="📝")

st.info("""**Explain:**
- In statistics, an augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity.
""", icon="📝")