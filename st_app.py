# import libraries
import streamlit as st
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error, r2_score

# define a plot function
def chart_plot(mod_name,y_pred, number_samples=200):
    chart_data = pd.DataFrame()
    chart_data['y_true'] = y_test[:number_samples]
    chart_data[mod_name] = y_pred[:number_samples]
    st.line_chart(chart_data)


# define side bar to interact with users
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Select a Model", ["Linear Regression", "Random Forest", "Comparison"])
st.sidebar.title("Number of Samples")
number_samples = st.sidebar.slider('#Visuliazation Samples', 10, 1000)

# load related data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
df = pd.read_csv('data/kc_house_data.csv', parse_dates=['date'])
df_new = df[['lat', 'long']]
df_new.columns = ['lat', 'lon']
st.map(df_new)

st.write('\n')

# plot LR model result
if model_name == "Linear Regression":
    model_lr = load('model/regr.joblib')
    y_pred_lr = model_lr.predict(X_test)
    st.write( 'R2 Error of ', model_name, 'is ', round(r2_score(y_test, y_pred_lr), 3))
    chart_plot(model_name, y_pred_lr, number_samples)
# plot RF model result    
elif model_name == "Random Forest":
    model_rf = load('model/rf.joblib')
    y_pred_rf = model_rf.predict(X_test)
    st.write( 'R2 Error of ', model_name, 'is ', round(r2_score(y_test, y_pred_rf), 3))
    chart_plot(model_name, y_pred_rf, number_samples)
# compare models
else:
    model_lr = load('model/regr.joblib')
    y_pred_lr = model_lr.predict(X_test)
    model_rf = load('model/rf.joblib')
    y_pred_rf = model_rf.predict(X_test)
    st.write( 'R2 Error of ', "Linear Regression", 'is ', round(r2_score(y_test, y_pred_lr), 3))
    st.write( 'R2 Error of ', "Random Forest", 'is ', round(r2_score(y_test, y_pred_rf), 3))
    chart_data = pd.DataFrame()
    chart_data['y_true'] = y_test[:number_samples]
    chart_data['Linear Regression'] = y_pred_lr[:number_samples]
    chart_data['Random Forest'] = y_pred_rf[:number_samples]
    st.line_chart(chart_data)
