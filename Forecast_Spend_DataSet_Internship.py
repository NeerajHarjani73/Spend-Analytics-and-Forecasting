import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from fbprophet import Prophet
import warnings
warnings.filterwarnings("ignore") 
#pip install dash
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.options.display.float_format = '{:,.2f}'.format
df = pd.read_csv('Spend_DataSet_Internship_Enriched_Version2.csv',index_col=0)
df['INVOICE_DATE'] = pd.to_datetime(df['INVOICE_DATE'],infer_datetime_format=True)
df['Month'] = pd.DatetimeIndex(df['INVOICE_DATE']).month
df['Year'] = pd.DatetimeIndex(df['INVOICE_DATE']).year
df['Date']= pd.to_datetime(df.Year.astype(str) + '/' + df.Month.astype(str)).dt.strftime('%Y-%m')

def initialize(df):
    global predictions
    predictions  = pd.DataFrame()
    user_input = input("Do you want to forecast by CATEGORY, VENDOR or PRODUCT? Select one: ")
    user_input = user_input.upper()
    user_selection = input('Type down your specific' + ' ' + user_input + ' in the same format' + ':')
    dataset = df[['INV_AMOUNT_USD',user_input,'Date']]
    dataset = dataset.groupby([user_input,'Date']).sum().reset_index()
    sample = dataset.loc[dataset[user_input]==user_selection][['INV_AMOUNT_USD','Date']]
    FBProphet(sample)
    AR(sample)
    ARIMA(sample)
    expo_smooth(sample)
    
#Prophet model
def FBProphet(timeseries):
    ts = timeseries.copy()
    ts = ts.rename(columns={'Date': 'ds', 'INV_AMOUNT_USD': 'y'})
    ts_model = Prophet(growth='linear',changepoint_prior_scale=0.5,seasonality_mode='additive',interval_width=0.95)
    ts_model.add_country_holidays(country_name='US')
    ts_model.fit(ts)
    ts_forecast = ts_model.make_future_dataframe(periods=12, freq='MS')
    ts_forecast = ts_model.predict(ts_forecast)
    #timeseries_model.plot(timeseries_forecast, xlabel = 'Date', ylabel = 'Invoice Amount')
    #plt.title('Prediction for next 12 months')
    prophet_forecast=ts_forecast.yhat[-12:]
    prophet_forecast = np.array(prophet_forecast)
    prophet_error = round(sqrt(mean_squared_error(ts.y[-12:],ts_forecast.yhat[-12:])))
    predictions['FBProphet Model'+', '+'Error(RMSE):'+str(prophet_error)] = prophet_forecast

#AR Model
def AR(timeseries):   
    ts = timeseries.copy()
    ts.set_index('Date', inplace=True)
    ts_diff = ts.diff(periods=1)
    #integrated od order 1 (d =1)
    ts_diff = ts_diff[1:]
    #from statsmodels.graphics.tsaplots import plot_acf
    #plot_acf(timeseries)
    #plot_acf(timeseries_diff)
    #timeseries_diff.plot()   
    from statsmodels.tsa.ar_model import AR
    model_ar = AR(ts.values)
    model_ar_fit = model_ar.fit()
    AR_predictions = model_ar_fit.predict(start=len(ts),end=len(ts)+12)
    AR_predictions = AR_predictions[1:]
    AR_predictions = np.array(AR_predictions)
    AR_error = round(sqrt(mean_squared_error(ts.INV_AMOUNT_USD[-12:],AR_predictions)))
    predictions['AutoRegressive Model'+', '+'Error(RMSE):'+str(AR_error)] = AR_predictions
    
#ARIMA 
def ARIMA(timeseries):
    ts = timeseries.copy()
    ts.set_index('Date', inplace=True)
    from statsmodels.tsa.arima_model import ARIMA
    import itertools
    p=d=q=range(0,5)
    pdq = list(itertools.product(p,d,q))
    parameters =[]
    for param in pdq:
        try:
            import warnings
            warnings.filterwarnings("ignore") 
            model_arima = ARIMA(ts.values,order=param)
            model_arima_fit = model_arima.fit()
            print('ARIMA{} - AIC:{}'.format(param,model_arima_fit.aic))
        except:
            continue
        aic = model_arima_fit.aic
        parameters.append([param,aic])
    parameters = pd.DataFrame(parameters)
    parameters.columns = ['parameters','aic']
    min_aic = min(parameters.aic)
    best_param = parameters.loc[parameters.aic == min_aic]['parameters']
    model_arima = ARIMA(ts.values,order=best_param.values[0])
    model_arima_fit = model_arima.fit()
    ARIMA_predictions = model_arima_fit.forecast(steps=12)[0]
    ARIMA_error = round(sqrt(mean_squared_error(ts.INV_AMOUNT_USD[-12:],ARIMA_predictions)))
    predictions['ARIMA Model'+', '+'Error(RMSE):'+str(ARIMA_error)] = ARIMA_predictions

def expo_smooth(timeseries):
    ts = timeseries.copy()
    ts.set_index('Date', inplace=True)
    import statsmodels.tsa.holtwinters as ets
    #Brown Simple Exponential Smoothing
    model_es = ets.ExponentialSmoothing(ts.values, trend='additive',damped=False, seasonal=None)
    model_es_fit = model_es.fit()
    ES_predictions = model_es_fit.forecast(steps=12)
    ES_error = round(sqrt(mean_squared_error(ts.INV_AMOUNT_USD[-12:],ES_predictions)))
    predictions['Double Exponential Smoothing Model'+', '+'Error(RMSE):'+str(ES_error)] = ES_predictions

#MA model
#def MA(timeseries):   
    #global MA_predictions
    #timeseries.set_index('Date', inplace=True)
    #timeseries_diff = timeseries.diff(periods=1)
    #integrated od order 1 (d =1)
    #timeseries_diff = timeseries_diff[1:]
    #from statsmodels.graphics.tsaplots import plot_acf
    #plot_acf(timeseries)
    #plot_acf(timeseries_diff)
    #timeseries_diff.plot()   
    #from statsmodels.tsa.arima_model import ARMA
    #model_ma = ARMA(timeseries.values, order=(0,1))
    #model_ma_fit = model_ma.fit()
    #MA_predictions = model_ma_fit.predict(start=24,end=36)
    #MA_predictions = MA_predictions[1:]
    #MA_predictions = np.array(MA_predictions)
    #error(timeseries.INV_AMOUNT_USD[-12:],AR_predictions)
    #create_df(AR_predictions)  