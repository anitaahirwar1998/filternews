import datetime
import gspread
from gspread_dataframe import set_with_dataframe
import pandas as pd
import yfinance as yf
from oauth2client.service_account import ServiceAccountCredentials
creds = ServiceAccountCredentials.from_json_keyfile_name('./gssep-399015-22a26cd898e7.json')
client = gspread.authorize(creds)
gsselect = client.open('News Database Sep 2023')
sheetsel_stock = gsselect.worksheet('SelectedStock')
sel_stk_all_records = sheetsel_stock.get_all_records()
stk_raw_df=pd.DataFrame(sel_stk_all_records)
def stock_history(name, days=1):
    days = str(days) + "d"
    name=str(name)
    symbol = name + ".NS"
    try:
        data =yf.download(symbol, period=days, progress=False)
        data1=yf.download(symbol, period='365d', progress=False)
        high=round(max(data1["High"]),2)
        low=round(min(data1["Low"]),2)
        if data.empty:
            print(f"No data available for {symbol}")
            return None
    except Exception as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    filtered_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    filtered_data['Tag'] = name
    filtered_data["52W_high"]=high
    filtered_data["52W_low"]=low
    final = filtered_data[['Tag', 'Open', 'Low', 'High', 'Close',"52W_high","52W_low",'Volume']].reset_index()
    return final
symbol_list =list(stk_raw_df['Tag'])
final = pd.DataFrame(columns=['Tag', 'Open', 'Low', 'High', 'Close',"52W_high","52W_low",'Volume'])
for x in symbol_list:
    df = stock_history(x)
    if df is not None:
        final = pd.concat([final, df])
sorted_final=final.sort_values(by='Volume',ascending=False)
merge_df=sorted_final.merge(stk_raw_df,on='Tag',how="left")
merge_df["Date"]=merge_df["Date"].dt.date
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
def predict_stock_rise_30(name, days=30):
    days = str(days) + "d"
    symbol = name + ".NS"
    try:
        data = yf.download(symbol, period=days, progress=False)
    except KeyError as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print(f"No data available for {symbol}")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Symbol'] = name

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram'] = df['MACD'] - df['Signal Line']
    df['Signal'] = df['MACD Histogram'].apply(lambda x: 1 if x > 0 else -1)
    filtered_df = df[df['Signal'] != 0]

    df['MA'] = ta.sma(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    bbands = ta.bbands(df['Close'], length=20)
    if bbands is None:
        print("Failed to calculate Bollinger Bands")
        return None
    df['BB Upper'] = bbands['BBL_20_2.0']
    df['BB Middle'] = bbands['BBM_20_2.0']
    df['BB Lower'] = bbands['BBU_20_2.0']
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    high = df['High'].max()
    low = df['Low'].min()
    fib_0 = high
    fib_0382 = high - 0.382 * (high - low)
    fib_0618 = high - 0.618 * (high - low)
    fib_1 = low

    df['Fib 0.0'] = fib_0
    df['Fib 0.382'] = fib_0382
    df['Fib 0.618'] = fib_0618
    df['Fib 1.0'] = fib_1

    ma_condition = df['Close'] > df['MA']
    rsi_condition = df['RSI'] > 50
    bb_condition = df['Close'] > df['BB Upper']
    adx_condition = df['ADX'] > 25
    obv_condition = df['OBV'] > 0
    fib_condition = df['Close'] > df['Fib 0.618']

    if (
        ma_condition.iloc[-1]
        and rsi_condition.iloc[-1]
        and bb_condition.iloc[-1]
        and adx_condition.iloc[-1]
        and obv_condition.iloc[-1]
        and fib_condition.iloc[-1]
    ):
        return "BULLISH"
    else:
        return "BEARISH"
def predict_stock_rise_200(name, days=200):
    days = str(days) + "d"
    symbol = name + ".NS"

    try:
        data = yf.download(symbol, period=days, progress=False)
    except KeyError as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print(f"No data available for {symbol}")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Symbol'] = name

    long_term_ema = 200  # Define the long-term EMA period
    df['EMA_long'] = df['Close'].ewm(span=long_term_ema, adjust=False).mean()
    df['MACD_long'] = df['Close'] - df['EMA_long']
    df['Signal Line_long'] = df['MACD_long'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram_long'] = df['MACD_long'] - df['Signal Line_long']
    df['Signal_long'] = df['MACD Histogram_long'].apply(lambda x: 1 if x > 0 else -1)

    df['EMA_short'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['MACD_short'] = df['Close'] - df['EMA_short']
    df['Signal Line_short'] = df['MACD_short'].ewm(span=9, adjust=False).mean()
    df['MACD Histogram_short'] = df['MACD_short'] - df['Signal Line_short']
    df['Signal_short'] = df['MACD Histogram_short'].apply(lambda x: 1 if x > 0 else -1)

    df['MA'] = ta.sma(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    bbands = ta.bbands(df['Close'], length=20)

    if bbands is None:
        print("Failed to calculate Bollinger Bands")
        return None

    df['BB Upper'] = bbands['BBL_20_2.0']
    df['BB Middle'] = bbands['BBM_20_2.0']
    df['BB Lower'] = bbands['BBU_20_2.0']

    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    high = df['High'].max()
    low = df['Low'].min()
    fib_0 = high
    fib_0382 = high - 0.382 * (high - low)
    fib_0618 = high - 0.618 * (high - low)
    fib_1 = low

    df['Fib 0.0'] = fib_0
    df['Fib 0.382'] = fib_0382
    df['Fib 0.618'] = fib_0618
    df['Fib 1.0'] = fib_1

    ma_condition = df['Close'] > df['MA']
    rsi_condition = df['RSI'] > 50
    bb_condition = df['Close'] > df['BB Upper']
    adx_condition = df['ADX'] > 25
    obv_condition = df['OBV'] > 0
    fib_condition = df['Close'] > df['Fib 0.618']

    if (
        ma_condition.iloc[-1]
        and rsi_condition.iloc[-1]
        and bb_condition.iloc[-1]
        and adx_condition.iloc[-1]
        and obv_condition.iloc[-1]
        and fib_condition.iloc[-1]
        and df['Signal_long'].iloc[-1] == 1
        and df['Signal_short'].iloc[-1] == 1
    ):
        return "BULLISH"
    else:
        return "BEARISH"
def expected_gain(name, days=14):
    days = str(days) + "d"
    symbol = name + ".NS"

    try:
        data = yf.download(symbol, period=days, progress=False)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print("No data available for the specified period.")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Symbol'] = name

    df['Gain'] = df['High'] - df['Open']
    average_gain = df['Gain'].tail(7).mean()
    previous_close = df['Close'].iloc[-2]
    average_gain_percentage=(average_gain / previous_close) * 100
    return round(average_gain_percentage,2)
def recent_drop(name, days=3):
    days = str(days) + "d"
    symbol = name + ".NS"

    try:
        data = yf.download(symbol, period=days, progress=False)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print("No data available for the specified period.")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Symbol'] = name
    df['drop'] = df['Open'] - df['Close']
    close_price_drop = any(df['drop'].tail(3)>0)
    return close_price_drop
def expected_loss(name, days=14):
    days = str(days) + "d"
    symbol = name + ".NS"

    try:
        data = yf.download(symbol, period=days, progress=False)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print("No data available for the specified period.")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Symbol'] = name

    df['Loss'] = df['Open'] - df['Low']
    average_loss = df['Loss'].tail(7).mean()
    previous_close = df['Close'].iloc[-2]
    average_loss_percentage = (average_loss / previous_close) * 100

    return average_loss_percentage
def convert_to_float(value):
    value=str(value)
    value = value.replace(',', '')
    try:
        return float(value)
    except ValueError:
        return None
merge_df['Close']=merge_df['Close'].apply(convert_to_float)
merge_df['Signal_30']=merge_df['Tag'].apply(predict_stock_rise_30)
merge_df['Gain']=merge_df['Tag'].apply(expected_gain)
final=merge_df.loc[merge_df['Gain']>0].reset_index(drop=True)
final['Down']=final['Tag'].apply(expected_loss)
final["Expected Down"]=round(((final["Close"]*(100-final["Down"]))/100),2)
final["Expected Up"]=round(((final["Close"]*(final["Gain"]+100))/100),2)
final=final.sort_values(by="Gain",ascending=False,ignore_index=True).copy()
final["Signal_200"]=final['Tag'].apply(predict_stock_rise_200)
final["Recent_Drop"]=final['Tag'].apply(recent_drop)
def calculate_return(name,days=30):
    import yfinance as yf
    import numpy as np
    symbol = name + ".NS"
    days = str(days) + "d"
    try:
        data = yf.download(symbol, period=days, progress=False)
    except Exception as e:
        print(f"Failed to download data for {symbol}: {str(e)}")
        return None

    if data.empty:
        print("No data available for the specified stock.")
        return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    open_pr=df["Open"][0]
    close_pr=df["Close"][-1]
    mr=(close_pr-open_pr)*100/open_pr
    return round(mr,2)
final['Month Return'] = final["Tag"].apply(lambda symbol: calculate_return(symbol, days=30))
final['Week Return'] = final["Tag"].apply(lambda symbol: calculate_return(symbol, days=7))
sorted_all=final[[ 'Date','Tag',"Open",'Close',"52W_high","52W_low",'Gain',"Down","Month Return","Week Return","Recent_Drop",'URL',"Signal_30","Signal_200"]].copy()
out_gs = client.open('News Database Sep 2023')
out_sheet=out_gs.worksheet("Final")
out_sheet.clear()
#save data
set_with_dataframe(out_sheet, sorted_all)
