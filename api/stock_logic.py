import os.path
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def perform_prediction(start_date: str, end_date: str, symbol: str):
    """Perform hourly prediction for a stock symbol."""
    data_path = os.path.join('hourlyExports', 'datas', f'{symbol}-data.csv')
    scaler_path = os.path.join('hourlyExports', 'scalers', f'{symbol}-scaler.pkl')
    df = pd.read_csv(data_path)
    df = df.reset_index()
    trainSet = df.iloc[:, 1:2].values

    with open(scaler_path, 'rb') as f:
        trainingSetScaledSaved = pickle.load(f)

    testDF = yf.download(symbol, interval='1h', start=end_date, end=datetime.now())
    realSP = testDF['Close'].values
    dfTotal = pd.concat((df['Open'], testDF['Open']), axis=0)
    modelInp = dfTotal[len(dfTotal) - len(testDF) - 60:].values
    modelInp = modelInp.reshape(-1, 1)
    modelInp = trainingSetScaledSaved.transform(modelInp)
    realData = [modelInp[len(modelInp) - 60:len(modelInp + 1), 0]]
    realData = np.array(realData)
    realData = np.reshape(realData, newshape=(realData.shape[0], realData.shape[1], 1))
    predic = prediction_function(symbol, realData, 'hourlyExports')
    prediction_scaled = trainingSetScaledSaved.inverse_transform(predic)
    return prediction_scaled


def prediction_function(symbol, realData, folderName: str):
    """Run TensorFlow Lite model for prediction."""
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(folderName, f'{symbol}.tflite'))
    interpreter.allocate_tensors()

    # Get input and output details from the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    input_shape = input_details[0]['shape']
    input_data = np.array(realData, dtype=np.float32)

    # Set the input tensor and run the inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensor and process the predictions
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data
    return predictions


def change_positive_negative(ltp, previous_close):
    """Determine if price change is positive, negative or neutral."""
    change = (ltp - previous_close)
    print(change)
    print(previous_close)
    if (change > 0):
        return "POSITIVE"
    elif (change < 0):
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def get_stock_name_from_csv(symbol, exchange='BSE'):
    """Get stock name from CSV file based on symbol and exchange."""
    try:
        if exchange == 'BSE':
            stock = symbol.split(".", 1)[0]
            df = pd.read_csv('equity_bse.csv')
            result = df.loc[df["SYMBOL"] == stock, "STOCK"]
            return result.values[0]
        elif exchange == 'NSE':
            stock = symbol.split(".", 1)[0]
            df = pd.read_csv('equity_nse.csv')
            result = df.loc[df["SYMBOL"] == stock, "COMPANY"]
            return result.values[0]
    except:
        return symbol


def get_comprehensive_stock_data(symbol):
    """Get comprehensive stock data including BSE/NSE details."""
    start_date = "2022-09-01"
    
    try:
        if symbol.split('.')[1] == "NS":
            # NSE data
            dfNSE = pd.read_csv('equity_nse.csv')
            resultNSE = dfNSE.loc[dfNSE["SYMBOL"] == symbol.split('.')[0], ["COMPANY", "ISIN", "FACE_VALUE"]]
            stockISIN = resultNSE["ISIN"].values[0]
            stockFV = int(resultNSE["FACE_VALUE"].values[0])
            stockCompany = resultNSE["COMPANY"].values[0]
            try:
                df = pd.read_csv('equity_bse.csv')
                result = df.loc[df["SYMBOL"] == symbol.split('.')[0], ["IndustryNew"]]
                stockIndustry = result["IndustryNew"].values[0]
            except:
                stockIndustry = ""
        else:
            if symbol.split('.')[1] == "BO":
                # BSE data
                df = pd.read_csv('equity_bse.csv')
                result = df.loc[df["SYMBOL"] == symbol.split('.')[0], ["FaceValue", "ISIN", "IndustryNew", "STOCK"]]
                stockIndustry = result["IndustryNew"].values[0]
                stockISIN = result["ISIN"].values[0]
                stockFV = result["FaceValue"].values[0]
                stockCompany = result["STOCK"].values[0]
    except:
        stockIndustry = "na"
        stockISIN = "na"
        stockFV = "na"
        stockCompany = symbol

    data = yf.download(symbol, interval='1d', start=start_date)
    
    return {
        "isin": stockISIN,
        "indusry": stockIndustry,  # Note: keeping original typo for consistency
        "face_value": stockFV,
        "company": stockCompany,
        "W52High": data['High'].max(),
        "W52Low": data['Low'].min(),
        "dayLow": data['Low'][-1],
        "dayHigh": data['High'][-1],
        "dayOpen": data["Open"].iloc[-1],
        "t1": data['Close'].iloc[-2],
        "d1": data.index[-2].strftime('%b %d, %Y'),
        "t2": data['Close'].iloc[-3],
        "d2": data.index[-3].strftime('%b %d, %Y'),
        "t3": data['Close'].iloc[-4],
        "d3": data.index[-4].strftime('%b %d, %Y'),
        "t4": data['Close'].iloc[-5],
        "d4": data.index[-5].strftime('%b %d, %Y'),
        "t5": data['Close'].iloc[-6],
        "d5": data.index[-6].strftime('%b %d, %Y'),
        "t6": data['Close'].iloc[-7],
        "d6": data.index[-7].strftime('%b %d, %Y')
    }


def get_technical_indicators(symbol):
    """Calculate technical indicators for a stock."""
    stock = symbol.split(".", 1)[0]
    df = pd.read_csv('equity_bse.csv')
    result = df.loc[df["SYMBOL"] == stock, ["FaceValue", "ISIN", "IndustryNew"]]

    data = yf.download(symbol, period="1y")
    sma50 = data['Close'].rolling(window=50).mean().iloc[-1]
    ema50 = data['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    sma100 = data['Close'].rolling(window=100).mean().iloc[-1]
    ema100 = data['Close'].ewm(span=100, adjust=False).mean().iloc[-1]
    sma200 = data['Close'].rolling(window=200).mean().iloc[-1]
    ema200 = data['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    rsi = 100 - (100 / (1 + (data['Close'].diff(1).fillna(0) > 0).rolling(window=14).mean())).iloc[-1]
    macd = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    bollingerBandUpper = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
    bollingerBandLower = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    atr = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()

    return {
        "faceValue": result["FaceValue"].values[0],
        "ISIN": result["ISIN"].values[0],
        "industry": result["IndustryNew"].values[0],
        "sma50": sma50,
        "ema50": ema50,
        "sma100": sma100,
        "ema100": ema100,
        "sma200": sma200,
        "ema200": ema200,
        "rsi": rsi,
        "macd": macd.iloc[-1],
        "bollingerBandUpper": bollingerBandUpper.iloc[-1],
        "bollingerBankLoweer": bollingerBandLower.iloc[-1],  # Note: keeping original typo for consistency
        "atr": atr.iloc[-1]
    }


def get_daily_prediction(symbol):
    """Get daily prediction for a stock."""
    end_date = '2023-01-20'
    df = yf.download(symbol, period='max', end=end_date)
    df = df.reset_index()
    trainSet = df.iloc[:, 1:2].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    trainingSetScaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    testDF = yf.download(symbol, period='max', start=end_date, end=datetime.now())
    realSP = testDF['Close'].values
    dfTotal = pd.concat((df['Open'], testDF['Open']), axis=0)
    modelInp = dfTotal[len(dfTotal) - len(testDF) - 60:].values
    modelInp = modelInp.reshape(-1, 1)
    modelInp = scaler.transform(modelInp)
    realData = [modelInp[len(modelInp) - 60:len(modelInp + 1), 0]]
    realData = np.array(realData)
    realData = np.reshape(realData, newshape=(realData.shape[0], realData.shape[1], 1))
    predic = prediction_function(symbol, realData, 'exports')
    prediction_scaled = scaler.inverse_transform(predic)
    
    return {
        "predicted_close": float(prediction_scaled[0][0])
    }