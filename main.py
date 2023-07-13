import os.path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)


@app.get("/")
async def root():
    response= {
        "message": "stockSense API backend",
        "version_info": "1.3.0"
    }
    return response


@app.get("/query/{symbol}")
async def query(symbol: str):
    ticker = yf.Ticker(symbol)
    stock_info = ticker.info
    stock_name = stock_info.get('longName')
    data= yf.download(symbol, interval='1d')
    JSONresponse= {
        "stock_name": stock_name,
        "t1": data['Close'].iloc[-1],
        "t2": data['Close'].iloc[-2],
        "t3": data['Close'].iloc[-3],
        "t4": data['Close'].iloc[-4],
        "t5": data['Close'].iloc[-5],
        "t6": data['Close'].iloc[-6]
    }
    return JSONresponse


@app.get("/info/{symbol}")
async def info(symbol: str):
    ticker = yf.Ticker(symbol)
    stock_info = ticker.info
    stock_name = stock_info.get('longName')
    quote_type = stock_info.get('quoteType')

    JSON = {
        "symbol": symbol,
        "stock_name": stock_name,
        "quote_type": quote_type
    }

    return JSON


@app.get("/ltp/{symbol}")
async def ltp(symbol):
    data= yf.download(symbol, interval='1m', period='1d')
    ltp= (data['Close'].iloc[-1])
    previous_close= data['Close'].iloc[0]
    response={
        "ltp": round(ltp, 2),
        "change": changePositiveNegative(ltp=ltp, previous_close=previous_close)
    }
    return response


@app.get("/prediction/{symbol}")
async def prediction(symbol: str):
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
    predic = predictionFunction(symbol, realData)
    prediction_scaled = scaler.inverse_transform(predic)
    jsonData = {
        "predicted_close": float(prediction_scaled[0][0])
    }
    return jsonData


def predictionFunction(symbol, realData):
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join('exports', f'{symbol}.tflite'))
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


def changePositiveNegative(ltp, previous_close):
    change= (ltp-previous_close)
    if(change>0):
        return "POSITIVE"
    elif(change<0):
        return "NEGATIVE"
    else:
        return "NEUTRAL"
