from datetime import datetime
import yfinance as yf
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .stock_logic import (
    perform_prediction, 
    change_positive_negative, 
    get_stock_name_from_csv,
    get_comprehensive_stock_data,
    get_technical_indicators,
    get_daily_prediction
)


@api_view(['GET'])
def root(request):
    """Root endpoint with API information."""
    response = {
        "message": "stockSense API backend",
        "version_info": "2.1.7",
        "changelog": "added stockInfo details and new function i.e. queryV3"
    }
    return Response(response)


@api_view(['GET'])
def query(request, symbol):
    """Basic stock query endpoint."""
    try:
        stock_name = get_stock_name_from_csv(symbol, 'BSE')
    except:
        stock_name = symbol

    data = yf.download(symbol, interval='1d')
    json_response = {
        "stock_name": stock_name,
        "t1": data['Close'].iloc[-2],
        "t2": data['Close'].iloc[-3],
        "t3": data['Close'].iloc[-4],
        "t4": data['Close'].iloc[-5],
        "t5": data['Close'].iloc[-6],
        "t6": data['Close'].iloc[-7]
    }
    return Response(json_response)


@api_view(['GET'])
def query_v2(request, symbol):
    """Stock query with dates."""
    try:
        stock_name = get_stock_name_from_csv(symbol, 'BSE')
    except:
        stock_name = symbol

    data = yf.download(symbol, interval='1d')
    print(data.tail())
    json_response = {
        "stock_name": stock_name,
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
    return Response(json_response)


@api_view(['GET'])
def query_v3(request, symbol):
    """Comprehensive stock query with BSE/NSE data."""
    try:
        response_data = get_comprehensive_stock_data(symbol)
        return Response(response_data)
    except Exception as e:
        return Response(
            {"error": f"Failed to fetch stock data: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def info(request, symbol):
    """Stock information using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        stock_info = ticker.info
        stock_name = stock_info.get('longName')
        quote_type = stock_info.get('quoteType')

        json_response = {
            "symbol": symbol,
            "stock_name": stock_name,
            "quote_type": quote_type
        }
        return Response(json_response)
    except Exception as e:
        return Response(
            {"error": f"Failed to fetch stock info: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def ltp(request, symbol):
    """Last traded price endpoint."""
    try:
        data = yf.download(symbol, interval='1m', period='1d')
        ltp_value = data['Close'].iloc[-1]
        previous_close = yf.download(symbol, interval='1d')['Close'].iloc[-1]
        response_data = {
            "ltp": round(ltp_value, 2),
            "change": change_positive_negative(ltp=ltp_value, previous_close=previous_close)
        }
        return Response(response_data)
    except Exception as e:
        return Response(
            {"error": f"Failed to fetch LTP: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def technicals(request, symbol):
    """Technical indicators endpoint."""
    try:
        response_data = get_technical_indicators(symbol)
        return Response(response_data)
    except Exception as e:
        return Response(
            {"error": f"Failed to fetch technical indicators: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def prediction(request, symbol):
    """Daily prediction endpoint."""
    try:
        response_data = get_daily_prediction(symbol)
        return Response(response_data)
    except Exception as e:
        return Response(
            {"error": f"Failed to get prediction: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def hourly_prediction(request, symbol):
    """Hourly prediction endpoint."""
    try:
        prediction_result = perform_prediction('2021-08-25', '2023-06-01', symbol)
        json_data = {
            "predicted_close": float(prediction_result[0][0])
        }
        return Response(json_data)
    except Exception as e:
        return Response(
            {"error": f"Failed to get hourly prediction: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
