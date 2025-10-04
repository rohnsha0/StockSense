from mcp.server.fastmcp import FastMCP
import pandas as pd
import yfinance as yf
from main import fetchStockPrediction

mcp = FastMCP(name="stocksense")


@mcp.tool()
def get_stock_prediction(symbol: str) -> float:
    """
    Get the stock prediction for a given symbol.
    Append `.NS` if the symbol is for NSE stocks and `.BO` for BSE stocks. If nothing is provided, default to NSE. If the symbol is not found, it will return Integer NaN.

    Args:
        symbol (str): The stock symbol to look up.

    Returns:
        float: The current stock prediction.
    """
    try:
        return fetchStockPrediction(symbol)
    except Exception as e:
        print(f"Error fetching stock prediction for {symbol}: {e}")
        return float("nan")


@mcp.tool()
def get_current_stock_prices(symbol: str) -> float:
    """
    Get the current stock price for a given symbol.
    Append `.NS` if the symbol is for NSE stocks and `.BO` for BSE stocks. If nothing is provided, default to NSE.

    Args:
        symbol (str): The stock symbol to look up.

    Returns:
        float: The current stock price.
    """

    tk = yf.Ticker(symbol)
    try:
        return tk.fast_info["last_price"]
    except Exception:
        hist = tk.history(period="1d", interval="1m")
        return float(hist["Close"].iloc[-1])


@mcp.tool()
def get_historical_stock_prices(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """Get comprehensive historical stock price data
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS for NSE stocks)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        Dict containing OHLCV data with timestamps
    """
    try:
        import numpy as np
        from datetime import datetime
        
        # Validate inputs
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        
        if period not in valid_periods:
            return {
                "status": "error",
                "message": f"Invalid period '{period}'. Valid periods: {', '.join(valid_periods)}",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
            
        if interval not in valid_intervals:
            return {
                "status": "error",
                "message": f"Invalid interval '{interval}'. Valid intervals: {', '.join(valid_intervals)}",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol.upper())
        
        # Download historical data
        try:
            hist = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=True)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch data from yfinance: {str(e)}",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
        
        # Check if data is empty
        if hist.empty:
            return {
                "status": "error",
                "message": f"No data available for symbol '{symbol}' with period '{period}' and interval '{interval}'",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
        
        # Clean and format data
        hist.reset_index(inplace=True)
        
        # Handle timezone-aware datetime
        if 'Datetime' in hist.columns:
            hist['Date'] = hist['Datetime']
            hist.drop('Datetime', axis=1, inplace=True)
        
        # Convert datetime to string format
        if pd.api.types.is_datetime64_any_dtype(hist['Date']):
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S' if interval.endswith('m') or interval.endswith('h') else '%Y-%m-%d')
        
        # Data validation and cleaning
        # Remove rows with invalid OHLC relationships
        invalid_mask = (hist['High'] < hist['Open']) | (hist['High'] < hist['Close']) | \
                      (hist['Low'] > hist['Open']) | (hist['Low'] > hist['Close']) | \
                      (hist['Open'] <= 0) | (hist['High'] <= 0) | (hist['Low'] <= 0) | (hist['Close'] <= 0)
        
        if invalid_mask.any():
            hist = hist[~invalid_mask]
        
        # Forward fill missing values
        hist.ffill(inplace=True)
        
        # Convert to list of dictionaries
        historical_data = []
        for _, row in hist.iterrows():
            try:
                data_point = {
                    "date": str(row['Date']),
                    "open": round(float(row['Open']), 4),
                    "high": round(float(row['High']), 4),
                    "low": round(float(row['Low']), 4),
                    "close": round(float(row['Close']), 4),
                    "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0,
                    "adj_close": round(float(row['Close']), 4)  # Already adjusted by yfinance
                }
                historical_data.append(data_point)
            except (ValueError, TypeError):
                # Skip invalid data points
                continue
        
        if not historical_data:
            return {
                "status": "error",
                "message": f"No valid data points found for symbol '{symbol}'",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
        
        # Calculate summary statistics
        closes = [d['close'] for d in historical_data]
        volumes = [d['volume'] for d in historical_data if d['volume'] > 0]
        highs = [d['high'] for d in historical_data]
        lows = [d['low'] for d in historical_data]
        
        # Price change calculation
        price_change = (closes[-1] - closes[0]) / closes[0] if len(closes) > 1 else 0.0
        
        # Volatility calculation (standard deviation of returns)
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # Data quality assessment
        total_expected_points = len(hist) if len(hist) > 0 else len(historical_data)
        completeness = len(historical_data) / total_expected_points if total_expected_points > 0 else 1.0
        missing_days = total_expected_points - len(historical_data)
        
        # Get ticker info for additional metadata
        try:
            info = ticker.info
            currency = info.get('currency', 'USD')
            timezone = info.get('exchangeTimezoneName', 'America/New_York')
        except Exception:
            currency = 'USD'
            timezone = 'America/New_York'
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data_points": len(historical_data),
            "last_updated": datetime.now().isoformat() + "Z",
            "currency": currency,
            "timezone": timezone,
            "historical_data": historical_data,
            "summary_statistics": {
                "price_change": round(price_change, 6),
                "price_change_percent": round(price_change * 100, 2),
                "volatility": round(volatility, 6),
                "max_price": round(max(highs), 4),
                "min_price": round(min(lows), 4),
                "avg_volume": int(np.mean(volumes)) if volumes else 0,
                "latest_price": closes[-1] if closes else 0
            },
            "data_quality": {
                "completeness": round(completeness, 4),
                "missing_days": missing_days,
                "total_points": len(historical_data),
                "data_source": "yfinance"
            },
            "status": "success",
            "message": f"Successfully retrieved {len(historical_data)} data points for {symbol.upper()}"
        }
        
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Required package not available: {str(e)}. Please install yfinance and pandas.",
            "symbol": symbol,
            "period": period,
            "interval": interval
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error fetching historical data: {str(e)}",
            "symbol": symbol,
            "period": period,
            "interval": interval
        }


@mcp.tool()
def get_historical_stock_prices_by_date(symbol: str, start_date: str, end_date: str) -> dict:
    """Get historical stock prices between specific dates (backward compatibility function)
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS for NSE stocks)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dict containing historical data with dates as keys and closing prices as values
    """
    try:
        from datetime import datetime
        
        # Validate date format
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return {
                "status": "error",
                "message": "Invalid date format. Please use YYYY-MM-DD format.",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            }
        
        if start_dt > end_dt:
            return {
                "status": "error",
                "message": "Start date cannot be after end date.",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Fetch data using yfinance
        ticker = yf.Ticker(symbol.upper())
        
        try:
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch data from yfinance: {str(e)}",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            }
        
        if hist.empty:
            return {
                "status": "error",
                "message": f"No data available for symbol '{symbol}' between {start_date} and {end_date}",
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Convert to the expected format (date strings mapping to closing prices)
        hist.reset_index(inplace=True)
        result = {}
        
        for _, row in hist.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            result[date_str] = round(float(row['Close']), 4)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
