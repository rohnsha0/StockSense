from mcp.server.fastmcp import FastMCP
import pandas as pd
import yfinance as yf
from main import fetchStockPrediction
from fundamentals import get_stock_fundamentals_analysis

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


@mcp.tool()
def get_technical_indicators(symbol: str, period: str = "1y", use_cache: bool = True) -> dict:
    """Get comprehensive technical indicators for stock analysis
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS for NSE stocks, RELIANCE.BO for BSE stocks)
        period: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: "1y"
        use_cache: Whether to use caching for faster responses. Default: True
        
    Returns:
        Dict containing comprehensive technical analysis including:
        - RSI (Relative Strength Index): 14-period for overbought/oversold signals
        - MACD: Signal line crossovers and histogram analysis  
        - Bollinger Bands: Price volatility and mean reversion indicators
        - Stochastic Oscillator: %K and %D lines for momentum shifts
        - ADX (Average Directional Index): Trend strength measurement
        - Parabolic SAR: Stop and reverse points
        - Ichimoku Cloud: Comprehensive trend analysis
        - ATR (Average True Range): Volatility measurement
        - Buy/sell/hold signals for each indicator
        - Overall market signal (bullish/bearish/neutral)
        - Signal interpretations and recommendations
        - Cache status and timestamp information
    """
    try:
        from technical_analysis import get_comprehensive_technical_indicators
        return get_comprehensive_technical_indicators(symbol, period=period, use_cache=use_cache)
    except Exception as e:
        return {
            "symbol": symbol,
            "error": f"Failed to get technical indicators: {str(e)}",
            "status": "error",
            "timestamp": pd.Timestamp.now().isoformat(),
            "period": period,
            "use_cache": use_cache
        }


@mcp.tool()
def get_specific_technical_indicator(symbol: str, indicator: str, period: str = "1y") -> dict:
    """Get a specific technical indicator for focused analysis
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS for NSE stocks)
        indicator: Specific indicator to calculate (rsi, macd, bollinger_bands, stochastic, adx, atr, parabolic_sar, ichimoku)
        period: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: "1y"
        
    Returns:
        Dict containing the specific technical indicator data with detailed analysis
    """
    try:
        from technical_analysis import get_comprehensive_technical_indicators
        
        # Validate indicator parameter
        valid_indicators = ['rsi', 'macd', 'bollinger_bands', 'stochastic', 'adx', 'atr', 'parabolic_sar', 'ichimoku']
        if indicator.lower() not in valid_indicators:
            return {
                "symbol": symbol,
                "indicator": indicator,
                "error": f"Invalid indicator. Valid options: {', '.join(valid_indicators)}",
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        # Get comprehensive data
        full_data = get_comprehensive_technical_indicators(symbol, period=period, use_cache=True)
        
        if full_data.get('status') == 'error':
            return full_data
        
        # Extract specific indicator
        specific_indicator = full_data['indicators'].get(indicator.lower(), {})
        
        # Create focused response
        result = {
            "symbol": symbol.upper(),
            "indicator": indicator.lower(),
            "period": period,
            "timestamp": full_data.get('timestamp'),
            "current_price": full_data.get('current_price'),
            "data": specific_indicator,
            "status": "success"
        }
        
        # Add interpretation based on indicator type
        if indicator.lower() == 'rsi':
            rsi_val = specific_indicator.get('value', 50)
            if rsi_val > 70:
                result['interpretation'] = "Overbought - potential sell signal"
            elif rsi_val < 30:
                result['interpretation'] = "Oversold - potential buy signal"
            else:
                result['interpretation'] = "Neutral territory"
                
        elif indicator.lower() == 'macd':
            macd_line = specific_indicator.get('macd_line', 0)
            signal_line = specific_indicator.get('signal_line', 0)
            if macd_line > signal_line:
                result['interpretation'] = "MACD above signal line - bullish momentum"
            else:
                result['interpretation'] = "MACD below signal line - bearish momentum"
                
        elif indicator.lower() == 'bollinger_bands':
            position = specific_indicator.get('position', 'mid')
            if position == 'above_upper':
                result['interpretation'] = "Price above upper band - potentially overbought"
            elif position == 'below_lower':
                result['interpretation'] = "Price below lower band - potentially oversold"
            else:
                result['interpretation'] = f"Price in {position.replace('_', ' ')} of bands"
        
        return result
        
    except Exception as e:
        return {
            "symbol": symbol,
            "indicator": indicator,
            "error": f"Failed to get specific technical indicator: {str(e)}",
            "status": "error",
            "timestamp": pd.Timestamp.now().isoformat(),
            "period": period
        }


@mcp.tool()
def get_trading_signals(symbol: str, period: str = "1y") -> dict:
    """Get trading signals and recommendations based on technical analysis
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS for NSE stocks)
        period: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: "1y"
        
    Returns:
        Dict containing focused trading signals, recommendations, and confidence levels
    """
    try:
        from technical_analysis import get_comprehensive_technical_indicators
        
        # Get comprehensive technical data
        tech_data = get_comprehensive_technical_indicators(symbol, period=period, use_cache=True)
        
        if tech_data.get('status') == 'error':
            return tech_data
        
        # Extract signals
        signals = tech_data.get('signals', {})
        indicators = tech_data.get('indicators', {})
        
        # Count signal types for confidence calculation
        buy_signals = sum(1 for signal in signals.values() if isinstance(signal, str) and 'buy' in signal.lower())
        sell_signals = sum(1 for signal in signals.values() if isinstance(signal, str) and 'sell' in signal.lower())
        neutral_signals = sum(1 for signal in signals.values() if isinstance(signal, str) and 'neutral' in signal.lower())
        
        total_signals = buy_signals + sell_signals + neutral_signals
        
        # Calculate confidence levels
        if total_signals > 0:
            buy_confidence = round((buy_signals / total_signals) * 100, 1)
            sell_confidence = round((sell_signals / total_signals) * 100, 1)
            neutral_confidence = round((neutral_signals / total_signals) * 100, 1)
        else:
            buy_confidence = sell_confidence = neutral_confidence = 0
        
        # Determine primary recommendation
        overall_signal = tech_data.get('overall_signal', 'neutral')
        
        # Create comprehensive trading signal response
        result = {
            "symbol": symbol.upper(),
            "period": period,
            "timestamp": tech_data.get('timestamp'),
            "current_price": tech_data.get('current_price'),
            "overall_signal": overall_signal,
            "confidence_levels": {
                "buy_confidence": buy_confidence,
                "sell_confidence": sell_confidence,
                "neutral_confidence": neutral_confidence
            },
            "signal_breakdown": {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "neutral_signals": neutral_signals,
                "total_indicators": total_signals
            },
            "individual_signals": signals,
            "key_levels": {
                "resistance": indicators.get('bollinger_bands', {}).get('upper', 0),
                "support": indicators.get('bollinger_bands', {}).get('lower', 0),
                "pivot": indicators.get('bollinger_bands', {}).get('middle', 0),
                "stop_loss": indicators.get('parabolic_sar', {}).get('value', 0)
            },
            "momentum_indicators": {
                "rsi": indicators.get('rsi', {}).get('value', 50),
                "stochastic_k": indicators.get('stochastic', {}).get('k_percent', 50),
                "trend_strength": indicators.get('adx', {}).get('value', 20)
            },
            "recommendations": [],
            "status": "success"
        }
        
        # Generate specific recommendations
        if overall_signal == 'bullish':
            result['recommendations'].append("Consider buying on pullbacks to support levels")
            if indicators.get('rsi', {}).get('value', 50) < 70:
                result['recommendations'].append("RSI not yet overbought - momentum may continue")
        elif overall_signal == 'bearish':
            result['recommendations'].append("Consider selling or taking profits")
            if indicators.get('rsi', {}).get('value', 50) > 30:
                result['recommendations'].append("RSI not yet oversold - downtrend may continue")
        else:
            result['recommendations'].append("Hold current position or wait for clearer signals")
            result['recommendations'].append("Monitor for breakout above resistance or below support")
        
        # Add volatility-based recommendations
        atr_volatility = indicators.get('atr', {}).get('volatility', 'moderate')
        if atr_volatility == 'high':
            result['recommendations'].append("High volatility detected - use wider stop losses")
        elif atr_volatility == 'low':
            result['recommendations'].append("Low volatility - consider tighter position sizing")
        
        return result
        
    except Exception as e:
        return {
            "symbol": symbol,
            "error": f"Failed to get trading signals: {str(e)}",
            "status": "error",
            "timestamp": pd.Timestamp.now().isoformat(),
            "period": period
        }


@mcp.tool()
def compare_technical_indicators(symbols: list, period: str = "1y") -> dict:
    """Compare technical indicators across multiple stocks for relative analysis
    
    Args:
        symbols: List of stock symbols to compare (e.g., ["AAPL", "MSFT", "GOOGL"])
        period: Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Default: "1y"
        
    Returns:
        Dict containing comparative analysis of technical indicators across all symbols
    """
    try:
        from technical_analysis import get_comprehensive_technical_indicators
        
        if not symbols or len(symbols) == 0:
            return {
                "error": "No symbols provided for comparison",
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        if len(symbols) > 10:
            return {
                "error": "Too many symbols for comparison (maximum 10 allowed)",
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        comparison_data = {}
        successful_symbols = []
        failed_symbols = []
        
        # Get technical data for each symbol
        for symbol in symbols:
            try:
                tech_data = get_comprehensive_technical_indicators(symbol, period=period, use_cache=True)
                if tech_data.get('status') == 'success':
                    comparison_data[symbol.upper()] = tech_data
                    successful_symbols.append(symbol.upper())
                else:
                    failed_symbols.append(symbol.upper())
            except Exception:
                failed_symbols.append(symbol.upper())
        
        if not successful_symbols:
            return {
                "error": "No valid data retrieved for any symbols",
                "failed_symbols": failed_symbols,
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        # Create comparison summary
        comparison_summary = {
            "period": period,
            "timestamp": pd.Timestamp.now().isoformat(),
            "successful_symbols": successful_symbols,
            "failed_symbols": failed_symbols,
            "comparison": {},
            "rankings": {},
            "status": "success"
        }
        
        # Compare RSI values
        rsi_data = {}
        macd_data = {}
        overall_signals = {}
        
        for symbol in successful_symbols:
            data = comparison_data[symbol]
            rsi_data[symbol] = data['indicators']['rsi']['value']
            macd_data[symbol] = data['indicators']['macd']['macd_line']
            overall_signals[symbol] = data['overall_signal']
        
        # Create rankings
        comparison_summary['rankings'] = {
            "most_oversold": sorted(rsi_data.items(), key=lambda x: x[1])[:3],
            "most_overbought": sorted(rsi_data.items(), key=lambda x: x[1], reverse=True)[:3],
            "strongest_momentum": sorted(macd_data.items(), key=lambda x: x[1], reverse=True)[:3],
            "weakest_momentum": sorted(macd_data.items(), key=lambda x: x[1])[:3]
        }
        
        # Signal distribution
        signal_count = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        for signal in overall_signals.values():
            signal_count[signal] = signal_count.get(signal, 0) + 1
        
        comparison_summary['signal_distribution'] = signal_count
        
        # Individual symbol data
        for symbol in successful_symbols:
            data = comparison_data[symbol]
            comparison_summary['comparison'][symbol] = {
                "current_price": data['current_price'],
                "overall_signal": data['overall_signal'],
                "rsi": data['indicators']['rsi']['value'],
                "macd_line": data['indicators']['macd']['macd_line'],
                "trend_strength": data['indicators']['adx']['value'],
                "volatility": data['indicators']['atr']['volatility']
            }
        
        return comparison_summary
        
    except Exception as e:
        return {
            "error": f"Failed to compare technical indicators: {str(e)}",
            "status": "error",
            "timestamp": pd.Timestamp.now().isoformat(),
            "period": period
        }


@mcp.tool()
def get_stock_fundamentals(symbol: str) -> dict:
    """Get detailed stock fundamentals like P/E, market cap, growth metrics, quality scores, and investment analysis
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
        
    Returns:
        Dict containing comprehensive fundamental analysis data including:
        - Financial performance (revenue growth, margins, ROE, ROA, ROIC)
        - Valuation metrics (P/E, P/B, P/S, EV/EBITDA, PEG ratio)
        - Balance sheet analysis (debt levels, cash position, working capital)
        - Growth analysis (revenue/earnings CAGR)
        - Quality scores (Piotroski F-Score, Altman Z-Score)
        - Peer comparison (industry averages)
        - Investment thesis (strengths, concerns, overall rating)
    """
    try:
        return get_stock_fundamentals_analysis(symbol)
    except Exception as e:
        return {
            "symbol": symbol,
            "error": f"Failed to get fundamentals analysis: {str(e)}",
            "analysis_date": pd.Timestamp.now().isoformat()
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
