"""
Technical Analysis Module for StockSense
Provides comprehensive technical indicators for stock analysis
"""

import pandas as pd
import yfinance as yf
import talib
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Simple file-based cache
CACHE_DIR = Path(".cache")
CACHE_EXPIRY_MINUTES = 15


def get_cache_key(symbol: str, period: str) -> str:
    """Generate cache key for symbol and period"""
    return f"{symbol.upper()}_{period}_technical.json"


def is_cache_valid(cache_file: Path) -> bool:
    """Check if cache file exists and is still valid"""
    if not cache_file.exists():
        return False
    
    # Check if cache is expired
    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    expiry_time = cache_time + timedelta(minutes=CACHE_EXPIRY_MINUTES)
    
    return datetime.now() < expiry_time


def load_from_cache(cache_file: Path) -> Dict[str, Any]:
    """Load technical indicators from cache"""
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_to_cache(cache_file: Path, data: Dict[str, Any]) -> None:
    """Save technical indicators to cache"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # Ignore cache errors


def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch stock data from yfinance with error handling
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
        period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        data = ticker.history(period=period, auto_adjust=True, prepost=True)
        
        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Ensure we have enough data for calculations
        if len(data) < 50:
            # Try to get more data with a longer period
            data = ticker.history(period="2y", auto_adjust=True, prepost=True)
            
        if len(data) < 20:
            raise ValueError(f"Insufficient data for technical analysis: {len(data)} days")
            
        return data
        
    except Exception as e:
        raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using TA-Lib"""
    try:
        rsi_values = talib.RSI(close_prices.values, timeperiod=period)
        return pd.Series(rsi_values, index=close_prices.index)
    except Exception:
        # Fallback manual calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


def calculate_macd(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD using TA-Lib"""
    try:
        macd_line, macd_signal, macd_histogram = talib.MACD(
            close_prices.values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        return {
            'macd_line': pd.Series(macd_line, index=close_prices.index),
            'signal_line': pd.Series(macd_signal, index=close_prices.index),
            'histogram': pd.Series(macd_histogram, index=close_prices.index)
        }
    except Exception:
        # Fallback manual calculation
        ema_fast = close_prices.ewm(span=fast_period).mean()
        ema_slow = close_prices.ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }


def calculate_bollinger_bands(close_prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands using TA-Lib"""
    try:
        upper, middle, lower = talib.BBANDS(
            close_prices.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return {
            'upper': pd.Series(upper, index=close_prices.index),
            'middle': pd.Series(middle, index=close_prices.index),
            'lower': pd.Series(lower, index=close_prices.index)
        }
    except Exception:
        # Fallback manual calculation
        middle = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }


def calculate_stochastic(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator using TA-Lib"""
    try:
        slowk, slowd = talib.STOCH(
            high_prices.values,
            low_prices.values,
            close_prices.values,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        return {
            'k_percent': pd.Series(slowk, index=close_prices.index),
            'd_percent': pd.Series(slowd, index=close_prices.index)
        }
    except Exception:
        # Fallback manual calculation
        lowest_low = low_prices.rolling(window=k_period).min()
        highest_high = high_prices.rolling(window=k_period).max()
        k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }


def calculate_adx(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX using TA-Lib"""
    try:
        return pd.Series(
            talib.ADX(high_prices.values, low_prices.values, close_prices.values, timeperiod=period),
            index=close_prices.index
        )
    except Exception:
        # Fallback manual calculation (simplified)
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift())
        tr3 = abs(low_prices - close_prices.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = high_prices.diff()
        minus_dm = low_prices.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx


def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range using TA-Lib"""
    try:
        return pd.Series(
            talib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=period),
            index=close_prices.index
        )
    except Exception:
        # Fallback manual calculation
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift())
        tr3 = abs(low_prices - close_prices.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


def calculate_parabolic_sar(high_prices: pd.Series, low_prices: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR using TA-Lib"""
    try:
        return pd.Series(
            talib.SAR(high_prices.values, low_prices.values, acceleration=acceleration, maximum=maximum),
            index=high_prices.index
        )
    except Exception:
        # Simplified fallback - return SMA as approximation
        return (high_prices + low_prices) / 2


def calculate_ichimoku(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series) -> Dict[str, pd.Series]:
    """Calculate Ichimoku Cloud components"""
    try:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high_prices.rolling(window=52).max() + low_prices.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = close_prices.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    except Exception:
        # Return simple moving averages as fallback
        return {
            'tenkan_sen': close_prices.rolling(window=9).mean(),
            'kijun_sen': close_prices.rolling(window=26).mean(),
            'senkou_span_a': close_prices.rolling(window=26).mean(),
            'senkou_span_b': close_prices.rolling(window=52).mean(),
            'chikou_span': close_prices
        }


def interpret_signals(indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
    """
    Interpret technical indicators to generate buy/sell/hold signals
    
    Args:
        indicators: Dictionary containing all calculated indicators
        current_price: Current stock price
        
    Returns:
        Dictionary with individual and overall signals
    """
    signals = {}
    
    # RSI Signal
    rsi_value = indicators['rsi']['value']
    if rsi_value > 70:
        signals['rsi'] = 'sell'  # Overbought
    elif rsi_value < 30:
        signals['rsi'] = 'buy'   # Oversold
    else:
        signals['rsi'] = 'neutral'
    
    # MACD Signal
    macd = indicators['macd']
    if macd['macd_line'] > macd['signal_line'] and macd['histogram'] > 0:
        signals['macd'] = 'buy'
    elif macd['macd_line'] < macd['signal_line'] and macd['histogram'] < 0:
        signals['macd'] = 'sell'
    else:
        signals['macd'] = 'neutral'
    
    # Bollinger Bands Signal
    bb = indicators['bollinger_bands']
    if current_price > bb['upper']:
        signals['bollinger_bands'] = 'sell'  # Above upper band
    elif current_price < bb['lower']:
        signals['bollinger_bands'] = 'buy'   # Below lower band
    else:
        signals['bollinger_bands'] = 'neutral'
    
    # Stochastic Signal
    stoch = indicators['stochastic']
    if stoch['k_percent'] > 80 and stoch['d_percent'] > 80:
        signals['stochastic'] = 'sell'  # Overbought
    elif stoch['k_percent'] < 20 and stoch['d_percent'] < 20:
        signals['stochastic'] = 'buy'   # Oversold
    else:
        signals['stochastic'] = 'neutral'
    
    # ADX Signal (trend strength)
    adx_value = indicators['adx']['value']
    if adx_value > 25:
        signals['adx'] = 'strong_trend'
    elif adx_value > 20:
        signals['adx'] = 'moderate_trend'
    else:
        signals['adx'] = 'weak_trend'
    
    # Overall Signal
    buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
    sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
    
    if buy_signals > sell_signals and buy_signals >= 2:
        signals['overall'] = 'bullish'
    elif sell_signals > buy_signals and sell_signals >= 2:
        signals['overall'] = 'bearish'
    else:
        signals['overall'] = 'neutral'
    
    return signals


def get_comprehensive_technical_indicators(symbol: str, period: str = "1y", use_cache: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive technical indicators for a stock symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
        period: Time period for analysis
        use_cache: Whether to use caching for faster responses
        
    Returns:
        Dictionary containing all technical indicators and signals
    """
    try:
        # Check cache first
        if use_cache:
            cache_key = get_cache_key(symbol, period)
            cache_file = CACHE_DIR / cache_key
            
            if is_cache_valid(cache_file):
                cached_data = load_from_cache(cache_file)
                if cached_data and cached_data.get('status') == 'success':
                    cached_data['from_cache'] = True
                    return cached_data
        # Fetch stock data
        data = get_stock_data(symbol, period)
        
        if len(data) < 20:
            raise ValueError(f"Insufficient data for analysis: {len(data)} days")
        
        # Get current values
        current_price = float(data['Close'].iloc[-1])
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']
        
        # Calculate all indicators
        rsi = calculate_rsi(close_prices)
        macd_data = calculate_macd(close_prices)
        bb_data = calculate_bollinger_bands(close_prices)
        stoch_data = calculate_stochastic(high_prices, low_prices, close_prices)
        adx = calculate_adx(high_prices, low_prices, close_prices)
        atr = calculate_atr(high_prices, low_prices, close_prices)
        sar = calculate_parabolic_sar(high_prices, low_prices)
        ichimoku_data = calculate_ichimoku(high_prices, low_prices, close_prices)
        
        # Get latest values (handle NaN values)
        def safe_value(series, default=0.0):
            try:
                val = series.iloc[-1]
                return float(val) if pd.notna(val) else default
            except (IndexError, ValueError, TypeError):
                return default
        
        # Compile indicators
        indicators = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat() + "Z",
            "current_price": round(current_price, 4),
            "data_points": len(data),
            "period": period,
            "indicators": {
                "rsi": {
                    "value": round(safe_value(rsi, 50), 2),
                    "period": 14,
                    "signal": "neutral"
                },
                "macd": {
                    "macd_line": round(safe_value(macd_data['macd_line']), 4),
                    "signal_line": round(safe_value(macd_data['signal_line']), 4),
                    "histogram": round(safe_value(macd_data['histogram']), 4),
                    "signal": "neutral"
                },
                "bollinger_bands": {
                    "upper": round(safe_value(bb_data['upper'], current_price * 1.02), 4),
                    "middle": round(safe_value(bb_data['middle'], current_price), 4),
                    "lower": round(safe_value(bb_data['lower'], current_price * 0.98), 4),
                    "position": "mid"
                },
                "stochastic": {
                    "k_percent": round(safe_value(stoch_data['k_percent'], 50), 2),
                    "d_percent": round(safe_value(stoch_data['d_percent'], 50), 2),
                    "signal": "neutral"
                },
                "adx": {
                    "value": round(safe_value(adx, 20), 2),
                    "trend_strength": "moderate",
                    "period": 14
                },
                "atr": {
                    "value": round(safe_value(atr, current_price * 0.02), 4),
                    "volatility": "moderate",
                    "period": 14
                },
                "parabolic_sar": {
                    "value": round(safe_value(sar, current_price), 4),
                    "trend": "neutral"
                },
                "ichimoku": {
                    "tenkan_sen": round(safe_value(ichimoku_data['tenkan_sen'], current_price), 4),
                    "kijun_sen": round(safe_value(ichimoku_data['kijun_sen'], current_price), 4),
                    "senkou_span_a": round(safe_value(ichimoku_data['senkou_span_a'], current_price), 4),
                    "senkou_span_b": round(safe_value(ichimoku_data['senkou_span_b'], current_price), 4),
                    "cloud_status": "neutral"
                }
            }
        }
        
        # Generate signals
        signals = interpret_signals(indicators['indicators'], current_price)
        
        # Update indicator signals
        indicators['indicators']['rsi']['signal'] = signals.get('rsi', 'neutral')
        indicators['indicators']['macd']['signal'] = signals.get('macd', 'neutral')
        indicators['indicators']['stochastic']['signal'] = signals.get('stochastic', 'neutral')
        indicators['indicators']['adx']['trend_strength'] = signals.get('adx', 'moderate_trend')
        
        # Bollinger Bands position
        bb = indicators['indicators']['bollinger_bands']
        if current_price > bb['upper']:
            bb['position'] = 'above_upper'
        elif current_price < bb['lower']:
            bb['position'] = 'below_lower'
        elif current_price > bb['middle']:
            bb['position'] = 'upper_half'
        else:
            bb['position'] = 'lower_half'
        
        # Parabolic SAR trend
        sar_val = indicators['indicators']['parabolic_sar']['value']
        if current_price > sar_val:
            indicators['indicators']['parabolic_sar']['trend'] = 'bullish'
        else:
            indicators['indicators']['parabolic_sar']['trend'] = 'bearish'
        
        # Ichimoku cloud status
        ich = indicators['indicators']['ichimoku']
        if current_price > max(ich['senkou_span_a'], ich['senkou_span_b']):
            ich['cloud_status'] = 'above_cloud'
        elif current_price < min(ich['senkou_span_a'], ich['senkou_span_b']):
            ich['cloud_status'] = 'below_cloud'
        else:
            ich['cloud_status'] = 'in_cloud'
        
        # ATR volatility classification
        atr_val = indicators['indicators']['atr']['value']
        atr_percentage = (atr_val / current_price) * 100
        if atr_percentage > 3:
            indicators['indicators']['atr']['volatility'] = 'high'
        elif atr_percentage > 1.5:
            indicators['indicators']['atr']['volatility'] = 'moderate'
        else:
            indicators['indicators']['atr']['volatility'] = 'low'
        
        # Set overall signal
        indicators['overall_signal'] = signals.get('overall', 'neutral')
        indicators['signals'] = signals
        
        indicators['status'] = 'success'
        indicators['message'] = f'Successfully calculated technical indicators for {symbol}'
        indicators['from_cache'] = False
        
        # Save to cache
        if use_cache:
            cache_key = get_cache_key(symbol, period)
            cache_file = CACHE_DIR / cache_key
            save_to_cache(cache_file, indicators)
        
        return indicators
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat() + "Z",
            "status": "error",
            "message": f"Failed to calculate technical indicators: {str(e)}",
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # Test the implementation
    test_symbols = ["AAPL", "RELIANCE.NS", "TCS.NS"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        result = get_comprehensive_technical_indicators(symbol)
        if result['status'] == 'success':
            print(f"✓ RSI: {result['indicators']['rsi']['value']}")
            print(f"✓ MACD: {result['indicators']['macd']['macd_line']}")
            print(f"✓ Overall Signal: {result['overall_signal']}")
        else:
            print(f"✗ Error: {result['message']}")