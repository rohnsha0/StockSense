# StockSense API

## Overview
StockSense is a comprehensive stock analysis API built with Django that provides real-time stock data, technical indicators, and machine learning-based predictions for Indian and international stock markets.

## Features
- **Real-time Stock Data**: Current prices and historical data using yfinance
- **Technical Analysis**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR indicators
- **ML Predictions**: Daily and hourly stock price predictions using TensorFlow Lite models
- **Multi-Exchange Support**: BSE and NSE stock data integration
- **RESTful API**: Clean JSON responses for easy integration

## API Endpoints

### Basic Information
- `GET /` - API version and information

### Stock Data
- `GET /query/{symbol}/` - Basic stock query with recent closing prices
- `GET /query/v2/{symbol}/` - Stock query with dates
- `GET /query/v3/{symbol}/` - Comprehensive stock data with BSE/NSE details
- `GET /info/{symbol}/` - Stock information using yfinance
- `GET /ltp/{symbol}/` - Last traded price with change indicator

### Analysis
- `GET /technical/{symbol}/` - Technical indicators (SMA, EMA, RSI, MACD, etc.)
- `GET /prediction/{symbol}/` - Daily price prediction
- `GET /prediction/hr/{symbol}/` - Hourly price prediction

## Setup and Installation

### Using Django (Recommended)

1. **Install Dependencies**
   ```bash
   pip install -r requirements_django.txt
   ```

2. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

3. **Start Development Server**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

4. **Test the API**
   ```bash
   curl http://localhost:8000/
   ```

### Using Docker

1. **Build Docker Image**
   ```bash
   docker build -f Dockerfile.django -t stocksense-api .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 stocksense-api
   ```

## Legacy FastAPI Implementation

The original FastAPI implementation is available in `main.py` and can be used with the original `Dockerfile`.

## Dependencies

### Core Framework
- Django 5.2.4
- Django REST Framework 3.16.0

### Data & ML
- pandas - Data manipulation
- numpy - Numerical operations
- tensorflow - Machine learning predictions
- scikit-learn - Data preprocessing
- yfinance - Stock data retrieval

### Additional
- beautifulsoup4 - Web scraping support
- requests - HTTP requests

## Data Files

- `equity_bse.csv` - BSE stock symbols and company information
- `equity_nse.csv` - NSE stock symbols and company information
- `exports/` - TensorFlow Lite models for daily predictions
- `hourlyExports/` - TensorFlow Lite models and data for hourly predictions

## Usage Examples

### Get Stock Information
```bash
curl "http://localhost:8000/info/RELIANCE.NS/"
```

### Get Technical Indicators
```bash
curl "http://localhost:8000/technical/TCS.NS/"
```

### Get Price Prediction
```bash
curl "http://localhost:8000/prediction/INFY.NS/"
```

## Development

### Project Structure
```
stocksense_api/          # Django project configuration
├── settings.py          # Django settings
├── urls.py             # Main URL configuration
└── wsgi.py             # WSGI configuration

api/                    # Main API application
├── views.py            # API endpoints
├── urls.py             # API URL patterns
├── stock_logic.py      # Business logic functions
└── models.py           # Django models (if needed)
```

### Running Tests
```bash
python manage.py test
```

## Production Deployment

For production deployment, consider using:
- **Gunicorn** as WSGI server
- **Nginx** as reverse proxy
- **PostgreSQL** or **MySQL** instead of SQLite
- **Redis** for caching
- Environment variables for configuration

Example production command:
```bash
gunicorn stocksense_api.wsgi:application --bind 0.0.0.0:8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.