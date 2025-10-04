"""
Stock Fundamentals Analysis Module
Provides comprehensive financial metrics, valuation models, and growth analysis
"""

from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import yfinance as yf


class StockFundamentalsAnalyzer:
    """Comprehensive stock fundamentals analysis engine"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        self._info = None
        self._financials = None
        self._balance_sheet = None
        self._cash_flow = None
        self._quarterly_financials = None
        self._quarterly_balance_sheet = None
        self._quarterly_cash_flow = None
        
    @property
    def info(self):
        """Lazy load ticker info"""
        if self._info is None:
            try:
                self._info = self.ticker.info
            except Exception:
                self._info = {}
        return self._info
    
    @property
    def financials(self):
        """Lazy load annual financials"""
        if self._financials is None:
            try:
                self._financials = self.ticker.financials
            except Exception:
                self._financials = pd.DataFrame()
        return self._financials
    
    @property
    def balance_sheet(self):
        """Lazy load annual balance sheet"""
        if self._balance_sheet is None:
            try:
                self._balance_sheet = self.ticker.balance_sheet
            except Exception:
                self._balance_sheet = pd.DataFrame()
        return self._balance_sheet
    
    @property
    def cash_flow(self):
        """Lazy load annual cash flow"""
        if self._cash_flow is None:
            try:
                self._cash_flow = self.ticker.cashflow
            except Exception:
                self._cash_flow = pd.DataFrame()
        return self._cash_flow
    
    @property
    def quarterly_financials(self):
        """Lazy load quarterly financials"""
        if self._quarterly_financials is None:
            try:
                self._quarterly_financials = self.ticker.quarterly_financials
            except Exception:
                self._quarterly_financials = pd.DataFrame()
        return self._quarterly_financials
    
    @property
    def quarterly_balance_sheet(self):
        """Lazy load quarterly balance sheet"""
        if self._quarterly_balance_sheet is None:
            try:
                self._quarterly_balance_sheet = self.ticker.quarterly_balance_sheet
            except Exception:
                self._quarterly_balance_sheet = pd.DataFrame()
        return self._quarterly_balance_sheet
    
    @property
    def quarterly_cash_flow(self):
        """Lazy load quarterly cash flow"""
        if self._quarterly_cash_flow is None:
            try:
                self._quarterly_cash_flow = self.ticker.quarterly_cashflow
            except Exception:
                self._quarterly_cash_flow = pd.DataFrame()
        return self._quarterly_cash_flow

    def get_financial_performance(self) -> Dict:
        """Calculate comprehensive financial performance metrics"""
        try:
            info = self.info
            financials = self.financials
            quarterly_financials = self.quarterly_financials
            
            performance = {}
            
            # Revenue Growth
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue'].dropna()
                if len(revenues) >= 2:
                    performance['revenue_growth_yoy'] = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]
                else:
                    performance['revenue_growth_yoy'] = None
            else:
                performance['revenue_growth_yoy'] = None
            
            # Quarterly revenue growth
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                q_revenues = quarterly_financials.loc['Total Revenue'].dropna()
                if len(q_revenues) >= 2:
                    performance['revenue_growth_qoq'] = (q_revenues.iloc[0] - q_revenues.iloc[1]) / q_revenues.iloc[1]
                else:
                    performance['revenue_growth_qoq'] = None
            else:
                performance['revenue_growth_qoq'] = None
            
            # Earnings Growth
            if not financials.empty and 'Net Income' in financials.index:
                earnings = financials.loc['Net Income'].dropna()
                if len(earnings) >= 2:
                    performance['earnings_growth_yoy'] = (earnings.iloc[0] - earnings.iloc[1]) / earnings.iloc[1]
                else:
                    performance['earnings_growth_yoy'] = None
            else:
                performance['earnings_growth_yoy'] = None
            
            # Free Cash Flow
            cash_flow = self.cash_flow
            if not cash_flow.empty and 'Free Cash Flow' in cash_flow.index:
                performance['free_cash_flow'] = cash_flow.loc['Free Cash Flow'].iloc[0] if not cash_flow.loc['Free Cash Flow'].empty else None
            else:
                performance['free_cash_flow'] = None
            
            # Margins
            if not financials.empty:
                latest_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else None
                latest_gross_profit = financials.loc['Gross Profit'].iloc[0] if 'Gross Profit' in financials.index else None
                latest_operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else None
                latest_net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else None
                
                if latest_revenue and latest_revenue != 0:
                    performance['gross_margin'] = latest_gross_profit / latest_revenue if latest_gross_profit else None
                    performance['operating_margin'] = latest_operating_income / latest_revenue if latest_operating_income else None
                    performance['net_margin'] = latest_net_income / latest_revenue if latest_net_income else None
                else:
                    performance['gross_margin'] = None
                    performance['operating_margin'] = None
                    performance['net_margin'] = None
            else:
                performance['gross_margin'] = None
                performance['operating_margin'] = None
                performance['net_margin'] = None
            
            # Return Metrics from info
            performance['roe'] = info.get('returnOnEquity')
            performance['roa'] = info.get('returnOnAssets')
            performance['roic'] = info.get('returnOnCapital')
            
            return performance
            
        except Exception as e:
            print(f"Error calculating financial performance for {self.symbol}: {e}")
            return {}

    def get_valuation_metrics(self) -> Dict:
        """Calculate comprehensive valuation metrics"""
        try:
            info = self.info
            
            valuation = {
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_cash_flow': info.get('priceToCashFlow'),
                'enterprise_value': info.get('enterpriseValue'),
                'market_cap': info.get('marketCap')
            }
            
            return valuation
            
        except Exception as e:
            print(f"Error calculating valuation metrics for {self.symbol}: {e}")
            return {}

    def get_balance_sheet_analysis(self) -> Dict:
        """Analyze balance sheet strength"""
        try:
            info = self.info
            balance_sheet = self.balance_sheet
            
            analysis = {
                'total_debt': info.get('totalDebt'),
                'net_debt': info.get('netDebt'),
                'cash_position': info.get('totalCash'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'working_capital': None,
                'book_value_per_share': info.get('bookValue'),
                'tangible_book_value': info.get('tangibleBookValue')
            }
            
            # Calculate working capital if balance sheet is available
            if not balance_sheet.empty:
                current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None
                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None
                
                if current_assets and current_liabilities:
                    analysis['working_capital'] = current_assets - current_liabilities
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing balance sheet for {self.symbol}: {e}")
            return {}

    def get_growth_analysis(self) -> Dict:
        """Calculate growth metrics and trends"""
        try:
            financials = self.financials
            info = self.info
            
            growth = {
                'revenue_cagr_3y': None,
                'revenue_cagr_5y': None,
                'earnings_cagr_3y': None,
                'earnings_cagr_5y': None,
                'book_value_cagr': None,
                'dividend_growth_rate': None
            }
            
            # Calculate revenue CAGR
            if not financials.empty and 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue'].dropna()
                if len(revenues) >= 4:  # 3 years
                    growth['revenue_cagr_3y'] = self._calculate_cagr(revenues.iloc[3], revenues.iloc[0], 3)
                if len(revenues) >= 6:  # 5 years
                    growth['revenue_cagr_5y'] = self._calculate_cagr(revenues.iloc[5], revenues.iloc[0], 5)
            
            # Calculate earnings CAGR
            if not financials.empty and 'Net Income' in financials.index:
                earnings = financials.loc['Net Income'].dropna()
                if len(earnings) >= 4:  # 3 years
                    growth['earnings_cagr_3y'] = self._calculate_cagr(earnings.iloc[3], earnings.iloc[0], 3)
                if len(earnings) >= 6:  # 5 years
                    growth['earnings_cagr_5y'] = self._calculate_cagr(earnings.iloc[5], earnings.iloc[0], 5)
            
            # Get other growth metrics from info
            growth['earnings_growth'] = info.get('earningsGrowth')
            growth['revenue_growth'] = info.get('revenueGrowth')
            
            return growth
            
        except Exception as e:
            print(f"Error calculating growth analysis for {self.symbol}: {e}")
            return {}

    def get_quality_scores(self) -> Dict:
        """Calculate quality scores including Piotroski F-Score and Altman Z-Score"""
        try:
            piotroski_score = self._calculate_piotroski_score()
            altman_z_score = self._calculate_altman_z_score()
            quality_rating = self._calculate_quality_rating(piotroski_score, altman_z_score)
            
            return {
                'piotroski_score': piotroski_score,
                'altman_z_score': altman_z_score,
                'quality_rating': quality_rating
            }
            
        except Exception as e:
            print(f"Error calculating quality scores for {self.symbol}: {e}")
            return {'piotroski_score': None, 'altman_z_score': None, 'quality_rating': None}

    def get_peer_comparison(self) -> Dict:
        """Perform peer comparison analysis"""
        try:
            info = self.info
            industry = info.get('industry', 'Unknown')
            sector = info.get('sector', 'Unknown')
            
            # Industry averages (simplified - in practice would use real industry data)
            industry_averages = self._get_industry_averages(industry)
            
            pe_ratio = info.get('trailingPE')
            roe = info.get('returnOnEquity')
            net_margin = None
            
            # Calculate net margin if possible
            financials = self.financials
            if not financials.empty and 'Total Revenue' in financials.index and 'Net Income' in financials.index:
                revenue = financials.loc['Total Revenue'].iloc[0]
                net_income = financials.loc['Net Income'].iloc[0]
                if revenue and revenue != 0:
                    net_margin = net_income / revenue
            
            comparison = {
                'industry': industry,
                'sector': sector,
                'pe_vs_industry': pe_ratio / industry_averages['pe'] if pe_ratio and industry_averages['pe'] else None,
                'roe_vs_industry': roe / industry_averages['roe'] if roe and industry_averages['roe'] else None,
                'margin_vs_industry': net_margin / industry_averages['net_margin'] if net_margin and industry_averages['net_margin'] else None,
                'industry_pe_avg': industry_averages['pe'],
                'industry_roe_avg': industry_averages['roe'],
                'industry_margin_avg': industry_averages['net_margin']
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error performing peer comparison for {self.symbol}: {e}")
            return {}

    def generate_investment_thesis(self) -> Dict:
        """Generate AI-powered investment thesis"""
        try:
            strengths = []
            concerns = []
            overall_rating = "HOLD"
            
            # Analyze financial metrics
            financial_performance = self.get_financial_performance()
            valuation_metrics = self.get_valuation_metrics()
            balance_sheet = self.get_balance_sheet_analysis()
            growth_analysis = self.get_growth_analysis()
            quality_scores = self.get_quality_scores()
            
            # Evaluate strengths
            if financial_performance.get('roe', 0) and financial_performance['roe'] > 0.15:
                strengths.append("Strong return on equity (>15%)")
            
            if balance_sheet.get('current_ratio', 0) and balance_sheet['current_ratio'] > 1.5:
                strengths.append("Strong liquidity position")
            
            if balance_sheet.get('debt_to_equity', float('inf')) and balance_sheet['debt_to_equity'] < 0.5:
                strengths.append("Conservative debt levels")
            
            if growth_analysis.get('revenue_growth', 0) and growth_analysis['revenue_growth'] > 0.1:
                strengths.append("Strong revenue growth")
            
            if quality_scores.get('piotroski_score', 0) and quality_scores['piotroski_score'] >= 7:
                strengths.append("High financial quality (Piotroski F-Score ≥7)")
            
            # Evaluate concerns
            if valuation_metrics.get('pe_ratio', 0) and valuation_metrics['pe_ratio'] > 25:
                concerns.append("High valuation (P/E > 25)")
            
            if financial_performance.get('revenue_growth_yoy', 0) and financial_performance['revenue_growth_yoy'] < 0:
                concerns.append("Declining revenue growth")
            
            if balance_sheet.get('debt_to_equity', 0) and balance_sheet['debt_to_equity'] > 1.0:
                concerns.append("High debt levels")
            
            if quality_scores.get('altman_z_score', 3) and quality_scores['altman_z_score'] < 1.8:
                concerns.append("Financial distress risk (Low Altman Z-Score)")
            
            # Determine overall rating
            strength_count = len(strengths)
            concern_count = len(concerns)
            
            if strength_count >= 3 and concern_count <= 1:
                overall_rating = "BUY"
            elif strength_count >= 2 and concern_count <= 2:
                overall_rating = "HOLD"
            else:
                overall_rating = "SELL"
            
            return {
                'strengths': strengths[:5],  # Limit to top 5
                'concerns': concerns[:5],    # Limit to top 5
                'overall_rating': overall_rating,
                'strength_score': strength_count,
                'concern_score': concern_count
            }
            
        except Exception as e:
            print(f"Error generating investment thesis for {self.symbol}: {e}")
            return {'strengths': [], 'concerns': [], 'overall_rating': 'UNKNOWN'}

    def _calculate_cagr(self, ending_value: float, beginning_value: float, years: int) -> float:
        """Calculate Compound Annual Growth Rate"""
        if beginning_value <= 0 or ending_value <= 0 or years <= 0:
            return None
        return (ending_value / beginning_value) ** (1/years) - 1

    def _calculate_piotroski_score(self) -> Optional[int]:
        """Calculate Piotroski F-Score (simplified version)"""
        try:
            score = 0
            info = self.info
            financials = self.financials
            cash_flow = self.cash_flow
            
            # Profitability (4 points)
            if info.get('returnOnAssets', 0) > 0:
                score += 1  # Positive ROA
            
            if not cash_flow.empty and 'Operating Cash Flow' in cash_flow.index:
                if cash_flow.loc['Operating Cash Flow'].iloc[0] > 0:
                    score += 1  # Positive operating cash flow
            
            if not financials.empty and 'Net Income' in financials.index:
                net_incomes = financials.loc['Net Income'].dropna()
                if len(net_incomes) >= 2 and net_incomes.iloc[0] > net_incomes.iloc[1]:
                    score += 1  # Increasing ROA
            
            # Leverage, Liquidity and Source of Funds (3 points) - simplified
            if info.get('debtToEquity', float('inf')) < info.get('debtToEquity', float('inf')):  # Simplified
                score += 1
            
            if info.get('currentRatio', 0) > 1.0:
                score += 1  # Current ratio > 1
            
            # Operating Efficiency (2 points) - simplified
            if info.get('grossMargins', 0) > 0.2:  # Gross margin > 20%
                score += 1
            
            return min(score, 9)  # Cap at 9
            
        except Exception:
            return None

    def _calculate_altman_z_score(self) -> Optional[float]:
        """Calculate Altman Z-Score (simplified version)"""
        try:
            info = self.info
            market_cap = info.get('marketCap', 0)
            total_debt = info.get('totalDebt', 0)
            
            if not market_cap or market_cap <= 0:
                return None
            
            # Simplified Z-Score calculation
            current_ratio = info.get('currentRatio', 1.0)
            roe = info.get('returnOnEquity', 0)
            debt_ratio = total_debt / (market_cap + total_debt) if market_cap + total_debt > 0 else 0.5
            
            z_score = (1.2 * (current_ratio - 1)) + (1.4 * roe) + (3.3 * 0.1) + (0.6 * (1 - debt_ratio)) + (1.0 * 0.1)
            
            return round(z_score, 2)
            
        except Exception:
            return None

    def _calculate_quality_rating(self, piotroski_score: Optional[int], altman_z_score: Optional[float]) -> Optional[float]:
        """Calculate overall quality rating (1-10 scale)"""
        try:
            if piotroski_score is None and altman_z_score is None:
                return None
            
            rating = 5.0  # Base rating
            
            if piotroski_score is not None:
                rating += (piotroski_score - 4.5) * 0.5  # Piotroski contributes 0-4.5 points
            
            if altman_z_score is not None:
                if altman_z_score > 2.99:
                    rating += 1.5  # Safe zone
                elif altman_z_score > 1.8:
                    rating += 0.5  # Gray zone
                else:
                    rating -= 1.0  # Distress zone
            
            return round(max(1.0, min(10.0, rating)), 1)
            
        except Exception:
            return None

    def _get_industry_averages(self, industry: str) -> Dict:
        """Get industry average metrics (simplified static data)"""
        # In a real implementation, this would fetch from a comprehensive database
        industry_defaults = {
            'pe': 20.0,
            'roe': 0.12,
            'net_margin': 0.08
        }
        
        # Industry-specific adjustments
        tech_industries = ['Software', 'Technology', 'Internet', 'Semiconductors']
        financial_industries = ['Banks', 'Insurance', 'Financial Services']
        
        if any(tech in industry for tech in tech_industries):
            return {'pe': 25.0, 'roe': 0.18, 'net_margin': 0.15}
        elif any(fin in industry for fin in financial_industries):
            return {'pe': 12.0, 'roe': 0.10, 'net_margin': 0.25}
        else:
            return industry_defaults

    def get_comprehensive_analysis(self) -> Dict:
        """Get complete fundamental analysis"""
        try:
            analysis = {
                'symbol': self.symbol,
                'analysis_date': datetime.now().isoformat(),
                'financial_performance': self.get_financial_performance(),
                'valuation_metrics': self.get_valuation_metrics(),
                'balance_sheet': self.get_balance_sheet_analysis(),
                'growth_analysis': self.get_growth_analysis(),
                'quality_scores': self.get_quality_scores(),
                'peer_comparison': self.get_peer_comparison(),
                'investment_thesis': self.generate_investment_thesis()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error generating comprehensive analysis for {self.symbol}: {e}")
            return {'symbol': self.symbol, 'error': str(e)}


def get_stock_fundamentals_analysis(symbol: str) -> Dict:
    """Main function to get comprehensive stock fundamentals analysis"""
    analyzer = StockFundamentalsAnalyzer(symbol)
    return analyzer.get_comprehensive_analysis()