"""
VWAP Z-Score Analysis Dashboard
Pulls OHLCV data from Binance, computes VWAP, z-scores, and visualizes with Plotly Dash.

Requirements: plotly, dash, pandas, numpy, requests
Install with: pip install plotly dash pandas numpy requests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, List
import requests
import time

# Plotly and Dash
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# BINANCE DATA FETCHING
# ============================================================================

class BinanceDataFetcher:
    """
    Robust Binance SPOT API client wrapper using requests library.
    
    Note: Uses Binance SPOT market data (https://api.binance.com/api/v3)
    NOT futures data. SPOT = actual tradeable pairs on Binance spot market.
    """
    
    # Binance SPOT API endpoint (not futures)
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Interval mapping
    INTERVAL_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'
    }
    
    def __init__(self):
        """Initialize Binance SPOT API fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("Binance SPOT API fetcher initialized")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = '1d',
        limit: int = 365,
        years: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Binance SPOT market (not futures).
        
        Handles API limit of 1000 records per request by making multiple requests.
        
        Args:
            symbol: SPOT trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1d', '4h', '1h', '15m')
            limit: DEPRECATED - use years instead. Number of candles per request
            years: Number of years of data to fetch (default 5)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Returns None on error
        
        Note: This fetches from Binance SPOT market, not Futures.
        Uses endpoint: https://api.binance.com/api/v3/klines
        """
        try:
            if symbol.upper() != symbol:
                symbol = symbol.upper()
            
            # Validate interval
            if interval not in self.INTERVAL_MAP:
                logger.warning(f"Invalid interval {interval}, using 1d")
                interval = '1d'
            
            # Calculate how many candles we need for the requested years
            # Rough estimation: 365 candles per year for 1d interval
            if interval == '1d':
                candles_needed = years * 365
            elif interval == '4h':
                candles_needed = years * 365 * 6  # 6 candles per day
            elif interval == '1h':
                candles_needed = years * 365 * 24  # 24 candles per day
            elif interval == '15m':
                candles_needed = years * 365 * 96  # 96 candles per day
            elif interval == '1w':
                candles_needed = years * 52  # ~52 weeks per year
            else:
                candles_needed = years * 365  # default fallback
            
            # Binance API limit is 1000 per request
            max_per_request = 1000
            num_requests = (candles_needed + max_per_request - 1) // max_per_request
            
            logger.info(f"Will fetch {candles_needed} SPOT candles for {symbol} ({years} years) in {num_requests} requests...")
            
            all_klines = []
            
            # Fetch data in reverse chronological order (newest first)
            for request_num in range(num_requests):
                try:
                    # Binance SPOT klines endpoint (not futures)
                    url = f"{self.BASE_URL}/klines"
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': max_per_request
                    }
                    
                    # If not the first request, use endTime to fetch older data
                    if request_num > 0 and all_klines:
                        # Get timestamp of oldest candle and fetch before it
                        oldest_timestamp = all_klines[0][0]  # First element is timestamp
                        params['endTime'] = oldest_timestamp - 1
                    
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    klines = response.json()
                    
                    if not klines:
                        logger.warning(f"No more data available for {symbol}")
                        break
                    
                    all_klines = klines + all_klines  # Prepend to maintain chronological order
                    logger.info(f"Request {request_num + 1}/{num_requests}: fetched {len(klines)} candles")
                    
                    # Small delay to avoid rate limiting
                    if request_num < num_requests - 1:
                        time.sleep(0.2)
                
                except Exception as e:
                    logger.error(f"Error in request {request_num + 1}: {e}")
                    if all_klines:
                        break
                    else:
                        return None
            
            if not all_klines:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Parse klines into DataFrame
            df = pd.DataFrame(
                all_klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ]
            )
            
            # Convert to numeric and proper types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} SPOT candles for {symbol}")
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None


# ============================================================================
# VWAP AND Z-SCORE CALCULATIONS
# ============================================================================

def compute_vwap(df: pd.DataFrame, window: int = 365) -> pd.Series:
    """
    Compute Volume-Weighted Average Price (VWAP) over a rolling window.
    
    VWAP = Σ(typical_price * volume) / Σ(volume)
    where typical_price = (high + low + close) / 3
    
    Args:
        df: DataFrame with columns: high, low, close, volume
        window: Rolling window size (default 365 bars)
    
    Returns:
        Series of VWAP values
    """
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    vwap = (
        df['tp_volume'].rolling(window=window, min_periods=1).sum() /
        df['volume'].rolling(window=window, min_periods=1).sum()
    )
    
    return vwap


def compute_z_score(
    close: pd.Series,
    vwap: pd.Series,
    rolling_std: pd.Series
) -> pd.Series:
    """
    Compute z-score of price versus VWAP.
    
    z = (P - VWAP) / σ
    where P is close price, VWAP is computed for that bar, σ is rolling std
    
    Args:
        close: Series of close prices
        vwap: Series of VWAP values
        rolling_std: Series of rolling standard deviations
    
    Returns:
        Series of z-score values
    """
    # Avoid division by zero
    z_score = np.where(
        rolling_std > 0,
        (close - vwap) / rolling_std,
        0
    )
    return pd.Series(z_score, index=close.index)


def process_market_data(
    df: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Process market data: compute VWAP and z-scores for multiple windows.
    
    Args:
        df: DataFrame with OHLCV data
        windows: List of rolling window sizes for VWAP and std
                Default: [365, 180, 90, 30] (1yr, 6mo, 3mo, 1mo)
    
    Returns:
        DataFrame with added columns: vwap_{window}, rolling_std_{window}, z_score_{window}
        for each window in the list
    """
    if windows is None:
        windows = [365, 180, 90, 30]
    
    df = df.copy()
    
    for window in windows:
        # Compute VWAP
        df[f'vwap_{window}'] = compute_vwap(df, window=window)
        
        # Compute rolling standard deviation of close price
        df[f'rolling_std_{window}'] = df['close'].rolling(window=window, min_periods=1).std()
        
        # Compute z-score
        df[f'z_score_{window}'] = compute_z_score(
            df['close'],
            df[f'vwap_{window}'],
            df[f'rolling_std_{window}']
        )
    
    return df


# ============================================================================
# MTM (Market Tension Map) CALCULATIONS
# ============================================================================

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=length).mean()
    return atr


def calculate_mtm_indicators(df: pd.DataFrame, period: int = 55) -> pd.DataFrame:
    """
    Calculate Market Tension Map indicators (from lubluniky/falx repo).
    
    MTM combines:
    - Relative Volatility (price std / price) - inverted scoring
    - Volume Score (normalized volume in range)
    - Tension Index (average of above scores)
    
    Higher tension_index = compressed spring = potential breakout signal
    
    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 55 from MTM repo)
    
    Returns:
        DataFrame with added MTM columns: tension_index, volatility_score, volume_score, atr
    """
    df = df.copy()
    
    # Relative Volatility
    df['relative_volatility'] = df['close'].rolling(window=period).std() / df['close']
    
    # Volatility Score (inverted - lower volatility = higher score = more tension)
    rv_max = df['relative_volatility'].rolling(window=period).max()
    rv_min = df['relative_volatility'].rolling(window=period).min()
    df['volatility_score'] = 100 * (rv_max - df['relative_volatility']) / (rv_max - rv_min + 1e-10)
    
    # Volume Score (normalized - higher volume = higher score)
    vol_max = df['volume'].rolling(window=period).max()
    vol_min = df['volume'].rolling(window=period).min()
    df['volume_score'] = 100 * (df['volume'] - vol_min) / (vol_max - vol_min + 1e-10)
    
    # Tension Index (combination - higher = more compressed spring)
    df['tension_index'] = (df['volatility_score'] + df['volume_score']) / 2
    
    # ATR for context
    df['atr'] = calculate_atr(df, length=14)
    
    return df


# ============================================================================
# BACKTEST LOGIC
# ============================================================================

def run_mean_reversion_backtest(
    df: pd.DataFrame,
    window: int
) -> dict:
    """
    Run mean-reversion backtest for a specific VWAP window.
    
    Strategy:
    - LONG when z-score < -2, EXIT when z-score crosses 0
    - SHORT when z-score > 2, EXIT when z-score crosses 0
    
    Args:
        df: DataFrame with 'close', 'timestamp', and f'z_score_{window}' columns
        window: VWAP window size
    
    Returns:
        Dictionary with trades list and equity curve
    """
    z_col = f'z_score_{window}'
    
    if z_col not in df.columns:
        return {'trades': [], 'equity_curve': [1.0] * len(df)}
    
    trades = []
    position = None  # None, 'long', or 'short'
    entry_price = None
    entry_idx = None
    entry_date = None
    
    equity = 1.0
    equity_curve = []
    
    for idx, row in df.iterrows():
        z = row[z_col]
        close = row['close']
        timestamp = row['timestamp']
        
        # Check for exit conditions first
        if position == 'long' and z >= 0:
            # Exit long
            pnl_pct = (close - entry_price) / entry_price
            pnl_raw = close - entry_price
            equity *= (1 + pnl_pct)
            
            trades.append({
                'type': 'long',
                'entry_date': entry_date,
                'exit_date': timestamp,
                'entry_price': entry_price,
                'exit_price': close,
                'entry_idx': entry_idx,
                'exit_idx': idx,
                'bars_held': idx - entry_idx,
                'pnl_pct': pnl_pct * 100,
                'pnl_raw': pnl_raw,
                'equity_after': equity
            })
            
            position = None
            entry_price = None
            entry_idx = None
            entry_date = None
        
        elif position == 'short' and z <= 0:
            # Exit short
            pnl_pct = (entry_price - close) / entry_price
            pnl_raw = entry_price - close
            equity *= (1 + pnl_pct)
            
            trades.append({
                'type': 'short',
                'entry_date': entry_date,
                'exit_date': timestamp,
                'entry_price': entry_price,
                'exit_price': close,
                'entry_idx': entry_idx,
                'exit_idx': idx,
                'bars_held': idx - entry_idx,
                'pnl_pct': pnl_pct * 100,
                'pnl_raw': pnl_raw,
                'equity_after': equity
            })
            
            position = None
            entry_price = None
            entry_idx = None
            entry_date = None
        
        # Check for entry conditions (only if no position)
        if position is None:
            if z < -2:
                # Enter long
                position = 'long'
                entry_price = close
                entry_idx = idx
                entry_date = timestamp
            
            elif z > 2:
                # Enter short
                position = 'short'
                entry_price = close
                entry_idx = idx
                entry_date = timestamp
        
        equity_curve.append(equity)
    
    # Close any open position at the end
    if position is not None:
        last_row = df.iloc[-1]
        close = last_row['close']
        timestamp = last_row['timestamp']
        idx = df.index[-1]
        
        if position == 'long':
            pnl_pct = (close - entry_price) / entry_price
            pnl_raw = close - entry_price
        else:  # short
            pnl_pct = (entry_price - close) / entry_price
            pnl_raw = entry_price - close
        
        equity *= (1 + pnl_pct)
        
        trades.append({
            'type': position,
            'entry_date': entry_date,
            'exit_date': timestamp,
            'entry_price': entry_price,
            'exit_price': close,
            'entry_idx': entry_idx,
            'exit_idx': idx,
            'bars_held': idx - entry_idx,
            'pnl_pct': pnl_pct * 100,
            'pnl_raw': pnl_raw,
            'equity_after': equity
        })
        
        equity_curve[-1] = equity
    
    return {
        'trades': trades,
        'equity_curve': equity_curve
    }


def run_mtf_vwap_mtm_backtest_simple(
    df_daily: pd.DataFrame,
    df_15m: pd.DataFrame,
    window: int = 30,
    mtm_threshold: float = 70.0,
    lookback_bars: int = 5
) -> dict:
    """
    Simplified MTF backtest: Daily VWAP z-score + 15m MTM confirmation.
    
    This is a SIMPLIFIED version for quick integration. Only works with window=30.
    For full multi-window MTF, see vwap_mtf_mtm.py
    
    Strategy:
    1. Daily VWAP z-score < -2 (long setup) or > 2 (short setup)
    2. Within lookback_bars days, check if 15m MTM tension_index > threshold
    3. Enter if both conditions met
    4. Exit when z-score returns to 0
    
    Args:
        df_daily: Daily OHLCV with z_score_30 column
        df_15m: 15m OHLCV data
        window: VWAP window (only 30 supported in simplified version)
        mtm_threshold: MTM tension threshold for confirmation (default 70)
        lookback_bars: Days to wait for MTM confirmation (default 5)
    
    Returns:
        Dictionary with trades list and equity curve
    """
    if window != 30:
        logger.warning(f"Simplified MTF only supports window=30, got {window}. Using 30.")
        window = 30
    
    z_col = f'z_score_{window}'
    
    if z_col not in df_daily.columns:
        logger.error(f"Column {z_col} not found in daily data")
        return {'trades': [], 'equity_curve': [1.0] * len(df_daily)}
    
    # Calculate MTM on 15m data
    df_15m = calculate_mtm_indicators(df_15m.copy(), period=55)
    
    # Create timestamp index for quick lookup
    df_15m_indexed = df_15m.set_index('timestamp')
    
    trades = []
    position = None
    entry_price = None
    entry_date = None
    entry_idx = None
    mtm_at_entry = None
    
    equity = 1.0
    equity_curve = []
    
    for idx, row in df_daily.iterrows():
        z = row[z_col]
        close = row['close']
        timestamp = row['timestamp']
        
        # Exit logic (same as vanilla VWAP)
        if position == 'long' and z >= 0:
            pnl_pct = (close - entry_price) / entry_price
            equity *= (1 + pnl_pct)
            
            trades.append({
                'type': 'long',
                'entry_date': entry_date,
                'exit_date': timestamp,
                'entry_price': entry_price,
                'exit_price': close,
                'bars_held': idx - entry_idx,
                'pnl_pct': pnl_pct * 100,
                'equity_after': equity,
                'mtm_at_entry': mtm_at_entry
            })
            
            position = None
            
        elif position == 'short' and z <= 0:
            pnl_pct = (entry_price - close) / entry_price
            equity *= (1 + pnl_pct)
            
            trades.append({
                'type': 'short',
                'entry_date': entry_date,
                'exit_date': timestamp,
                'entry_price': entry_price,
                'exit_price': close,
                'bars_held': idx - entry_idx,
                'pnl_pct': pnl_pct * 100,
                'equity_after': equity,
                'mtm_at_entry': mtm_at_entry
            })
            
            position = None
        
        # Entry logic (only if no position)
        if position is None:
            setup_triggered = False
            setup_type = None
            
            if z < -2:
                setup_triggered = True
                setup_type = 'long'
            elif z > 2:
                setup_triggered = True
                setup_type = 'short'
            
            if setup_triggered:
                # Check MTM confirmation in next lookback_bars days
                mtm_confirmed = False
                mtm_value = 0
                
                # Look at 15m data around this daily bar
                day_start = pd.Timestamp(timestamp).floor('D')
                day_end = day_start + pd.Timedelta(days=lookback_bars)
                
                try:
                    # Get 15m bars in this window
                    m15_window = df_15m_indexed.loc[
                        (df_15m_indexed.index >= day_start) & 
                        (df_15m_indexed.index <= day_end)
                    ]
                    
                    if not m15_window.empty and 'tension_index' in m15_window.columns:
                        max_tension = m15_window['tension_index'].max()
                        
                        if not pd.isna(max_tension) and max_tension > mtm_threshold:
                            mtm_confirmed = True
                            mtm_value = max_tension
                except Exception as e:
                    logger.debug(f"MTM lookup error: {e}")
                    mtm_confirmed = False
                
                # Enter if MTM confirmed
                if mtm_confirmed:
                    position = setup_type
                    entry_price = close
                    entry_date = timestamp
                    entry_idx = idx
                    mtm_at_entry = mtm_value
        
        equity_curve.append(equity)
    
    # Close open position at end
    if position is not None:
        last_row = df_daily.iloc[-1]
        close = last_row['close']
        timestamp = last_row['timestamp']
        
        if position == 'long':
            pnl_pct = (close - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - close) / entry_price
        
        equity *= (1 + pnl_pct)
        
        trades.append({
            'type': position,
            'entry_date': entry_date,
            'exit_date': timestamp,
            'entry_price': entry_price,
            'exit_price': close,
            'bars_held': len(df_daily) - 1 - entry_idx,
            'pnl_pct': pnl_pct * 100,
            'equity_after': equity,
            'mtm_at_entry': mtm_at_entry
        })
        
        equity_curve[-1] = equity
    
    return {
        'trades': trades,
        'equity_curve': equity_curve
    }


def calculate_backtest_metrics(backtest_results: dict, df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Args:
        backtest_results: Output from run_mean_reversion_backtest
        df: Original dataframe (for time calculations)
    
    Returns:
        Dictionary with all performance metrics
    """
    trades = backtest_results['trades']
    equity_curve = backtest_results['equity_curve']
    
    if not trades:
        return {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'avg_pnl_raw': 0.0,
            'median_pnl_pct': 0.0,
            'min_pnl_pct': 0.0,
            'max_pnl_pct': 0.0,
            'std_pnl_pct': 0.0,
            'avg_bars_held': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'std_error': 0.0,
            'exposure_pct': 0.0
        }
    
    # Basic counts
    total_trades = len(trades)
    long_trades = sum(1 for t in trades if t['type'] == 'long')
    short_trades = sum(1 for t in trades if t['type'] == 'short')
    
    # PnL statistics
    pnls_pct = [t['pnl_pct'] for t in trades]
    pnls_raw = [t['pnl_raw'] for t in trades]
    
    winning_trades = [p for p in pnls_pct if p > 0]
    losing_trades = [p for p in pnls_pct if p < 0]
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
    
    avg_pnl_pct = np.mean(pnls_pct)
    avg_pnl_raw = np.mean(pnls_raw)
    median_pnl_pct = np.median(pnls_pct)
    min_pnl_pct = np.min(pnls_pct)
    max_pnl_pct = np.max(pnls_pct)
    std_pnl_pct = np.std(pnls_pct)
    
    # Time in trade
    bars_held = [t['bars_held'] for t in trades]
    avg_bars_held = np.mean(bars_held)
    
    # Total return
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - 1.0) * 100
    
    # Sharpe Ratio (annualized, assuming daily data)
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    returns = returns[returns != 0]  # Remove zero returns
    
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino_ratio = 0.0
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peak) / peak
    max_drawdown_pct = np.min(drawdown) * 100
    
    # Profit Factor
    total_wins = sum(winning_trades) if winning_trades else 0
    total_losses = abs(sum(losing_trades)) if losing_trades else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Expectancy and Standard Error
    expectancy = avg_pnl_pct
    std_error = std_pnl_pct / np.sqrt(total_trades) if total_trades > 0 else 0.0
    
    # Exposure (% of time in market)
    total_bars_in_trade = sum(bars_held)
    total_bars = len(df)
    exposure_pct = (total_bars_in_trade / total_bars * 100) if total_bars > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': round(win_rate, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 2),
        'avg_pnl_raw': round(avg_pnl_raw, 2),
        'median_pnl_pct': round(median_pnl_pct, 2),
        'min_pnl_pct': round(min_pnl_pct, 2),
        'max_pnl_pct': round(max_pnl_pct, 2),
        'std_pnl_pct': round(std_pnl_pct, 2),
        'avg_bars_held': round(avg_bars_held, 1),
        'total_return_pct': round(total_return_pct, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        'expectancy': round(expectancy, 2),
        'std_error': round(std_error, 2),
        'exposure_pct': round(exposure_pct, 2)
    }


# ============================================================================
# PLOTLY CHART GENERATION
# ============================================================================

def get_z_score_color(z_value: float) -> str:
    """
    Determine color based on z-score zone.
    
    Green: z < -2 (risk-on)
    Red: z > 2 (risk-off)
    Black: -0.5 <= z <= 0.5 (mean zone)
    Light blue: -2 <= z < -0.5 (neutral-bullish)
    Light orange: 0.5 < z <= 2 (neutral-bearish)
    
    Args:
        z_value: Z-score value
    
    Returns:
        Color string (hex or named)
    """
    if z_value < -2:
        return '#00AA00'  # Green
    elif z_value > 2:
        return '#FF0000'  # Red
    elif -0.5 <= z_value <= 0.5:
        return '#000000'  # Black
    elif z_value < -0.5:
        return '#1f77b4'  # Light blue (bullish gradient)
    else:
        return '#ff7f0e'  # Light orange (bearish gradient)


def create_multi_window_charts(
    df: pd.DataFrame,
    symbol: str,
    windows: List[int] = None
) -> go.Figure:
    """
    Create ONE OHLC chart at top, then 4 Z-score panels below (one per VWAP window).
    
    Layout:
    - Row 1-2: OHLC candlestick with volume overlay
    - Row 3: Z-Score for 365-bar VWAP
    - Row 4: Z-Score for 180-bar VWAP
    - Row 5: Z-Score for 90-bar VWAP
    - Row 6: Z-Score for 30-bar VWAP
    
    Args:
        df: DataFrame with OHLCV and computed VWAP/z-score columns
        symbol: Trading pair symbol for title
        windows: List of window sizes (default [365, 180, 90, 30])
    
    Returns:
        Plotly Figure with 1 OHLC panel + 4 Z-score panels
    """
    if windows is None:
        windows = [365, 180, 90, 30]
    
    # Create subplots: 
    # Row 1: OHLC candlestick
    # Row 2: Volume
    # Rows 3-6: Z-scores for each window
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.1, 0.11, 0.11, 0.11, 0.11],
        subplot_titles=[
            f"OHLC Chart - {symbol} (1D SPOT)",
            "Volume",
            "Z-Score VWAP 365",
            "Z-Score VWAP 180",
            "Z-Score VWAP 90",
            "Z-Score VWAP 30"
        ]
    )
    
    # ===== OHLC Candlestick (Row 1) =====
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # ===== Volume Bars (Row 2) =====
    colors = ['#26A69A' if close >= open_ else '#FF6692'
              for close, open_ in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker=dict(color=colors, opacity=0.5),
            showlegend=False,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # ===== Z-Score Lines for Each Window (Rows 3-6) =====
    for idx, window in enumerate(windows):
        zscore_row = idx + 3  # Rows: 3, 4, 5, 6
        z_col = f'z_score_{window}'
        
        # Draw line segments with individual colors
        for i in range(len(df) - 1):
            color = get_z_score_color(df.iloc[i][z_col])
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[i:i+2]['timestamp'],
                    y=df.iloc[i:i+2][z_col],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>Z-{window}: %{{y:.2f}}<extra></extra>'
                ),
                row=zscore_row, col=1
            )
        
        # ===== Reference Lines for Z-Score =====
        for z_val in [-3, -2, -1, 0, 1, 2, 3]:
            line_style = 'solid' if z_val == 0 else 'dash'
            line_width = 2 if z_val == 0 else 1
            line_color = '#808080' if z_val == 0 else '#CCCCCC'
            
            fig.add_hline(
                y=z_val,
                line_dash=line_style,
                line_width=line_width,
                line_color=line_color,
                row=zscore_row, col=1
            )
        
        # ===== Zone Backgrounds for Z-Score =====
        fig.add_hrect(
            y0=-3, y1=-2,
            fillcolor="green", opacity=0.05,
            layer="below", line_width=0,
            row=zscore_row, col=1
        )
        fig.add_hrect(
            y0=2, y1=3,
            fillcolor="red", opacity=0.05,
            layer="below", line_width=0,
            row=zscore_row, col=1
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text=f"Z-{window}b", row=zscore_row, col=1)
    
    # Update layout
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=6, col=1)
    
    fig.update_layout(
        title=f"Multi-Window VWAP Z-Score Analysis - {symbol} (SPOT)",
        height=1400,
        hovermode='x unified',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    return fig


def create_equity_curves_chart(
    backtest_results_dict: dict,
    df: pd.DataFrame,
    windows: List[int] = None
) -> go.Figure:
    """
    Create equity curve chart comparing all VWAP windows.
    
    Args:
        backtest_results_dict: Dict with window as key, backtest results as value
        df: Original dataframe with timestamps
        windows: List of windows
    
    Returns:
        Plotly Figure with multiple equity curves
    """
    if windows is None:
        windows = [365, 180, 90, 30]
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, window in enumerate(windows):
        if window not in backtest_results_dict:
            continue
        
        equity_curve = backtest_results_dict[window]['equity_curve']
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=equity_curve,
                mode='lines',
                name=f'VWAP {window}',
                line=dict(color=colors[idx], width=2),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Equity: %{y:.3f}<extra></extra>'
            )
        )
    
    # Add horizontal line at 1.0 (starting equity)
    fig.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='gray',
        annotation_text='Start',
        annotation_position='right'
    )
    
    fig.update_layout(
        title="Cumulative Equity Curves by VWAP Window",
        xaxis_title="Date",
        yaxis_title="Equity Multiplier",
        height=400,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(__name__)

# Global data store (in production, use a proper database)
data_cache = {}

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("VWAP Z-Score Analysis Dashboard (Binance SPOT)", style={'textAlign': 'center'}),
        html.Hr(),
    ]),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Symbol:"),
            dcc.Input(
                id='symbol-input',
                type='text',
                value='BTCUSDT',
                style={'marginRight': '20px', 'padding': '8px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Interval:"),
            dcc.Dropdown(
                id='interval-dropdown',
                options=[
                    {'label': '1 Day', 'value': '1d'},
                    {'label': '4 Hours', 'value': '4h'},
                    {'label': '1 Hour', 'value': '1h'},
                    {'label': '15 Min', 'value': '15m'},
                ],
                value='1d',
                style={'width': '150px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Div([
            html.Label("Years of Data:"),
            dcc.Dropdown(
                id='years-dropdown',
                options=[
                    {'label': '1 Year', 'value': 1},
                    {'label': '2 Years', 'value': 2},
                    {'label': '3 Years', 'value': 3},
                    {'label': '5 Years', 'value': 5},
                ],
                value=5,
                style={'width': '150px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        html.Button(
            'Load Data',
            id='load-button',
            n_clicks=0,
            style={
                'padding': '8px 20px',
                'backgroundColor': '#007BFF',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontSize': '14px'
            }
        ),
    ], style={
        'display': 'flex',
        'justifyContent': 'flex-start',
        'alignItems': 'center',
        'padding': '20px',
        'backgroundColor': '#F8F9FA',
        'borderRadius': '8px',
        'marginBottom': '20px'
    }),
    
    # Status message
    html.Div(id='status-message', style={
        'padding': '10px',
        'marginBottom': '20px',
        'fontSize': '14px',
        'color': '#666'
    }),
    
    # Charts - Multi-window analysis
    html.Div([
        dcc.Graph(id='multi-window-chart')
    ]),
    
    # Backtest Statistics Section
    html.Div([
        html.Hr(style={'marginTop': '40px', 'marginBottom': '20px'}),
        html.H2("📊 Backtest Statistics & Performance Metrics", 
                style={'textAlign': 'center', 'color': '#333', 'marginBottom': '20px'}),
        
        html.P([
            "Mean-Reversion Strategy: ",
            html.B("LONG"), " when z-score < -2, ",
            html.B("SHORT"), " when z-score > 2, ",
            html.B("EXIT"), " when z-score crosses 0."
        ], style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666', 'marginBottom': '30px'}),
        
        # Statistics Table
        html.Div([
            dash_table.DataTable(
                id='backtest-stats-table',
                columns=[],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontSize': '13px',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': '#007BFF',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '14px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9f9f9'
                    }
                ]
            )
        ], style={'marginBottom': '30px'}),
        
        # Equity Curves Chart
        html.Div([
            dcc.Graph(id='equity-curves-chart')
        ])
    ]),
    
    # Hidden div to store data
    dcc.Store(id='data-store', data={})
    
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px', 'margin': '0 auto'})


# ============================================================================
# DASH CALLBACKS
# ============================================================================

@app.callback(
    [
        Output('multi-window-chart', 'figure'),
        Output('backtest-stats-table', 'columns'),
        Output('backtest-stats-table', 'data'),
        Output('backtest-stats-table', 'style_data_conditional'),
        Output('equity-curves-chart', 'figure'),
        Output('status-message', 'children'),
        Output('data-store', 'data')
    ],
    Input('load-button', 'n_clicks'),
    [
        State('symbol-input', 'value'),
        State('interval-dropdown', 'value'),
        State('years-dropdown', 'value')
    ],
    prevent_initial_call=False
)
def update_charts(n_clicks, symbol, interval, years):
    """
    Fetch multi-year data, process metrics, run backtests, and update all charts.
    
    Args:
        n_clicks: Number of button clicks
        symbol: Trading pair symbol
        interval: Kline interval
        years: Number of years of data to fetch
    
    Returns:
        Tuple of (chart, table_cols, table_data, table_styles, equity_chart, status, stored_data)
    """
    empty_table_cols = []
    empty_table_data = []
    empty_table_styles = []
    empty_equity_fig = go.Figure().add_annotation(text="No data")
    
    if not symbol or not interval or not years:
        return (
            go.Figure().add_annotation(text="Invalid input"),
            empty_table_cols,
            empty_table_data,
            empty_table_styles,
            empty_equity_fig,
            "❌ Invalid input parameters",
            {}
        )
    
    try:
        # Fetch data from Binance
        fetcher = BinanceDataFetcher()
        
        logger.info(f"Loading {years} years of {interval} data for {symbol}...")
        
        df = fetcher.fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            years=years
        )
        
        if df is None or df.empty:
            return (
                go.Figure().add_annotation(text=f"No data for {symbol}"),
                empty_table_cols,
                empty_table_data,
                empty_table_styles,
                empty_equity_fig,
                f"❌ Failed to fetch data for {symbol}. Check symbol name.",
                {}
            )
        
        # Process data with multiple VWAP windows
        windows = [365, 180, 90, 30]
        df = process_market_data(df, windows=windows)
        
        # Generate multi-window chart
        fig = create_multi_window_charts(df, symbol, windows=windows)
        
        # ===== RUN BACKTESTS =====
        logger.info("Running backtests for all windows...")
        backtest_results = {}
        metrics_list = []
        
        for window in windows:
            # Run backtest
            bt_results = run_mean_reversion_backtest(df, window)
            backtest_results[window] = bt_results
            
            # Calculate metrics
            metrics = calculate_backtest_metrics(bt_results, df)
            metrics['window'] = f"{window}b"
            metrics_list.append(metrics)
        
        # ===== CREATE BACKTEST TABLE =====
        table_columns = [
            {'name': 'Window', 'id': 'window'},
            {'name': 'Trades', 'id': 'total_trades'},
            {'name': 'Longs', 'id': 'long_trades'},
            {'name': 'Shorts', 'id': 'short_trades'},
            {'name': 'Win Rate %', 'id': 'win_rate'},
            {'name': 'Avg PnL %', 'id': 'avg_pnl_pct'},
            {'name': 'Med PnL %', 'id': 'median_pnl_pct'},
            {'name': 'Std PnL %', 'id': 'std_pnl_pct'},
            {'name': 'Avg Bars', 'id': 'avg_bars_held'},
            {'name': 'Total Ret %', 'id': 'total_return_pct'},
            {'name': 'Sharpe', 'id': 'sharpe_ratio'},
            {'name': 'Sortino', 'id': 'sortino_ratio'},
            {'name': 'Max DD %', 'id': 'max_drawdown_pct'},
            {'name': 'Profit Factor', 'id': 'profit_factor'},
            {'name': 'Exposure %', 'id': 'exposure_pct'}
        ]
        
        # Style conditional - highlight best values
        # Find best values for highlighting
        if metrics_list:
            best_win_rate = max(m['win_rate'] for m in metrics_list)
            best_sharpe = max(m['sharpe_ratio'] for m in metrics_list)
            best_return = max(m['total_return_pct'] for m in metrics_list)
            
            style_conditional = [
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                },
                {
                    'if': {
                        'filter_query': f'{{win_rate}} = {best_win_rate}',
                        'column_id': 'win_rate'
                    },
                    'backgroundColor': '#d4edda',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': f'{{sharpe_ratio}} = {best_sharpe}',
                        'column_id': 'sharpe_ratio'
                    },
                    'backgroundColor': '#d4edda',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': f'{{total_return_pct}} = {best_return}',
                        'column_id': 'total_return_pct'
                    },
                    'backgroundColor': '#d4edda',
                    'fontWeight': 'bold'
                }
            ]
        else:
            style_conditional = []
        
        # ===== CREATE EQUITY CURVES CHART =====
        equity_fig = create_equity_curves_chart(backtest_results, df, windows)
        
        # Status message with latest values from all windows
        latest_close = df.iloc[-1]['close']
        status_parts = [f"✅ Loaded {len(df)} candles for {symbol} ({years} years) | Latest Close: {latest_close:.2f}"]
        
        for window in windows:
            latest_vwap = df.iloc[-1][f'vwap_{window}']
            latest_z = df.iloc[-1][f'z_score_{window}']
            status_parts.append(f"{window}b: VWAP={latest_vwap:.2f}, Z={latest_z:.2f}")
        
        status = " | ".join(status_parts)
        
        # Store data for potential further use
        data_dict = {
            'symbol': symbol,
            'interval': interval,
            'years': years,
            'rows': len(df),
            'windows': windows
        }
        
        return fig, table_columns, metrics_list, style_conditional, equity_fig, status, data_dict
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        return (
            go.Figure().add_annotation(text=error_msg),
            empty_table_cols,
            empty_table_data,
            empty_table_styles,
            empty_equity_fig,
            error_msg,
            {}
        )


# ============================================================================
# APP ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting VWAP Z-Score Dashboard...")
    logger.info("Open your browser and navigate to http://127.0.0.1:8050/")
    app.run(debug=True, port=8050)
