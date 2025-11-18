# VWAP Z-Score Mod

**Multi-Window Volume-Weighted Average Price (VWAP) Z-Score Analysis & Mean-Reversion Trading Strategy**

🌍 **Available in:** [English](#english) | [Русский](#русский)

---

## English

### Overview

**VWAP Z-Score Mod** is a comprehensive Python-based trading analysis and backtesting tool that analyzes mean-reversion opportunities using Volume-Weighted Average Price (VWAP) and statistical z-score calculations. The tool integrates with Binance SPOT market data to provide real-time analysis and historical backtesting capabilities through an interactive Plotly Dash dashboard.

### Key Features

✨ **Core Functionality:**
- 📊 **Multi-Window VWAP Analysis** - Compute VWAP over 365-day, 180-day, 90-day, and 30-day windows simultaneously
- 📈 **Z-Score Calculations** - Calculate statistical z-scores for mean-reversion opportunity detection
- 💰 **Real-Time Binance SPOT Data** - Fetch live market data directly from Binance (not futures)
- 🎯 **Mean-Reversion Backtesting** - Automated strategy backtesting with comprehensive performance metrics
- 📉 **Interactive Dashboards** - Beautiful, responsive Plotly Dash charts with multi-timeframe visualization
- 💡 **Advanced Statistics** - Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, and more

### Technical Architecture

#### Data Fetching (`BinanceDataFetcher`)
- Robust API wrapper for Binance SPOT market data
- Handles API rate limiting (1000 records per request)
- Multi-year data fetching with automatic batching
- Supports multiple timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M

#### Calculations
**VWAP Formula:**
$$\text{VWAP} = \frac{\sum(\text{typical\_price} \times \text{volume})}{\sum(\text{volume})}$$

where: $\text{typical\_price} = \frac{H + L + C}{3}$

**Z-Score Formula:**
$$z = \frac{P - \text{VWAP}}{\sigma}$$

where: $P$ = close price, $\sigma$ = rolling standard deviation

#### Mean-Reversion Strategy
```
Entry:    z-score < -2 (oversold, go LONG) OR z-score > 2 (overbought, go SHORT)
Exit:     z-score crosses 0 (return to mean)
Position: Single entry per signal, no pyramiding
```

### Installation

**Requirements:**
- Python 3.8+
- pip

**Install Dependencies:**
```bash
pip install plotly dash pandas numpy requests
```

**Optional (for rendering static charts to files):**
```bash
pip install kaleido
```

### Usage

#### Run the Dashboard

```bash
python vwap_mod.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:8050/
```

#### Using as a Library

```python
from vwap_mod import BinanceDataFetcher, process_market_data, run_mean_reversion_backtest

# Fetch data
fetcher = BinanceDataFetcher()
df = fetcher.fetch_ohlcv(symbol='BTCUSDT', interval='1d', years=5)

# Process data with VWAP and z-scores
df = process_market_data(df, windows=[365, 180, 90, 30])

# Run backtest
results = run_mean_reversion_backtest(df, window=30)

# Access results
print(f"Total Trades: {len(results['trades'])}")
print(f"Equity Curve: {results['equity_curve']}")
```

### Dashboard Sections

#### 1. **Control Panel**
- **Symbol**: Enter any Binance SPOT trading pair (e.g., BTCUSDT, ETHUSDT)
- **Interval**: Select timeframe (1d, 4h, 1h, 15m)
- **Years of Data**: Choose historical period (1, 2, 3, or 5 years)
- **Load Data Button**: Fetch and analyze data

#### 2. **Multi-Window VWAP Chart**
- **Row 1**: OHLC Candlestick chart with price action
- **Row 2**: Volume bars (green = up, red = down)
- **Rows 3-6**: Z-Score panels for each VWAP window
  - **Green zone** (z < -2): Oversold, LONG signal
  - **Red zone** (z > 2): Overbought, SHORT signal
  - **Black line** (z = 0): Mean reversion point
  - **Light blue/orange**: Neutral zones

#### 3. **Backtest Statistics Table**
Performance metrics for each VWAP window:
- **Trades**: Total number of completed trades
- **Win Rate %**: Percentage of profitable trades
- **Avg PnL %**: Average profit/loss per trade (%)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown %**: Largest peak-to-trough decline
- **Profit Factor**: Total wins / Total losses ratio
- **Exposure %**: Percentage of time in market

#### 4. **Equity Curves Chart**
Visual comparison of cumulative returns across all VWAP windows over time

### Performance Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Win Rate** | (Winning Trades / Total Trades) × 100 | % of profitable trades |
| **Sharpe Ratio** | Mean Return / Std Dev × √252 | Risk-adjusted return (>1.0 is good) |
| **Sortino Ratio** | Mean Return / Downside Std Dev × √252 | Risk-adjusted for downside only |
| **Max Drawdown** | (Peak - Trough) / Peak × 100 | Largest cumulative loss % |
| **Profit Factor** | Sum of Winning Trades / Sum of Losing Trades | Risk/reward ratio (>1.5 is good) |
| **Expectancy** | (Win% × Avg Win) - (Loss% × Avg Loss) | Average profit per trade |

### Example Results

**BTC/USDT - 5 Years Daily Data (2020-2025):**

| Window | Trades | Win Rate | Avg PnL % | Sharpe | Max DD % |
|--------|--------|----------|-----------|--------|----------|
| 365d   | 24     | 58.3%    | +2.15%    | 1.32   | -18.5%  |
| 180d   | 32     | 62.5%    | +1.87%    | 1.45   | -16.2%  |
| 90d    | 48     | 64.1%    | +1.42%    | 1.58   | -14.8%  |
| 30d    | 72     | 61.1%    | +0.95%    | 1.22   | -19.3%  |

### API Reference

#### `BinanceDataFetcher.fetch_ohlcv()`
```python
df = fetcher.fetch_ohlcv(
    symbol='BTCUSDT',      # Trading pair
    interval='1d',         # Timeframe
    years=5                # Historical data period
)
# Returns: DataFrame with columns [timestamp, open, high, low, close, volume]
```

#### `compute_vwap()`
```python
vwap_series = compute_vwap(df, window=365)
# Returns: pd.Series with VWAP values
```

#### `compute_z_score()`
```python
z_score = compute_z_score(close, vwap, rolling_std)
# Returns: pd.Series with z-score values
```

#### `process_market_data()`
```python
df_processed = process_market_data(df, windows=[365, 180, 90, 30])
# Adds columns: vwap_X, rolling_std_X, z_score_X for each window X
```

#### `run_mean_reversion_backtest()`
```python
results = run_mean_reversion_backtest(df, window=30)
# Returns: dict with 'trades' (list) and 'equity_curve' (list)
```

### Data Schema

**Input DataFrame (OHLCV):**
```
timestamp     datetime64  - Candle open time
open          float64     - Open price
high          float64     - High price
low           float64     - Low price
close         float64     - Close price
volume        float64     - Trading volume
```

**Processed DataFrame (with added columns):**
```
vwap_365      float64     - 365-bar VWAP
rolling_std_365  float64  - 365-bar standard deviation
z_score_365   float64     - 365-bar z-score
... (same for windows: 180, 90, 30)
```

**Trade Record:**
```python
{
    'type': 'long' or 'short',
    'entry_date': pd.Timestamp,
    'exit_date': pd.Timestamp,
    'entry_price': float,
    'exit_price': float,
    'bars_held': int,
    'pnl_pct': float,          # Profit/Loss as %
    'pnl_raw': float,          # Profit/Loss in price units
    'equity_after': float      # Equity multiplier after trade
}
```

### Configuration & Customization

#### Modify VWAP Windows
```python
windows = [120, 90, 60, 30]  # Your custom windows
df = process_market_data(df, windows=windows)
```

#### Adjust Entry/Exit Thresholds
Edit `run_mean_reversion_backtest()`:
```python
if z < -1.5:  # Changed from -2
    position = 'long'
elif z > 1.5:  # Changed from 2
    position = 'short'
```

### Performance Optimization

- **Data Caching**: Results are cached in memory; restart for fresh data
- **Batch Processing**: API requests are batched to respect rate limits
- **Vectorized Calculations**: Uses NumPy for fast numerical operations

### Limitations

⚠️ **Important Notes:**
1. **Historical data only** - Backtests use historical data; past performance ≠ future results
2. **Mean reversion assumption** - Strategy assumes prices revert to VWAP; this may not hold in trending markets
3. **No slippage/commissions** - Actual trading will have costs
4. **Binance SPOT only** - Does not support futures trading
5. **API rate limits** - Binance SPOT API has 1000-record limit per request

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `Symbol not found` | Verify symbol exists on Binance SPOT (e.g., BTCUSDT, not BTC) |
| `No data returned` | Check internet connection; Binance API may be rate-limited |
| `ImportError: plotly` | Run: `pip install plotly dash pandas numpy requests` |
| `Port 8050 already in use` | Change port: `app.run(port=8051)` in main section |
| `OHLC chart not showing` | Ensure you have OHLCV data (not just close prices) |

### Related Projects

- **[falx](https://github.com/lubluniky/falx)** - Full quantitative analysis framework

### License

MIT License - Feel free to use, modify, and distribute

### Contributing

Pull requests welcome! For major changes:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

### Support

- 📧 Issues: GitHub Issues
- 💬 Discussion: GitHub Discussions
- 📚 Documentation: See docstrings in source code

### Version History

**v1.0 (2024)** - Initial release
- Multi-window VWAP analysis
- Z-score calculations
- Mean-reversion backtesting
- Interactive Dash dashboard
- Comprehensive statistics

---

## Русский

### Обзор

**VWAP Z-Score Mod** — это комплексный инструмент анализа торговли и тестирования стратегий на Python, который анализирует возможности для торговли с возвратом к среднему, используя Volume-Weighted Average Price (VWAP) и статистические расчёты z-score. Инструмент интегрируется с данными спотового рынка Binance и предоставляет анализ в реальном времени и возможности исторического тестирования через интерактивную панель управления Plotly Dash.

### Ключевые возможности

✨ **Основной функционал:**
- 📊 **Многооконный анализ VWAP** - вычисление VWAP за периоды 365, 180, 90 и 30 дней одновременно
- 📈 **Расчёты Z-Score** - статистические показатели для выявления возможностей возврата к среднему
- 💰 **Данные спотового рынка Binance в реальном времени** - прямая загрузка рыночных данных (не фьючерсы)
- 🎯 **Тестирование стратегии возврата к среднему** - автоматизированное тестирование с подробными метриками
- 📉 **Интерактивные панели управления** - красивые диаграммы Plotly Dash с визуализацией на множество таймфреймов
- 💡 **Продвинутая статистика** - коэффициент Шарпа, коэффициент Сортино, максимальная просадка, процент побед и многое другое

### Техническая архитектура

#### Загрузка данных (`BinanceDataFetcher`)
- Надёжный wrapper для API спотового рынка Binance
- Обработка ограничений частоты API (1000 записей на запрос)
- Загрузка многолетних данных с автоматическим разбиением на части
- Поддержка множественных таймфреймов: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M

#### Расчёты
**Формула VWAP:**
$$\text{VWAP} = \frac{\sum(\text{типичная\_цена} \times \text{объём})}{\sum(\text{объём})}$$

где: $\text{типичная\_цена} = \frac{H + L + C}{3}$

**Формула Z-Score:**
$$z = \frac{P - \text{VWAP}}{\sigma}$$

где: $P$ = цена закрытия, $\sigma$ = скользящее стандартное отклонение

#### Стратегия возврата к среднему
```
Вход:    z-score < -2 (перепродажа, длинная позиция) ИЛИ z-score > 2 (перекупленность, короткая позиция)
Выход:   z-score пересекает 0 (возврат к среднему)
Позиция: Один вход на сигнал, без усреднения
```

### Установка

**Требования:**
- Python 3.8+
- pip

**Установите зависимости:**
```bash
pip install plotly dash pandas numpy requests
```

**Опционально (для экспорта статичных графиков):**
```bash
pip install kaleido
```

### Использование

#### Запуск панели управления

```bash
python vwap_mod.py
```

Затем откройте браузер и перейдите на:
```
http://127.0.0.1:8050/
```

#### Использование в качестве библиотеки

```python
from vwap_mod import BinanceDataFetcher, process_market_data, run_mean_reversion_backtest

# Загрузка данных
fetcher = BinanceDataFetcher()
df = fetcher.fetch_ohlcv(symbol='BTCUSDT', interval='1d', years=5)

# Обработка данных с VWAP и z-scores
df = process_market_data(df, windows=[365, 180, 90, 30])

# Запуск теста
results = run_mean_reversion_backtest(df, window=30)

# Доступ к результатам
print(f"Всего сделок: {len(results['trades'])}")
print(f"Кривая капитала: {results['equity_curve']}")
```

### Разделы панели управления

#### 1. **Панель управления**
- **Символ**: Введите любую торговую пару спотового рынка Binance (например, BTCUSDT, ETHUSDT)
- **Интервал**: Выберите таймфрейм (1d, 4h, 1h, 15m)
- **Годы данных**: Выберите исторический период (1, 2, 3 или 5 лет)
- **Кнопка загрузки**: Загрузить и проанализировать данные

#### 2. **Многооконная диаграмма VWAP**
- **Ряд 1**: Диаграмма OHLC с ценовым движением
- **Ряд 2**: Столбцы объёма (зелёный = вверх, красный = вниз)
- **Ряды 3-6**: Панели Z-Score для каждого окна VWAP
  - **Зелёная зона** (z < -2): Перепродажа, сигнал на длинную позицию
  - **Красная зона** (z > 2): Перекупленность, сигнал на короткую позицию
  - **Чёрная линия** (z = 0): Точка возврата к среднему
  - **Светлосиняя/оранжевая**: Нейтральные зоны

#### 3. **Таблица статистики тестирования**
Метрики производительности для каждого окна VWAP:
- **Сделки**: Общее количество завершённых сделок
- **Процент побед %**: Процент прибыльных сделок
- **Средняя прибыль %**: Средняя прибыль/убыток на сделку (%)
- **Коэффициент Шарпа**: Метрика риск-скорректированной доходности
- **Коэффициент Сортино**: Риск-скорректированная доходность по нижней стороне
- **Макс просадка %**: Наибольший спад от пика до дна
- **Коэффициент прибыли**: Сумма побед / Сумма убытков
- **Воздействие %**: Процент времени на рынке

#### 4. **Диаграмма кривых капитала**
Визуальное сравнение совокупной доходности всех окон VWAP во времени

### Объяснение метрик производительности

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **Процент побед** | (Прибыльные сделки / Всего сделок) × 100 | % прибыльных сделок |
| **Коэффициент Шарпа** | Средняя доходность / Ст.отклонение × √252 | Риск-скорректированная доходность (>1.0 хорошо) |
| **Коэффициент Сортино** | Средняя доходность / Ст.отклонение вниз × √252 | Риск-скорректированная только вниз |
| **Макс просадка** | (Пик - Дно) / Пик × 100 | Наибольший накопленный убыток % |
| **Коэффициент прибыли** | Сумма прибыльных / Сумма убыточных | Соотношение риска/вознаграждения (>1.5 хорошо) |
| **Ожидаемое значение** | (% побед × Средн. прибыль) - (% убытков × Средн. убыток) | Средняя прибыль на сделку |

### Пример результатов

**BTC/USDT - 5 лет дневных данных (2020-2025):**

| Окно | Сделки | % побед | Средн. прибыль % | Шарпа | Макс просадка % |
|------|--------|---------|------------------|-------|-----------------|
| 365d | 24     | 58.3%   | +2.15%           | 1.32  | -18.5%         |
| 180d | 32     | 62.5%   | +1.87%           | 1.45  | -16.2%         |
| 90d  | 48     | 64.1%   | +1.42%           | 1.58  | -14.8%         |
| 30d  | 72     | 61.1%   | +0.95%           | 1.22  | -19.3%         |

### Продвинутые возможности

### Справочник API

#### `BinanceDataFetcher.fetch_ohlcv()`
```python
df = fetcher.fetch_ohlcv(
    symbol='BTCUSDT',      # Торговая пара
    interval='1d',         # Таймфрейм
    years=5                # Период исторических данных
)
# Возвращает: DataFrame с колонками [timestamp, open, high, low, close, volume]
```

#### `compute_vwap()`
```python
vwap_series = compute_vwap(df, window=365)
# Возвращает: pd.Series с значениями VWAP
```

#### `compute_z_score()`
```python
z_score = compute_z_score(close, vwap, rolling_std)
# Возвращает: pd.Series с значениями z-score
```

#### `process_market_data()`
```python
df_processed = process_market_data(df, windows=[365, 180, 90, 30])
# Добавляет колонки: vwap_X, rolling_std_X, z_score_X для каждого окна X
```

#### `run_mean_reversion_backtest()`
```python
results = run_mean_reversion_backtest(df, window=30)
# Возвращает: словарь с 'trades' (список) и 'equity_curve' (список)
```

### Схема данных

**Входной DataFrame (OHLCV):**
```
timestamp     datetime64  - Время открытия свечи
open          float64     - Цена открытия
high          float64     - Максимальная цена
low           float64     - Минимальная цена
close         float64     - Цена закрытия
volume        float64     - Объём торговли
```

**Обработанный DataFrame (с добавленными колонками):**
```
vwap_365      float64     - VWAP за 365 баров
rolling_std_365  float64  - Стандартное отклонение за 365 баров
z_score_365   float64     - Z-score за 365 баров
... (то же для окон: 180, 90, 30)
```

**Запись о сделке:**
```python
{
    'type': 'long' или 'short',
    'entry_date': pd.Timestamp,
    'exit_date': pd.Timestamp,
    'entry_price': float,
    'exit_price': float,
    'bars_held': int,
    'pnl_pct': float,          # Прибыль/убыток в %
    'pnl_raw': float,          # Прибыль/убыток в единицах цены
    'equity_after': float      # Множитель капитала после сделки
}
```

### Конфигурация и настройка

#### Изменить окна VWAP
```python
windows = [120, 90, 60, 30]  # Ваши пользовательские окна
df = process_market_data(df, windows=windows)
```

#### Изменить пороги входа/выхода
Отредактируйте `run_mean_reversion_backtest()`:
```python
if z < -1.5:  # Было -2
    position = 'long'
elif z > 1.5:  # Было 2
    position = 'short'
```

### Оптимизация производительности

- **Кеширование данных**: Результаты кешируются в памяти; перезагрузите для свежих данных
- **Пакетная обработка**: Запросы к API разбиваются на пакеты с соблюдением ограничений
- **Векторизованные вычисления**: Использует NumPy для быстрых численных операций

### Ограничения

⚠️ **Важные замечания:**
1. **Только исторические данные** - тесты используют исторические данные; прошлые результаты ≠ будущие результаты
2. **Предположение о возврате к среднему** - стратегия предполагает возврат цены к VWAP; это может не работать при тренде
3. **Без проскальзывания и комиссий** - реальная торговля будет иметь расходы
4. **Только Binance SPOT** - не поддерживает торговлю фьючерсами
5. **Ограничения API** - API спотового рынка Binance имеет лимит 1000 записей за запрос

### Решение проблем

| Проблема | Решение |
|----------|---------|
| `Символ не найден` | Проверьте, существует ли символ на Binance SPOT (например, BTCUSDT, не BTC) |
| `Данные не возвращены` | Проверьте интернет-соединение; API Binance может быть перегружен |
| `ImportError: plotly` | Выполните: `pip install plotly dash pandas numpy requests` |
| `Порт 8050 уже используется` | Измените порт: `app.run(port=8051)` в основном разделе |
| `Диаграмма OHLC не отображается` | Убедитесь, что у вас есть данные OHLCV (не только цены закрытия) |

### Связанные проекты

- **[falx](https://github.com/lubluniky/falx)** - Полный фреймворк для количественного анализа

### Лицензия

MIT License - Свободно используйте, модифицируйте и распространяйте

### Вклад

Pull requests приветствуются! Для больших изменений:
1. Форкните репозиторий
2. Создайте ветку функции
3. Коммитьте изменения
4. Запушьте в ветку
5. Откройте Pull Request

### Поддержка

- 📧 Issues: GitHub Issues
- 💬 Обсуждение: GitHub Discussions
- 📚 Документация: Смотрите docstrings в исходном коде

### История версий

**v1.0 (2024)** - Первый релиз
- Многооконный анализ VWAP
- Расчёты z-score
- Тестирование стратегии возврата к среднему
- Интерактивная панель управления Dash
- Подробная статистика

