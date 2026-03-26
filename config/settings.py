"""
EdgeFinder Configuration — All Tunable Parameters
===================================================
Every threshold, weight, and limit lives here. NEVER hardcode values in modules.
The Strategy Optimizer (Module 5) will eventually modify these programmatically.
"""

# ============================================================
# ACCOUNT SETTINGS
# ============================================================
STARTING_CAPITAL = 2500.00
MAX_RISK_PER_TRADE_PCT = 0.02          # 2% of account per trade
MAX_OPEN_POSITIONS = 5                  # Max simultaneous positions
MAX_PORTFOLIO_CONCENTRATION_PCT = 0.20  # 20% max in one position
PDT_DAY_TRADE_LIMIT = 3                # Per 5 rolling business days
PDT_WINDOW_DAYS = 5                     # Rolling window for PDT

# ============================================================
# RISK MANAGEMENT
# ============================================================
DRAWDOWN_CIRCUIT_BREAKER_PCT = 0.20    # Halt at 20% drawdown from peak
MIN_REWARD_TO_RISK_RATIO = 1.5         # Minimum R:R for any trade
REVENGE_TRADE_COOLDOWN_MINUTES = 30    # Wait after a stop-out
MAX_SAME_SECTOR_POSITIONS = 3          # Correlation limit
TRAILING_STOP_ACTIVATION_R = 1.0       # Move stop to breakeven at 1R
TRAILING_STOP_TRAIL_R = 2.0            # Trail by 1R once at 2R

# ============================================================
# FUNDAMENTAL SCANNER — LYNCH CRITERIA
# ============================================================
LYNCH_WEIGHT = 0.50  # Weight in composite score

LYNCH_PEG_IDEAL = 1.0          # PEG below this = full score
LYNCH_PEG_MAX = 1.5            # PEG above this = disqualified
LYNCH_PEG_WEIGHT = 0.25        # 25% of Lynch sub-score

LYNCH_EARNINGS_GROWTH_MIN = 0.15    # 15% annualized earnings growth
LYNCH_EARNINGS_GROWTH_WEIGHT = 0.20

LYNCH_DEBT_TO_EQUITY_PREFERRED = 0.5
LYNCH_DEBT_TO_EQUITY_MAX = 1.0       # Hard ceiling
LYNCH_DEBT_TO_EQUITY_WEIGHT = 0.15

LYNCH_REVENUE_GROWTH_MIN = 0.10      # 10% YoY
LYNCH_REVENUE_GROWTH_WEIGHT = 0.15

LYNCH_INSTITUTIONAL_MIN = 0.20       # 20% floor
LYNCH_INSTITUTIONAL_MAX = 0.70       # 70% ceiling (overcrowded above)
LYNCH_INSTITUTIONAL_WEIGHT = 0.10

LYNCH_CATEGORY_WEIGHT = 0.15         # Stock classification score

# ============================================================
# FUNDAMENTAL SCANNER — BURRY CRITERIA
# ============================================================
BURRY_WEIGHT = 0.50  # Weight in composite score

BURRY_FCF_YIELD_STRONG = 0.08       # 8% = strong
BURRY_FCF_YIELD_ACCEPTABLE = 0.05   # 5% = acceptable
BURRY_FCF_YIELD_WEIGHT = 0.30

BURRY_PRICE_TO_TANGIBLE_BOOK_DEEP = 1.0    # Below 1.0 = deep value
BURRY_PRICE_TO_TANGIBLE_BOOK_VALUE = 2.0   # Below 2.0 = value
BURRY_PRICE_TO_TANGIBLE_BOOK_WEIGHT = 0.25

BURRY_SHORT_INTEREST_CONTRARIAN = 0.15     # Above 15% = contrarian signal
BURRY_SHORT_INTEREST_WEIGHT = 0.15

BURRY_EV_TO_EBITDA_MAX = 8.0               # Prefer below 8x
BURRY_EV_TO_EBITDA_WEIGHT = 0.15

BURRY_CURRENT_RATIO_MIN = 1.5              # Can survive stress
BURRY_CURRENT_RATIO_WEIGHT = 0.15

# ============================================================
# COMPOSITE SCORING
# ============================================================
WATCHLIST_MIN_COMPOSITE_SCORE = 60   # Fallback only — used when no strategies are loaded
# Note: The primary watchlist gate is now strategy-driven.
# Each strategy's qualifies_stock() determines whether a stock stays active.
# composite_score is still computed for display/sorting purposes.

# ============================================================
# SCANNER FILTERS (Pre-screening before scoring)
# ============================================================
SCANNER_MIN_MARKET_CAP = 300_000_000       # $300M minimum (no penny stocks)
SCANNER_MAX_MARKET_CAP = 200_000_000_000   # $200B max (no mega-caps)
SCANNER_MIN_AVG_VOLUME = 500_000           # 500K avg daily volume
SCANNER_MIN_PRICE = 5.00                    # No stocks under $5
SCANNER_MAX_PRICE = 500.00                  # No ultra-high-price stocks
SCANNER_EXCLUDED_SECTORS = [               # Sectors to skip
    "Utilities",                           # Too slow for this system
]

# ============================================================
# TECHNICAL SIGNAL ENGINE
# ============================================================
# EMA Crossovers
SIGNAL_EMA_FAST_DAY = 9
SIGNAL_EMA_SLOW_DAY = 21
SIGNAL_EMA_FAST_SWING = 50
SIGNAL_EMA_SLOW_SWING = 200

# RSI
SIGNAL_RSI_PERIOD = 14
SIGNAL_RSI_OVERSOLD = 30
SIGNAL_RSI_OVERBOUGHT = 70

# MACD
SIGNAL_MACD_FAST = 12
SIGNAL_MACD_SLOW = 26
SIGNAL_MACD_SIGNAL = 9

# Volume
SIGNAL_VOLUME_SPIKE_MULTIPLIER = 1.5  # 1.5x 20-day average
SIGNAL_VOLUME_AVG_PERIOD = 20

# Bollinger Bands
SIGNAL_BB_PERIOD = 20
SIGNAL_BB_STD = 2.0
SIGNAL_BB_SQUEEZE_THRESHOLD = 0.04     # BBWidth below this = squeeze

# ATR (Average True Range)
SIGNAL_ATR_PERIOD = 14

# Stochastic Oscillator
SIGNAL_STOCH_K = 14
SIGNAL_STOCH_D = 3
SIGNAL_STOCH_OVERSOLD = 20
SIGNAL_STOCH_OVERBOUGHT = 80

# ADX (Average Directional Index)
SIGNAL_ADX_PERIOD = 14
SIGNAL_ADX_STRONG_TREND = 25           # ADX above this = trending

# OBV (On-Balance Volume)
SIGNAL_OBV_DIVERGENCE_BARS = 10        # Lookback for price vs OBV divergence

# 52-Week Context
SIGNAL_52W_HIGH_PROXIMITY = 0.05       # Within 5% of 52w high
SIGNAL_52W_LOW_PROXIMITY = 0.10        # Within 10% of 52w low

# Confidence Thresholds
SIGNAL_CONFIDENCE_LOW = 40             # 1 indicator — no trade
SIGNAL_CONFIDENCE_MODERATE = 60        # 2 indicators — half position
SIGNAL_CONFIDENCE_HIGH = 80            # 3+ indicators — full position
SIGNAL_MIN_CONFIDENCE_TO_TRADE = 60    # Minimum to enter

# ============================================================
# NEWS SENTIMENT GATE
# ============================================================
SENTIMENT_LOOKBACK_HOURS = 48          # Check news from last 48 hours
SENTIMENT_STRONG_NEGATIVE = -0.5       # VADER compound score
SENTIMENT_MILD_NEGATIVE = -0.2
SENTIMENT_MILD_POSITIVE = 0.2
SENTIMENT_STRONG_POSITIVE = 0.5

SENTIMENT_STRONG_NEG_ACTION = "BLOCK"           # Block trade entirely
SENTIMENT_MILD_NEG_ACTION = "REDUCE_50"          # Cut position 50%
SENTIMENT_NEUTRAL_ACTION = "PROCEED"             # No change
SENTIMENT_MILD_POS_ACTION = "CONFIDENCE_PLUS_10" # +10% confidence
SENTIMENT_STRONG_POS_ACTION = "CONFIDENCE_PLUS_20"  # +20% confidence

SENTIMENT_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

# ============================================================
# SCHEDULING
# ============================================================
SCANNER_RUN_TIME = "18:15"             # 6:15 PM ET (after extended trading)
SIGNAL_CHECK_INTERVAL_MINUTES = 15     # During market hours
MARKET_OPEN_ET = "07:00"               # Pre-market coverage
MARKET_CLOSE_ET = "18:00"              # Extended hours coverage
POSITION_MONITOR_INTERVAL_MINUTES = 5  # Price check frequency

# ============================================================
# TIMEOUT & PARALLELISM
# ============================================================
YFINANCE_CALL_TIMEOUT = 10            # Seconds per individual yfinance call
YFINANCE_BARS_TIMEOUT = 15            # Seconds for bars fetch (larger payload)
PRICE_FETCH_WORKERS = 8               # Parallel threads for price fetching
PRICE_FETCH_TOTAL_TIMEOUT = 30        # Total seconds for parallel price batch
BARS_FETCH_TOTAL_TIMEOUT = 60         # Total seconds for parallel bars batch
SCHEDULER_EXECUTOR_THREADS = 3        # APScheduler thread pool size

# ============================================================
# DEFAULT TICKER UNIVERSE (used when Wikipedia S&P lists are unavailable)
# ============================================================
SCANNER_DEFAULT_TICKERS = [
    # Tech / Software
    "PYPL", "SNAP", "PLTR", "ROKU", "SQ", "PATH", "BILL",
    "CFLT", "MDB", "DDOG", "NET", "ZS", "CRWD", "DUOL",
    "DOCS", "HIMS", "APP", "TOST",
    # Semiconductors
    "ON", "SWKS", "QRVO", "MRVL", "SMCI", "ENPH", "SEDG", "FSLR",
    # Consumer
    "ETSY", "W", "CHWY", "CELH", "ELF", "RVLV", "DECK",
    "CAVA", "BROS", "SHAK", "LULU",
    # Energy
    "DVN", "MRO", "OVV", "CTRA", "FANG",
    # Materials / Industrials
    "CLF", "AA", "X", "NUE", "STLD", "AXON",
    # Transport
    "DAL", "UAL", "LUV", "JBLU", "UBER", "LYFT",
    # Gaming / Entertainment
    "MGM", "WYNN", "CZR", "DKNG",
    # Fintech
    "SOFI", "COIN", "HOOD",
    # Travel / Delivery
    "ABNB", "DASH",
    # Space / Quantum / EV
    "RKLB", "IONQ", "RIVN",
    # Pharma
    "PFE",
]

# ============================================================
# SECTOR ROTATION — Nightly scan rotates through sectors
# ============================================================
# Each day scans a subset of sectors. Over a week, full universe is covered.
# Watchlist entries persist until a re-scan shows the stock no longer qualifies
# for any registered strategy.
SCAN_SECTOR_ROTATION = [
    ["Technology", "Communication Services"],             # Monday
    ["Healthcare"],                                       # Tuesday
    ["Financial Services", "Real Estate"],                # Wednesday
    ["Consumer Cyclical", "Consumer Defensive"],           # Thursday
    ["Industrials", "Energy", "Basic Materials"],          # Friday
]

# Volume-aware slippage model (for honest execution)
SLIPPAGE_BASE_RATE = 0.0005          # 0.05% base slippage
SLIPPAGE_VOLUME_FACTOR = 1.5         # Multiplier for low-volume stocks
SLIPPAGE_MIN_AVG_VOLUME = 100_000    # Below this = high slippage penalty

# ============================================================
# DATABASE — Trade Data (SACRED — never auto-delete)
# ============================================================
DATABASE_PATH = "data/edgefinder.db"

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"
LOG_FILE = "data/edgefinder.log"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


# ============================================================
# ARENA — Multi-Strategy Configuration
# ============================================================
ARENA_STARTING_CAPITAL_PER_STRATEGY = 2500.00   # Each strategy gets this much
ARENA_MAX_TOTAL_POSITION_VALUE = 2500.00         # Max aggregate cost of open positions per strategy
ARENA_MAX_POSITIONS_PER_STRATEGY = 5             # Max open positions per strategy
ARENA_DRAWDOWN_PAUSE_PCT = -15.0                 # Auto-pause at -15% drawdown
