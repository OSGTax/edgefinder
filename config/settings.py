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
DAILY_LOSS_LIMIT_PCT = 0.05            # Stop trading if down 5% in a day
WEEKLY_LOSS_LIMIT_PCT = 0.10           # Pause if down 10% in a week
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
WATCHLIST_MIN_COMPOSITE_SCORE = 60   # Out of 100, minimum to make watchlist
WATCHLIST_MAX_SIZE = 100             # Cap watchlist at this many stocks

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
SCANNER_EXCHANGES = ["NYSE", "NASDAQ"]     # US exchanges only

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
SCANNER_RUN_TIME = "16:30"             # 4:30 PM ET (after market close)
SIGNAL_CHECK_INTERVAL_MINUTES = 15     # During market hours
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
DAY_TRADE_CUTOFF_ET = "15:50"          # Close day trades by 3:50 PM
EXTENDED_HOURS_START_ET = "07:00"      # Pre-market monitoring start
EXTENDED_HOURS_END_ET = "18:00"        # Post-market monitoring end
POSITION_MONITOR_INTERVAL_MINUTES = 5  # Price check frequency
ACCOUNT_SNAPSHOT_TIME = "16:05"        # Daily snapshot after close

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
# Watchlist entries persist across days (7-day TTL) so good plays aren't lost.
SCAN_SECTOR_ROTATION = [
    ["Technology", "Communication Services"],             # Monday
    ["Healthcare"],                                       # Tuesday
    ["Financial Services", "Real Estate"],                # Wednesday
    ["Consumer Cyclical", "Consumer Defensive"],           # Thursday
    ["Industrials", "Energy", "Basic Materials"],          # Friday
]
SCAN_WATCHLIST_TTL_DAYS = 7  # Deactivate watchlist entries older than this

# ============================================================
# DATA SERVICE — API Sources
# ============================================================
# Alpaca: real-time and historical bars (free paper trading account)
# Keys loaded from config/secrets.env (ALPACA_API_KEY, ALPACA_SECRET_KEY)
DATA_SERVICE_ALPACA_ENABLED = True

# FMP (Financial Modeling Prep): fundamentals, ratios, earnings calendar
# Keys loaded from config/secrets.env (FMP_API_KEY)
# Free tier: 250 requests/day — cache is critical
DATA_SERVICE_FMP_ENABLED = True

# yfinance: always-available fallback (no key needed, but rate limited)
DATA_SERVICE_YFINANCE_FALLBACK = True

# Cache settings
DATA_CACHE_PATH = "data/cache.db"    # Separate from trade data — safe to delete
DATA_CACHE_CLEANUP_HOUR = 4          # Run cache cleanup at 4 AM ET

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
# DASHBOARD
# ============================================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8000

# ============================================================
# ARENA — Multi-Strategy Configuration
# ============================================================
ARENA_STARTING_CAPITAL_PER_STRATEGY = 2500.00   # Each strategy gets this much
ARENA_MAX_POSITIONS_PER_STRATEGY = 5             # Max open positions per strategy
ARENA_DRAWDOWN_PAUSE_PCT = -15.0                 # Auto-pause at -15% drawdown
ARENA_SIGNAL_CHECK_INTERVAL_MINUTES = 15         # How often to check signals
ARENA_POSITION_MONITOR_INTERVAL_MINUTES = 5      # How often to check stops/targets
ARENA_SNAPSHOT_TIME = "16:05"                     # Daily snapshot after close
