from setuptools import setup, find_packages

setup(
    name="edgefinder",
    version="1.0.0",
    description="Intelligent paper trading system with Lynch/Burry fundamentals",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.31",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pandas-ta>=0.3.14b",
        "vaderSentiment>=3.3.2",
        "feedparser>=6.0.10",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "apscheduler>=3.10.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
    ],
)
