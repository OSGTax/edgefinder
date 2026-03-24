#!/bin/bash
# ============================================================
# EdgeFinder — Codespace Setup (runs automatically on creation)
# ============================================================
set -e

echo "========================================="
echo "  EdgeFinder — Setting up Codespace..."
echo "========================================="

# Install Python dependencies
echo "→ Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Initialize the database
echo "→ Setting up database..."
python scripts/setup_db.py --yes 2>/dev/null || python scripts/setup_db.py

# Generate secrets.env from Codespace Secrets (environment variables)
# GitHub Codespaces injects secrets as env vars automatically
if [ ! -f config/secrets.env ]; then
    echo "→ Generating config/secrets.env from Codespace Secrets..."
    cat > config/secrets.env << ENVFILE
ALPACA_API_KEY=${ALPACA_API_KEY:-your_alpaca_api_key_here}
ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY:-your_alpaca_secret_key_here}
FMP_API_KEY=${FMP_API_KEY:-your_fmp_api_key_here}
ENVFILE
    echo "  ✓ secrets.env created from environment variables"
else
    echo "  ✓ secrets.env already exists"
fi

# Verify installation
echo ""
echo "→ Running verification..."
python scripts/verify_install.py 2>/dev/null || echo "  (verify_install had warnings — this is OK)"

echo ""
echo "========================================="
echo "  EdgeFinder Codespace is ready!"
echo ""
echo "  Quick commands:"
echo "    python scripts/verify_data_service.py  — test API connections"
echo "    python -m pytest tests/ -v -m 'not integration'  — run tests"
echo "    python scripts/run_scanner.py  — run stock scanner"
echo "========================================="
