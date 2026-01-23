#!/bin/bash

# Installation script for new fundamental data sources
# Installs UN Comtrade API and FRED packages from local directories

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

echo "========================================"
echo "Installing Fundamental Data Sources"
echo "========================================"
echo ""

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

# Pre-install checks
echo "Performing pre-install checks..."

# Detect which Python binary is available (prefer python3, fallback to python)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Neither python3 nor python found on PATH"
    echo "   Please install Python 3.8 or later"
    exit 1
fi
echo "   ✅ Python found: $PYTHON_CMD"

if ! $PYTHON_CMD -m pip &> /dev/null; then
    echo "❌ Error: pip not found. Please install Python and pip first."
    exit 1
fi
echo "   ✅ pip found: $($PYTHON_CMD -m pip --version)"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "   ⚠️  Warning: Not running in a virtual environment"
    echo "      Consider activating a virtual environment first"
fi
echo ""

# Function to install a package
install_package() {
    local package_name=$1
    local package_dir=$2

    echo "Installing $package_name..."
    if [ -d "$package_dir" ]; then
        if $PYTHON_CMD -m pip install -e "$package_dir"; then
            echo "   ✅ $package_name installed"
        else
            echo "   ❌ Error installing $package_name from $package_dir"
            echo "      Please check the package contents and try again"
            exit 1
        fi
    else
        echo "   ⚠️  Warning: $package_dir not found, skipping"
    fi
    echo ""
}

# Check if the data collection directory exists
DATA_DIR="$PROJECT_ROOT/new_data_sources"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Directory '$DATA_DIR' not found"
    echo "   Make sure the 'new_data_sources' directory exists"
    exit 1
fi

# Install UN Comtrade API
echo "1. Installing UN Comtrade API..."
install_package "UN Comtrade API" "$DATA_DIR/comtradeapicall"

# Install FRED (Federal Reserve Economic Data)
echo "2. Installing FRED API..."
install_package "FRED API" "$DATA_DIR/FRB"

# ECB Shocks - no installation needed (CSV data only)
echo "3. Checking ECB Monetary Policy Shocks data..."
ECB_DIR="$DATA_DIR/jkshocks_update_ecb"
if [ -d "$ECB_DIR" ]; then
    if [ -f "$ECB_DIR/shocks_ecb_mpd_me_d.csv" ] && [ -f "$ECB_DIR/shocks_ecb_mpd_me_m.csv" ]; then
        echo "   ✅ ECB shocks data found (CSV files, no installation needed)"
    else
        echo "   ⚠️  Warning: ECB shock CSV files not found in $ECB_DIR"
    fi
else
    echo "   ⚠️  Warning: $ECB_DIR not found"
fi
echo ""

echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set up API keys:"
echo "   export FRED_API_KEY='your_fred_key_here'"
echo "   export COMTRADE_API_KEY='your_comtrade_key_here'  # optional"
echo ""
echo "2. Get API keys:"
echo "   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html (free)"
echo "   - Comtrade: https://comtradeplus.un.org/ (free tier available)"
echo ""
echo "3. Test the installation:"
echo "   python run/scripts/test_data_sources.py"
echo ""
echo "4. See documentation:"
echo "   - Quick start: docs/QUICK_START_DATA_SOURCES.md"
echo "   - Full docs: docs/NEW_DATA_SOURCES.md"
echo ""
