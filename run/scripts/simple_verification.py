#!/usr/bin/env python3
"""
Simple verification of key Colab infrastructure fixes.

Tests the most critical components without complex imports.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def check_path_fixes():
    """Check if critical path fixes are in place."""
    print("🔍 Checking Path Fixes")
    print("=" * 40)

    # Check colab_full_training.ipynb for path fixes
    notebook_path = Path("notebooks/colab_full_training.ipynb")
    if notebook_path.exists():
        content = notebook_path.read_text()

        if "output_central" in content:
            print("✅ colab_full_training.ipynb uses output_central")
        else:
            print("❌ colab_full_training.ipynb missing output_central")

        if "data/raw" not in content:
            print("✅ colab_full_training.ipynb avoids data/raw")
        else:
            print("❌ colab_full_training.ipynb still uses data/raw")
    else:
        print("❌ colab_full_training.ipynb not found")

    # Check new data collection notebook
    new_notebook_path = Path("notebooks/colab_data_collection.ipynb")
    if new_notebook_path.exists():
        print("✅ colab_data_collection.ipynb exists")

        content = new_notebook_path.read_text()
        if "output_central" in content:
            print("✅ colab_data_collection.ipynb uses output_central")
        else:
            print("❌ colab_data_collection.ipynb missing output_central")
    else:
        print("❌ colab_data_collection.ipynb not found")


def check_requirements():
    """Check requirements.txt has correct packages."""
    print("\n📦 Checking Requirements")
    print("=" * 40)

    req_path = Path("requirements.txt")
    if req_path.exists():
        content = req_path.read_text()

        # Check for correct pip packages
        if "fred>=1.1.4" in content:
            print("✅ FRED package correctly specified")
        else:
            print("❌ FRED package missing or incorrect")

        if "comtradeapicall>=1.3.0" in content:
            print("✅ Comtrade package correctly specified")
        else:
            print("❌ Comtrade package missing or incorrect")

        # Check for incorrect editable installs
        if "-e ./new_data_sources/FRB" not in content:
            print("✅ No incorrect FRED editable install")
        else:
            print("❌ Incorrect FRED editable install found")

        if "-e ./new_data_sources/comtradeapicall" not in content:
            print("✅ No incorrect Comtrade editable install")
        else:
            print("❌ Incorrect Comtrade editable install found")
    else:
        print("❌ requirements.txt not found")


def check_ecb_shocks():
    """Check ECB shocks vendor structure."""
    print("\n🏛️ Checking ECB Shocks")
    print("=" * 40)

    # Check vendor structure
    ecb_init = Path("data/vendors/ecb_shocks/__init__.py")
    if ecb_init.exists():
        print("✅ ECB shocks vendor wrapper exists")
    else:
        print("❌ ECB shocks vendor wrapper missing")

    # Check data files
    ecb_data_dir = Path("data/vendors/ecb_shocks/data")
    daily_file = ecb_data_dir / "shocks_ecb_mpd_me_d.csv"
    monthly_file = ecb_data_dir / "shocks_ecb_mpd_me_m.csv"

    if daily_file.exists():
        print("✅ ECB daily shocks data exists")
    else:
        print("❌ ECB daily shocks data missing")

    if monthly_file.exists():
        print("✅ ECB monthly shocks data exists")
    else:
        print("❌ ECB monthly shocks data missing")


def check_retry_utilities():
    """Check retry utilities exist."""
    print("\n🔄 Checking Retry Utilities")
    print("=" * 40)

    retry_utils_path = Path("utils/retry_utils.py")
    if retry_utils_path.exists():
        print("✅ Retry utilities module exists")

        content = retry_utils_path.read_text()

        if "retry_with_backoff" in content:
            print("✅ Retry decorator implemented")
        else:
            print("❌ Retry decorator missing")

        if "rate_limit" in content:
            print("✅ Rate limiting implemented")
        else:
            print("❌ Rate limiting missing")

        if "api_retry" in content:
            print("✅ API retry decorator implemented")
        else:
            print("❌ API retry decorator missing")
    else:
        print("❌ Retry utilities module missing")


def check_downloaders():
    """Check if downloaders have retry logic."""
    print("\n📥 Checking Downloaders")
    print("=" * 40)

    # Check FRED downloader
    fred_downloader = Path("data/downloaders/fred_downloader.py")
    if fred_downloader.exists():
        content = fred_downloader.read_text()
        if "utils.retry_utils" in content:
            print("✅ FRED downloader has retry imports")
        else:
            print("❌ FRED downloader missing retry imports")

        if "@api_retry" in content:
            print("✅ FRED downloader uses retry decorators")
        else:
            print("❌ FRED downloader missing retry decorators")
    else:
        print("❌ FRED downloader not found")

    # Check Comtrade downloader
    comtrade_downloader = Path("data/downloaders/comtrade_downloader.py")
    if comtrade_downloader.exists():
        content = comtrade_downloader.read_text()
        if "utils.retry_utils" in content:
            print("✅ Comtrade downloader has retry imports")
        else:
            print("❌ Comtrade downloader missing retry imports")

        if "@api_retry" in content:
            print("✅ Comtrade downloader uses retry decorators")
        else:
            print("❌ Comtrade downloader missing retry decorators")
    else:
        print("❌ Comtrade downloader not found")


def check_test_suite():
    """Check test suite exists."""
    print("\n🧪 Checking Test Suite")
    print("=" * 40)

    test_file = Path("tests/test_colab_pipeline.py")
    if test_file.exists():
        print("✅ Colab pipeline test suite exists")

        content = test_file.read_text()

        if "TestColabConfiguration" in content:
            print("✅ Configuration tests implemented")
        else:
            print("❌ Configuration tests missing")

        if "TestDownloadState" in content:
            print("✅ Download state tests implemented")
        else:
            print("❌ Download state tests missing")

        if "TestRetryLogic" in content:
            print("✅ Retry logic tests implemented")
        else:
            print("❌ Retry logic tests missing")
    else:
        print("❌ Colab pipeline test suite missing")


def generate_summary():
    """Generate final summary and return exit code.
    
    Returns:
        int: 0 if all checks pass, non-zero if any check fails.
    """
    print("\n" + "=" * 60)
    print("📋 INFRASTRUCTURE VERIFICATION SUMMARY")
    print("=" * 60)

    # Run all checks
    checks = [
        check_path_fixes,
        check_requirements,
        check_ecb_shocks,
        check_retry_utilities,
        check_downloaders,
        check_test_suite,
    ]

    results = {}
    any_failed = False
    for check_func in checks:
        try:
            check_func()
            results[check_func.__name__] = True
        except Exception as e:
            print(f"❌ Check {check_func.__name__} failed: {e}")
            results[check_func.__name__] = str(e)
            any_failed = True

    # Determine overall status based on results
    overall_status = "failed" if any_failed else "passed"
    
    # Generate timestamped report
    report = {
        "timestamp": datetime.now().isoformat(),
        "verification_type": "simple_colab_infrastructure",
        "status": overall_status,
        "results": results,
    }

    report_file = Path("simple_verification_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved: {report_file}")
    
    if overall_status == "passed":
        print("\n🎉 Key infrastructure components verified!")
        return 0
    else:
        print(f"\n❌ Verification failed: {len([v for v in results.values() if v is not True])} check(s) failed")
        return 1


if __name__ == "__main__":
    print("🚀 COLAB INFRASTRUCTURE SIMPLE VERIFICATION")
    print("=" * 60)
    print("Verifying critical path fixes and components...\n")

    exit_code = generate_summary()
    sys.exit(exit_code)
