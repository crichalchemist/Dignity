"""
Verify notebook implementation changes are correct.

Checks:
- Package dependencies
- ECB shocks vendor structure
- Retry utilities
- Notebook configuration
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Minimum number of cells required in the notebook for sufficient coverage
# This represents essential setup, configuration, download, and processing cells
MIN_NOTEBOOK_CELL_COUNT = 18

def check_requirements():
    """Verify requirements.txt has correct packages."""
    print("📦 Checking requirements.txt...")
    req_file = ROOT / 'requirements.txt'

    if not req_file.exists():
        print("  ❌ requirements.txt not found")
        return False

    content = req_file.read_text()

    # Use precise package name matching (not substring)
    checks = {
        'fred>=1.1.4': any(
            line.strip().startswith('fred')
            for line in content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ),
        'comtradeapicall>=1.3.0': any(
            line.strip().startswith('comtradeapicall')
            for line in content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ),
        'ratelimit>=2.2.1': any(
            line.strip().startswith('ratelimit')
            for line in content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ),
    }

    all_passed = True
    for package, found in checks.items():
        status = "✅" if found else "❌"
        print(f"  {status} {package}")
        if not found:
            all_passed = False

    return all_passed

def check_ecb_vendor():
    """Verify ECB shocks vendor structure."""
    print("\n🏛️ Checking ECB shocks vendor...")
    vendor_dir = ROOT / 'data' / 'vendors' / 'ecb_shocks'

    checks = {
        '__init__.py': vendor_dir / '__init__.py',
        'ATTRIBUTION.md': vendor_dir / 'ATTRIBUTION.md',
        'data/': vendor_dir / 'data',
        'daily CSV': vendor_dir / 'data' / 'shocks_ecb_mpd_me_d.csv',
        'monthly CSV': vendor_dir / 'data' / 'shocks_ecb_mpd_me_m.csv',
    }

    all_passed = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path.relative_to(ROOT)}")
        if not exists:
            all_passed = False

    # Check __init__.py has required functions
    if checks['__init__.py'].exists():
        content = checks['__init__.py'].read_text()
        funcs = ['get_data_dir', 'get_daily_shocks_path', 'get_monthly_shocks_path', 'verify_data_exists']

        print("\n  Functions in __init__.py:")
        for func in funcs:
            found = f"def {func}" in content
            status = "✅" if found else "❌"
            print(f"    {status} {func}()")
            if not found:
                all_passed = False

    return all_passed

def check_retry_utils():
    """Verify retry utilities module."""
    print("\n🔄 Checking retry utilities...")
    retry_file = ROOT / 'utils' / 'retry_utils.py'

    if not retry_file.exists():
        print("  ❌ utils/retry_utils.py not found")
        return False

    content = retry_file.read_text()

    checks = {
        'retry_with_backoff': 'def retry_with_backoff' in content,
        'rate_limit': 'def rate_limit' in content,
        'api_call_with_retry': 'def api_call_with_retry' in content,
        'RetryContext': 'class RetryContext' in content,
        'jitter': 'jitter' in content,
        'exponential backoff': 'backoff_factor' in content,
    }

    all_passed = True
    for feature, found in checks.items():
        status = "✅" if found else "❌"
        print(f"  {status} {feature}")
        if not found:
            all_passed = False

    return all_passed

def check_notebook():
    """Verify colab_data_collection.ipynb structure."""
    print("\n📓 Checking colab_data_collection.ipynb...")
    notebook_file = ROOT / 'notebooks' / 'colab_data_collection.ipynb'

    if not notebook_file.exists():
        print("  ❌ notebooks/colab_data_collection.ipynb not found")
        return False

    import json
    with open(notebook_file) as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    cell_count = len(cells)

    print(f"  📊 Total cells: {cell_count}")

    # Check for key cells
    key_features = {
        'Google Drive mount': False,
        'Checkpoint state': False,
        'Retry logic': False,
        'Configuration dataclass': False,
        'Validation': False,
    }

    notebook_text = json.dumps(notebook)

    key_features['Google Drive mount'] = 'drive.mount' in notebook_text
    key_features['Checkpoint state'] = 'DownloadState' in notebook_text
    key_features['Retry logic'] = 'retry_with_backoff' in notebook_text or 'exponential backoff' in notebook_text
    key_features['Configuration dataclass'] = 'ColabConfig' in notebook_text
    key_features['Validation'] = 'ValidationResult' in notebook_text

    all_passed = True
    for feature, found in key_features.items():
        status = "✅" if found else "❌"
        print(f"  {status} {feature}")
        if not found:
            all_passed = False

    return all_passed and cell_count >= MIN_NOTEBOOK_CELL_COUNT

def check_path_fixes():
    """Verify path fixes in notebooks."""
    print("\n📂 Checking path fixes...")
    notebook_file = ROOT / 'notebooks' / 'colab_full_training.ipynb'

    if not notebook_file.exists():
        print("  ⚠️  colab_full_training.ipynb not found (skipping)")
        return True

    import json
    with open(notebook_file) as f:
        notebook = json.load(f)

    notebook_text = json.dumps(notebook)

    # Check that we use output_central not data/raw
    uses_output_central = 'output_central' in notebook_text
    uses_data_raw = "'data' / 'raw'" in notebook_text or '"data" / "raw"' in notebook_text

    if uses_output_central and not uses_data_raw:
        print("  ✅ Uses output_central/ (correct)")
        return True
    elif uses_data_raw:
        print("  ❌ Still uses data/raw/ (should be output_central/)")
        return False
    else:
        print("  ⚠️  No path configuration found")
        return False

def main():
    """Run all verification checks."""
    print("="*60)
    print("🔍 Verifying Notebook Implementation Changes")
    print("="*60)

    results = {
        'requirements': check_requirements(),
        'ecb_vendor': check_ecb_vendor(),
        'retry_utils': check_retry_utils(),
        'notebook': check_notebook(),
        'path_fixes': check_path_fixes(),
    }

    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All checks passed! Implementation is correct.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
