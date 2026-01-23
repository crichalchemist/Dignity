import argparse
import csv
import os
from datetime import datetime

from histdata.api import download_hist_data


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            # Directory already exists - this is fine
            pass
        else:
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Download historical FX data for all pairs")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year for downloads")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="End year for downloads")
    parser.add_argument("--max-downloads", type=int, default=100, help="Maximum number of downloads to attempt")
    parser.add_argument("--output", type=str, default='output', help="Output directory for data")
    return parser.parse_args()


def download_all(args):
    """Download FX data with proper bounds to prevent infinite loops."""
    output = args.output

    # Prefer repo-root pairs.csv to keep full symbol coverage; allow override via env.
    default_pairs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pairs.csv'))
    alt_pairs = os.path.join(os.path.dirname(__file__), 'pairs.csv')
    pairs_file = os.environ.get("PAIRS_CSV", default_pairs if os.path.exists(default_pairs) else alt_pairs)
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found at {pairs_file}")

    total_downloads = 0
    max_downloads = args.max_downloads

    with open(pairs_file) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            if total_downloads >= max_downloads:
                print(f"Reached max downloads limit ({max_downloads}). Stopping.")
                break

            currency_pair_name, pair, history_first_trading_month = row
            start_year = int(history_first_trading_month[0:4])
            end_year = min(args.end_year, datetime.now().year)

            print(f"Downloading {currency_pair_name} ({pair}) from {start_year} to {end_year}")
            output_folder = os.path.join(output, pair)
            mkdir_p(output_folder)

            try:
                for year in range(start_year, end_year + 1):
                    if total_downloads >= max_downloads:
                        print(f"Reached max downloads limit ({max_downloads}). Stopping.")
                        break

                    could_download_full_year = False
                    try:
                        result = download_hist_data(year=year,
                                                    pair=pair,
                                                    output_directory=output_folder,
                                                    verbose=False)
                        print(f'- {year}: {result}')
                        could_download_full_year = True
                        total_downloads += 1
                    except AssertionError:
                        pass  # lets download it month by month.

                    if not could_download_full_year:
                        for month in range(1, 13):
                            if total_downloads >= max_downloads:
                                print(f"Reached max downloads limit ({max_downloads}). Stopping.")
                                break

                            try:
                                result = download_hist_data(year=str(year),
                                                            month=str(month),
                                                            pair=pair,
                                                            output_directory=output_folder,
                                                            verbose=False)
                                print(f'- {year}-{month:02d}: {result}')
                                total_downloads += 1
                            except Exception as e:
                                print(f"Failed to download {pair} {year}-{month}: {e}")
                                break  # Stop trying months for this year if one fails

            except Exception as e:
                print(f'[DONE] for currency {currency_pair_name} - error: {e}')

            print(f"Completed {currency_pair_name}. Total downloads so far: {total_downloads}")

    print(f"Download complete. Total downloads: {total_downloads}")


if __name__ == '__main__':
    args = parse_args()
    download_all(args)
