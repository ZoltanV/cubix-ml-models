# Usage Instructions:
# Install dependencies:
#     pip install requests pandas openpyxl certifi
# Run the script to download and convert dataset:
#     python download_online_retail.py

import os
import requests
import pandas as pd
import certifi
import sys

# URL of the Online Retail dataset (Excel format)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Local paths
DATA_DIR = "data"
EXCEL_FILENAME = "Online_Retail.xlsx"
CSV_FILENAME = "Online_Retail.csv"


def download_file(url: str, dest_path: str):
    """
    Download a file from a URL to a local destination.
    """
    # verify SSL certificate using certifi bundle
    with requests.get(url, stream=True, verify=certifi.where()) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded file to {dest_path}")


def main():
    # Paths
    excel_path = os.path.join(DATA_DIR, EXCEL_FILENAME)
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

    # Download Excel dataset if not already present
    if os.path.exists(excel_path):
        print(f"Excel file already exists at {excel_path}, skipping download")
    else:
        download_file(URL, excel_path)

    # Convert to CSV if not already present
    if os.path.exists(csv_path):
        print(f"CSV file already exists at {csv_path}, skipping conversion")
    else:
        print(f"Reading Excel file from {excel_path}")
        try:
            df = pd.read_excel(excel_path)
        except ImportError:
            print("Error: Missing 'openpyxl' dependency. Install with: pip install openpyxl")
            sys.exit(1)
        df.to_csv(csv_path, index=False)
        print(f"Converted to CSV and saved to {csv_path}")


if __name__ == "__main__":
    main()
