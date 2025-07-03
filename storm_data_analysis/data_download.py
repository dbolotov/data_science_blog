"""
This script downloads and extracts NOAA Storm Events data files (details, fatalities, and/or locations)
from user-supplied years. It saves compressed files to 'data/download', extracts them to 
'data/extracted', and checks header consistency across the extracted CSVs.
"""

import os
import requests
import gzip
import shutil
from tqdm import tqdm
import glob
import csv

start_year = 2000 #1950 or later
end_year = 2024

# Set up directories relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(base_dir, "data", "download")
extracted_dir = os.path.join(base_dir, "data", "extracted")

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extracted_dir, exist_ok=True)

base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

file_types = ["details"] #choose one or more from "details", "fatalities", "locations"
total_attempted = 0
successful_downloads = 0
successful_extractions = 0


def download_file(url, output_path):
    global successful_downloads
    if os.path.exists(output_path):
        print(f"Skipped download (already exists): {output_path}")
        return
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {output_path}")
        successful_downloads += 1
    except requests.HTTPError as e:
        print(f"Failed to download {url} ({e})")


def extract_gz_file(input_path, output_path):
    global successful_extractions
    if os.path.exists(output_path):
        print(f"Skipped extraction (already exists): {output_path}")
        return
    try:
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {output_path}")
        successful_extractions += 1
    except Exception as e:
        print(f"Failed to extract {input_path} ({e})")

# Download and extract
for year in tqdm(range(start_year, end_year + 1)):
    for ftype in file_types:
        total_attempted += 1
        filename = f"StormEvents_{ftype}-ftp_v1.0_d{year}_c20250520.csv.gz"
        download_path = os.path.join(download_dir, filename)
        extracted_filename = filename[:-3]  # Remove .gz
        extracted_path = os.path.join(extracted_dir, extracted_filename)

        # Download
        download_file(base_url + filename, download_path)

        # Extract
        extract_gz_file(download_path, extracted_path)

print("\nAll downloads and extractions complete.")
print(f"Total files attempted: {total_attempted}")
print(f".gz files downloaded: {successful_downloads}")
print(f".csv files extracted: {successful_extractions}")

# Check headers for consistency
print("\nChecking CSV header consistency...")

for ftype in file_types:
    header_set = set()
    ftype_files = glob.glob(os.path.join(extracted_dir, f"*{ftype}*.csv"))
    for file in ftype_files:
        with open(file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = tuple(next(reader))
            header_set.add(header)

    print(f"\n{ftype.capitalize()} file headers ({len(header_set)} unique):")
    for header in header_set:
        print(header)

    if len(header_set) == 1:
        print(f"All {ftype} files have consistent headers.")
    else:
        print(f"WARNING: Inconsistent headers found in {ftype} files.")
