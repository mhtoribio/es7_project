#!/usr/bin/env python3
# ***** 5th DNS Challenge at ICASSP 2023*****
# Track 2 Speakerphone Clean speech: All Languages
# -------------------------------------------------------------
# In all, you will need about 1TB to store the UNPACKED data.
# Archived, the same data takes about 550GB total.

# Please comment out the files you don't need before launching
# the script.

# NOTE: By default, the script *DOES NOT* DOWNLOAD ANY FILES!
# Please scroll down and edit this script to pick the
# downloading method that works best for you.

# -------------------------------------------------------------
# The directory structure of the unpacked data is:

# datasets_fullband 
# \-- clean_fullband 827G
#     +-- emotional_speech 2.4G
#     +-- french_speech 62G
#     +-- german_speech 319G
#     +-- italian_speech 42G
#     +-- read_speech 299G
#     +-- russian_speech 12G
#     +-- spanish_speech 65G
#     +-- vctk_wav48_silence_trimmed 27G
#     \-- VocalSet_48kHz_mono 974M

import os
import logging
import requests
import tarfile
from tqdm import tqdm
import argparse

AZURE_URL = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
OUTPUT_PATH = "./datasets_fullband"

EMO_NAMES = [
"clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2"
]

READ_NAMES = [
"clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_025_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_026_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_027_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_028_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_029_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_030_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_031_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_032_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_033_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_034_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_035_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_036_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_037_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_038_NA_NA.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.read_speech_039_NA_NA.tar.bz2"

]

SILENT_NAMES = [
"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_002.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_003.tar.bz2",
"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_004.tar.bz2"
]

# Setup argparse
parser = argparse.ArgumentParser(description="Download DNS 5th Challenge datasets")
parser.add_argument(
    "--dataset", nargs='+', choices=["emo", "read", "silent", "all"], required=True,
    help="Which dataset(s) to download: emo, read, silent, or all"
)
parser.add_argument(
    "--count-emo", type=int, default=None, help="Number of files to download from emo dataset"
)
parser.add_argument(
    "--count-read", type=int, default=None, help="Number of files to download from read dataset"
)
parser.add_argument(
    "--count-silent", type=int, default=None, help="Number of files to download from silent dataset"
)

parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv)")

args = parser.parse_args()


logging.basicConfig(
    level = logging.WARNING - 10*min(args.verbose, 2),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Determine which datasets to download
selected_datasets = args.dataset
if "all" in selected_datasets:
    selected_datasets = ["emo", "read", "silent"]

dataset_map = {
    "emo": EMO_NAMES,
    "read": READ_NAMES,
    "silent": SILENT_NAMES
}

count_map = {
    "emo": args.count_emo,
    "read": args.count_read,
    "silent": args.count_silent
}

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Download function
for ds_name in selected_datasets:
    BLOB_NAMES = dataset_map[ds_name]
    
    # Apply per-dataset count limit if specified
    if count_map[ds_name] is not None:
        BLOB_NAMES = BLOB_NAMES[:count_map[ds_name]]

    logging.info(f"Starting download for dataset: {ds_name} ({len(BLOB_NAMES)} files)")
    
    for blob in BLOB_NAMES:
        url = f"{AZURE_URL}/{blob}"
        local_path = os.path.join(OUTPUT_PATH, blob)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logging.debug(f"Downloading {blob} ...")

        # HEAD request to check size
        head = requests.head(url)
        size = head.headers.get('Content-Length', 'unknown')
        logging.debug(f"File size: {size} bytes")

        # Download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 10 * 1024 * 1024  # 10 MB chunks

            with open(local_path, "wb") as f, tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024,
                desc=blob,
                ncols=100
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        
        logging.debug(f"Done: {blob}")

        # Extract the tar.bz2
        try:
            logging.info(f"Extracting {local_path} ...")
            with tarfile.open(local_path, "r:bz2") as tar:
                tar.extractall(path=os.path.dirname(local_path))
            logging.debug(f"Extraction complete: {blob}")

            # Remove the zip file after extraction
            os.remove(local_path)
            logging.info(f"Removed archive file: {blob}")

        except Exception as e:
            logging.error(f"Failed to extract {blob}: {e}")