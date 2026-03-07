# Data Download Guide

## Prerequisites

- [rclone](https://rclone.org/install/) installed
- `unzip` installed

---

## Step 1: Configure rclone

Run the interactive rclone configuration to add a Google Drive remote named `gdrive`:

```bash
rclone config
```

Follow the prompts:
1. Select `n` for new remote
2. Name it `gdrive`
3. Select `drive` as the storage type
4. Follow the OAuth flow to authorize access to your Google account
5. Leave other options as default and confirm

Verify the connection works:

```bash
rclone ls gdrive:
```

---

## Step 2: Copy file names from Google Drive

Go to the shared Google Drive folder:

[https://drive.google.com/drive/folders/1WMZtwoAmxtbmu3WDWbJVdYetsJ9eVFXR?usp=sharing](https://drive.google.com/drive/folders/1WMZtwoAmxtbmu3WDWbJVdYetsJ9eVFXR?usp=sharing)

Browse the raw files you want to download and note their exact filenames (including the `.zip` extension).

---

## Step 3: Populate files.txt

Edit `files.txt` and add one filename per line, for example:

```
gpt_balanced_20260225_110457.zip
gemini_balanced_20260225_110457.zip
```

Blank lines and non-`.zip` entries will be skipped automatically.

---

## Step 4: Configure download.sh

Open `download.sh` and update the following variables at the top of the file:

| Variable | Description |
|---|---|
| `REMOTE_BASE` | The path inside your Google Drive where the raw zip files are stored (relative to the rclone remote root). Change this if the files are located in a different folder on Drive. |
| `OUT_DIR` | The local directory where zip files will be **extracted**. All downloaded archives are unzipped here. |
| `DOWNLOAD_DIR` | The local **temporary** directory where zip files are saved before extraction. Zips are deleted after successful extraction. |

Example:

```bash
REMOTE_BASE="ImageDetection_Benchmark/final_bench_raw"
OUT_DIR="/data/your_user/omni_backup/raw_outputs"
DOWNLOAD_DIR="/data/your_user/omni_backup/temp"
```

---

## Step 5: Run the download script

```bash
bash download.sh
```

The script will:
1. Read each filename from `files.txt`
2. Download the zip from `gdrive:${REMOTE_BASE}/<filename>` into `DOWNLOAD_DIR`
3. Extract the zip into `OUT_DIR`
4. Delete the zip after successful extraction
