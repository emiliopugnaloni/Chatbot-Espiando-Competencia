## pip install pip install google-api-python-client

import os
import re
import time
import zipfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io

# Constants
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'google_cloud_service_account.json'

def authenticate():
    """Authenticate with Google Drive API using a service account."""
    return service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

def sanitize_file_name(file_name):
    """Replace invalid characters in a file name with underscores."""
    return re.sub(r'[<>:"/\\|?*]', '_', file_name)

def unzip_and_delete(file_path):
    """
    Unzip a file to its containing directory and delete the zip file.
    :param file_path: Path to the zip file.
    """
    try:
        # Extract directory name from the zip file name
        extract_dir = file_path.replace('.zip', '')
        os.makedirs(extract_dir, exist_ok=True)  # Create folder if it doesn't exist

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)  # Extract to the named folder
        print(f"Unzipped: {file_path} to {extract_dir}")
        time.sleep(1)  # Allow file handles to release
        os.remove(file_path)
        print(f"Unzipped and deleted: {file_path}")
    except zipfile.BadZipFile:
        print(f"Invalid zip file: {file_path}")
    except Exception as e:
        print(f"Error handling {file_path}: {e}")

def download_file(service, file_id, file_name, download_path):
    """
    Download a single file from Google Drive.
    :param service: Google Drive service instance.
    :param file_id: File ID from Google Drive.
    :param file_name: Name of the file.
    :param download_path: Local path for saving the file.
    """
    local_file_path = os.path.join(download_path, sanitize_file_name(file_name))
    with io.FileIO(local_file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=file_id))
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {file_name}: {int(status.progress() * 100)}% complete.")
    return local_file_path

def download_and_process_folder(folder_id, download_path):
    """
    Download all files in a Google Drive folder and process them.
    :param folder_id: Google Drive folder ID.
    :param download_path: Local directory to save files.
    """
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    os.makedirs(download_path, exist_ok=True)

    query = f"'{folder_id}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'"
    files = service.files().list(q=query, fields="files(id, name)").execute().get('files', [])

    if not files:
        print("No files found in the folder.")
        return

    print(f"Found {len(files)} file(s) in the folder.")
    for file in files:
        try:
            file_path = download_file(service, file['id'], file['name'], download_path)
            if file_path.endswith('.zip'):
                unzip_and_delete(file_path)
        except Exception as e:
            print(f"Error downloading {file['name']}: {e}")

    print("All files processed successfully.")
    print()


if __name__ == "__main__":
    # Example usage
    folder_id = '1UefA4gnBD1X4Whi9aJ6gwLGW5cKWwwAo'
    download_path = 'Chatbot/tmp'
    download_and_process_folder(folder_id, download_path)
    print("Done.")
