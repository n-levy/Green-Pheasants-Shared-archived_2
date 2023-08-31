#################################
#### Testing the python code ####
#################################

"""
This script tests the python code
of the Green Pheasants recommendation system.
"""

"""
Install and import the packages for the testing script
"""
# pip install requests
# pip install subprocess

import requests
import os
print("hi")

"""
Define functions for downloading the scripts and sample datasets from Github and saving them in the local environment
"""

def download_github_file(url, save_path):
    """
    Download a file from GitHub and save it to the specified path.
    
    Parameters:
    - url (str): The URL of the raw file from GitHub.
    - save_path (str): The local path where the file should be saved.
    """
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as file:
        file.write(response.content)

def download_multiple_files(file_list):
    """
    Download multiple files from GitHub.
    
    Parameters:
    - file_list (list of tuples): Each tuple contains the GitHub URL and the local save path.
    """
    for url, save_path in file_list:
        download_github_file(url, save_path)
        print(f"Downloaded {url} to {save_path}")

"""
Download the files from Github
"""

### Download the scripts
if __name__ == '__main__':
    files_to_download = [
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/configuration.py', './downloaded_scripts/configuration.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/functions.py', './downloaded_scripts/functions.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/train_visitors.py', './downloaded_scripts/train_visitors.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/train_users.py', './downloaded_scripts/train_users.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/choose_item_online_visitor.py', './downloaded_scripts/choose_item_online_visitor.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/choose_item_online_user.py', './downloaded_scripts/choose_item_online_user.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/choose_items_many_offline_users.py', './downloaded_scripts/choose_items_many_offline_users.py')
    ]

    download_multiple_files(files_to_download)

### Download the sample dataframes
if __name__ == '__main__':
    files_to_download = [
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/users.py', './example_data/users.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/items.py', './example_data/items.py'),
        ('https://raw.githubusercontent.com/n-levy/Green-Pheasants-Shared/main/interactions.py', './example_data/interactions.py')
    ]

    download_multiple_files(files_to_download)




