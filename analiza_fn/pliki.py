import os

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def check_file(file_name):
    return os.path.exists(file_name)