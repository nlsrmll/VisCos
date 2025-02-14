import os


class Settings:
    BASE_DIR = os.getcwd()
    DATA_BASE_PATH = os.path.normpath(f"{BASE_DIR}/data")
    IMAGE_BASE_PATH = os.path.normpath(f"{BASE_DIR}/data/images")


settings = Settings()
