from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = os.environ['APP_NAME']

    MODEL_PATH: str = os.environ['LiLT_PATH']
    MODEL_NAME: str = os.environ['LiLT_NAME']

    class Config:
        case_sensitive = True


settings = Settings()
