from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = os.environ['APP_NAME']

    TOKENIZER_NAME: str = os.environ['TOKENIZER_NAME']
    MODEL_NAME: str = os.environ['MODEL_NAME']

    class Config:
        case_sensitive = True


settings = Settings()
