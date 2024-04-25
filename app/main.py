from app.utils.logger import logger
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.classifiers import document_classifier
from app.core.config import settings
from app.services.models.LayoutLMv3Classifier import LayoutLMv3Classifier
from app.utils.globals import GlobalsMiddleware, g


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    classifier_model = LayoutLMv3Classifier(
        model_name=settings.MODEL_NAME, tokenizer_name=settings.TOKENIZER_NAME
    )
    g.set_default("lmv3_classifier", classifier_model)
    print("startup fastapi")
    yield
    # shutdown
    g.cleanup()


def init_app():
    desc = """
    This is a demo Fast API application for Document Classification with LiLT.
    The model was trained on a sample of RVL-CDIP dataset.
    """

    logger.info("Starting FAST API application...")

    app = FastAPI(title=settings.PROJECT_NAME, description=desc, lifespan=lifespan)

    app.add_middleware(GlobalsMiddleware)

    app.include_router(document_classifier.router)

    @app.get("/")
    async def index():
        return {"message": "ML Document Classifier API"}

    return app


app = init_app()
