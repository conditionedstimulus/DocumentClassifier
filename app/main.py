from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager
from app.core.config import settings
from app.utils.globals import GlobalsMiddleware, g
from app.services.models.LiltClassifier import (
    LiltClassifier,
)
from app.api.classifiers import document_classifier

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    lilt_model = LiltClassifier(model_path=settings.MODEL_PATH)
    g.set_default("lilt_classifier", lilt_model)
    print("startup fastapi")
    yield
    # shutdown
    g.cleanup()


def init_app():
    desc = """
    This is a demo Fast API application for Document Classification with LiLT.
    The model was trained on a sample of RVL-CDIP dataset.
    """

    app = FastAPI(title=settings.PROJECT_NAME, description=desc, lifespan=lifespan)

    app.add_middleware(GlobalsMiddleware)

    app.include_router(document_classifier.router)

    @app.get('/')
    async def index():
        return {'message': 'ML Document Classifier API'}

    return app


app = init_app()
