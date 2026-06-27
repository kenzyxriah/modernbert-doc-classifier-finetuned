from api_response import *
from utils import logger, redis_client
from pydantic import Field, BaseModel
from fastapi import FastAPI, Depends
from auth import verify_api_key
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from classification import classify_doc, update_category

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Document(BaseModel):
    document: str = Field(..., examples=["https://flowmonostorage.blob.core.windows.net/uploads/doc_xyz.pdf"])

    class Config:
        extra = "forbid"

class UpdateComplianceCategoryReq(BaseModel):
    document: str = Field(..., description="The document blob file URL")
    category: str = Field(..., description="The new compliance category to override with")


# Endpoints
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(
        openapi_url="/openapi.json", 
        title="Secure API")


@app.get("/", tags=["Health"])
@app.post("/", tags=["Health"])
def root():
    return generate_response(entity={"message": "Welcome to Secure AI API"}, message="Health Check Passed")


@app.patch("/api/v1/BERT/update-doc-category", summary="Update the compliance category for a document", tags=["AI Suite"])
async def update_doc_category(payload: UpdateComplianceCategoryReq, current_user: str = Depends(verify_api_key)):
    try:
        result = await update_category(payload.document, payload.category)
        return generate_response(entity=result, message="Compliance category updated")

    except Exception as e:
        err_msg = str(e)
        logger.exception("An unexpected error occurred updating document category.\n")
        return generate_error_response(err_msg, message=f"Error updating document category", status_code=500)

@app.post("/api/v1/BERT/classify", summary="Fetch the compliance category for a document", tags=["AI Suite"])
def fetch_doc_category(payload: Document, current_user: str = Depends(verify_api_key)):
    try:
        result = classify_doc(payload.document)
        return generate_response(entity=result, message="Compliance category retrieved")
        
    except Exception as e:
        err_msg = str(e)
        logger.exception("An unexpected error occurred fetching document category.\n")
        return generate_error_response(err_msg, message=f"Error fetching document category", status_code=500)