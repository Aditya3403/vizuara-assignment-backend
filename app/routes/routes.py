from fastapi import APIRouter, UploadFile, File
from app.controllers.controller import (
    handle_upload,
    handle_preprocess,
    handle_split,
    handle_train
)

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    return handle_upload(file)

@router.post("/preprocess")
async def preprocess(standardize: bool, normalize: bool):
    return handle_preprocess(standardize, normalize)

@router.post("/split")
async def split(split_ratio: int):
    return handle_split(split_ratio)

@router.post("/train")
async def train(model: str):
    return handle_train(model)
