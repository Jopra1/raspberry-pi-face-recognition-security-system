# upload_only.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, shutil, time
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

root_dataset_path = "dataset"
os.makedirs(root_dataset_path, exist_ok=True)

@app.post("/upload-images")
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    person_folder = os.path.join(root_dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(person_folder, f"{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    return JSONResponse({"status": "success", "files": saved_files})

@app.get("/")
def root():
    return {"status": "upload-only server running"}
