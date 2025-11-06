from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
import time
from typing import List

app = FastAPI()

# Allow CORS for testing with React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root folder to store dataset
root_dataset_path = "dataset"
os.makedirs(root_dataset_path, exist_ok=True)

@app.post("/upload-images")
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Upload multiple images for a person and save them under /dataset/<name>/.
    """
    if not name or not files:
        return JSONResponse(status_code=400, content={"message": "Name and files are required."})

    # Create folder for person if it doesn't exist
    person_folder = os.path.join(root_dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = os.path.join(person_folder, f"{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    return {"status": "success",
            "message": f"{len(saved_files)} files uploaded for {name}.",
            "files": saved_files}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
