from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets first
app.mount("/assets", StaticFiles(directory="../front-end/dist/assets"), name="assets")

# API endpoints
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint - serve About page
@app.get("/")
async def root():
    return FileResponse("../front-end/dist/index.html")

# Home page endpoint
@app.get("/home")
async def home():
    return FileResponse("../front-end/dist/index.html")

# All other routes should serve the SPA
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    return FileResponse("../front-end/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)