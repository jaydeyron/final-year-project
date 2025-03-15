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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS, images)
app.mount("/assets", StaticFiles(directory="../front-end/dist/assets"), name="assets")

# Handle client-side routing - must be before static file mounting
@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    if full_path.startswith("assets/"):
        return None  # Let the static files handler deal with assets
    return FileResponse("../front-end/dist/index.html")

# Note: Removed the root static files mount since the catch-all handler above will serve index.html