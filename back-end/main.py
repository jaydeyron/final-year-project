from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import sys
import logging
import asyncio
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
training_progress = 0
training_lock = asyncio.Lock()

# CORS Middleware - only needed during development
app.add_middleware(
    CORSMiddleware,
    # During development, frontend runs on 5173; in production everything is on 8000
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
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

class ModelRequest(BaseModel):
    symbol: str

# Load niftyStocks data
def load_nifty_stocks():
    try:
        with open('../front-end/src/data/niftyStocks.js', 'r') as f:
            content = f.read()
            # Extract the array part between the export and the closing bracket
            start_index = content.find('[')
            end_index = content.rfind(']') + 1
            json_content = content[start_index:end_index]
            return json.loads(json_content)
    except Exception as e:
        logger.error(f"Error loading niftyStocks: {str(e)}")
        # Fallback to a minimal stock list if file can't be loaded
        return [
            {"symbol": "SENSEX", "bseSymbol": "^BSESN"},
            {"symbol": "TCS", "bseSymbol": "TCS.BO"}
        ]

# Map displaySymbol to bseSymbol
def map_symbol_to_bse(symbol):
    stocks = load_nifty_stocks()
    
    # Direct match
    for stock in stocks:
        if stock["symbol"] == symbol:
            logger.info(f"Mapped {symbol} to {stock['bseSymbol']}")
            return stock["bseSymbol"]
    
    # Remove BSE: prefix if present
    if symbol.startswith("BSE:"):
        clean_symbol = symbol[4:]
        for stock in stocks:
            if stock["symbol"] == clean_symbol:
                logger.info(f"Mapped {symbol} to {stock['bseSymbol']}")
                return stock["bseSymbol"]
    
    # Fallback: add .BO suffix if not found
    logger.warning(f"No mapping found for {symbol}, using fallback")
    return f"{symbol}.BO" if not symbol.endswith(".BO") else symbol

@app.post('/train')
async def train_model(request: ModelRequest):
    try:
        # Map the display symbol to BSE symbol
        display_symbol = request.symbol
        bse_symbol = map_symbol_to_bse(display_symbol)
        
        logger.info(f"Starting training for {display_symbol} (BSE: {bse_symbol})")
        async with training_lock:
            training_progress = 0
        
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            'tft/train.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            async with training_lock:
                training_progress = 100
            return {"status": "success"}
        else:
            raise HTTPException(status_code=500, detail=stderr.decode())
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict')
async def predict(request: ModelRequest):
    try:
        # Map the display symbol to BSE symbol
        display_symbol = request.symbol
        bse_symbol = map_symbol_to_bse(display_symbol)
        
        logger.info(f"Predicting for {display_symbol} (BSE: {bse_symbol})")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            'tft/predict.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            prediction = float(stdout.decode().strip())
            return {"prediction": prediction}
        else:
            raise HTTPException(status_code=500, detail=stderr.decode())
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/progress')
async def get_progress():
    return {"progress": training_progress}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Explicitly set port to 8000