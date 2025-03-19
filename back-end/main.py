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

# Define niftyStocks data directly in Python
NIFTY_STOCKS = [
    { 
        "name": "Bombay Stock Exchange SENSEX",
        "symbol": "SENSEX",
        "tradingViewSymbol": "SENSEX",
        "bseSymbol": "^BSESN",
        "startDate": "1997-07-01"
    },
    { 
        "name": "Asian Paints Limited",
        "symbol": "ASIANPAINT",
        "tradingViewSymbol": "BSE:ASIANPAINT",
        "bseSymbol": "ASIANPAINT.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Axis Bank Limited",
        "symbol": "AXISBANK",
        "tradingViewSymbol": "BSE:AXISBANK",
        "bseSymbol": "AXISBANK.BO",
        "startDate": "2000-05-23"
    },
    { 
        "name": "Bajaj Finance Limited",
        "symbol": "BAJFINANCE",
        "tradingViewSymbol": "BSE:BAJFINANCE",
        "bseSymbol": "BAJFINANCE.BO",
        "startDate": "2000-06-16"
    },
    { 
        "name": "Bharti Airtel Limited",
        "symbol": "BHARTIARTL",
        "tradingViewSymbol": "BSE:BHARTIARTL",
        "bseSymbol": "BHARTIARTL.BO",
        "startDate": "2002-02-18"
    },
    { 
        "name": "HDFC Bank Limited",
        "symbol": "HDFCBANK",
        "tradingViewSymbol": "BSE:HDFCBANK",
        "bseSymbol": "HDFCBANK.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Hindustan Unilever Limited",
        "symbol": "HINDUNILVR",
        "tradingViewSymbol": "BSE:HINDUNILVR",
        "bseSymbol": "HINDUNILVR.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "ICICI Bank Limited",
        "symbol": "ICICIBANK",
        "tradingViewSymbol": "BSE:ICICIBANK",
        "bseSymbol": "ICICIBANK.BO",
        "startDate": "2002-04-29"
    },
    { 
        "name": "Infosys Limited",
        "symbol": "INFY",
        "tradingViewSymbol": "BSE:INFY",
        "bseSymbol": "INFY.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "ITC Limited",
        "symbol": "ITC",
        "tradingViewSymbol": "BSE:ITC",
        "bseSymbol": "ITC.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Kotak Mahindra Bank Limited",
        "symbol": "KOTAKBANK",
        "tradingViewSymbol": "BSE:KOTAKBANK",
        "bseSymbol": "KOTAKBANK.BO",
        "startDate": "2000-03-21"
    },
    { 
        "name": "Larsen & Toubro Limited",
        "symbol": "LT",
        "tradingViewSymbol": "BSE:LT",
        "bseSymbol": "LT.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Maruti Suzuki India Limited",
        "symbol": "MARUTI",
        "tradingViewSymbol": "BSE:MARUTI",
        "bseSymbol": "MARUTI.BO",
        "startDate": "2003-07-09"
    },
    { 
        "name": "Power Grid Corporation of India Limited",
        "symbol": "POWERGRID",
        "tradingViewSymbol": "BSE:POWERGRID",
        "bseSymbol": "POWERGRID.BO",
        "startDate": "2007-10-05"
    },
    { 
        "name": "Reliance Industries Limited",
        "symbol": "RELIANCE",
        "tradingViewSymbol": "BSE:RELIANCE",
        "bseSymbol": "RELIANCE.BO",
        "startDate": "2000-01-04"
    },
    { 
        "name": "State Bank of India",
        "symbol": "SBIN",
        "tradingViewSymbol": "BSE:SBIN",
        "bseSymbol": "SBIN.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Sun Pharmaceutical Industries Limited",
        "symbol": "SUNPHARMA",
        "tradingViewSymbol": "BSE:SUNPHARMA",
        "bseSymbol": "SUNPHARMA.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Tata Motors Limited",
        "symbol": "TATAMOTORS",
        "tradingViewSymbol": "BSE:TATAMOTORS",
        "bseSymbol": "TATAMOTORS.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Tata Steel Limited",
        "symbol": "TATASTEEL",
        "tradingViewSymbol": "BSE:TATASTEEL",
        "bseSymbol": "TATASTEEL.BO",
        "startDate": "2000-01-03"
    },
    { 
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS",
        "tradingViewSymbol": "BSE:TCS",
        "bseSymbol": "TCS.BO",
        "startDate": "2024-08-25"
    }
]

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
    if (full_path.startswith("assets/")):
        return None  # Let the static files handler deal with assets
    return FileResponse("../front-end/dist/index.html")

class ModelRequest(BaseModel):
    symbol: str
    start_date: str = None  # Optional start date
    model_symbol: str = None  # Optional model override (e.g., "TCS")

# Map displaySymbol to bseSymbol
def map_symbol_to_bse(symbol):
    """Map a display symbol to its BSE/Yahoo Finance symbol"""
    # Direct match against symbol field
    for stock in NIFTY_STOCKS:
        if stock["symbol"] == symbol:
            logger.info(f"Mapped {symbol} to {stock['bseSymbol']}")
            return stock["bseSymbol"], stock["symbol"]
    
    # Check if it's already a BSE symbol
    for stock in NIFTY_STOCKS:
        if stock["bseSymbol"] == symbol:
            logger.info(f"{symbol} is already a BSE symbol")
            return symbol, stock["symbol"]
    
    # Remove BSE: prefix if present and try again
    if symbol.startswith("BSE:"):
        clean_symbol = symbol[4:]
        for stock in NIFTY_STOCKS:
            if stock["symbol"] == clean_symbol:
                logger.info(f"Mapped {symbol} to {stock['bseSymbol']}")
                return stock["bseSymbol"], stock["symbol"]
    
    # Fallback: add .BO suffix if not found
    logger.warning(f"No mapping found for {symbol}, using fallback")
    display_symbol = symbol.replace('.BO', '') if symbol.endswith('.BO') else symbol
    bse_symbol = f"{display_symbol}.BO" if not symbol.endswith(".BO") else symbol
    return bse_symbol, display_symbol

@app.post('/train')
async def train_model(request: ModelRequest):
    try:
        # Map the symbol to BSE symbol and get display symbol
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        logger.info(f"Starting training for display:{display_symbol} using data:{bse_symbol}")
        
        # Set up the command with optional start date
        cmd = [
            sys.executable,
            'tft/train.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
        ]
        
        # Add start date if provided
        if request.start_date:
            cmd.extend(['--start-date', request.start_date])
            logger.info(f"Using custom start date: {request.start_date}")
        
        global training_progress
        async with training_lock:
            training_progress = 0
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            async with training_lock:
                training_progress = 100
                logger.info(f"Training completed. Set training_progress to {training_progress}")
            return {"status": "success"}
        else:
            raise HTTPException(status_code=500, detail=stderr.decode())
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict')
async def predict(request: ModelRequest):
    try:
        # Map the symbol to BSE symbol and get display symbol
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        # Allow overriding which model to use
        model_override = request.model_symbol
        if model_override:
            logger.info(f"Using {model_override} model for prediction")
        
        cmd = [
            sys.executable,
            'tft/predict.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
        ]
        
        # Add model override if specified
        if model_override:
            cmd.extend(['--model-symbol', model_override])
        
        logger.info(f"Predicting for display:{display_symbol} using data:{bse_symbol}")
        
        # Set up environment with explicit PYTHONIOENCODING
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        # Log everything for debugging
        stdout_content = stdout.decode().strip()
        stderr_content = stderr.decode().strip() if stderr else ""
        
        if stderr_content:
            logger.info(f"Prediction stderr output: {stderr_content}")
        
        logger.info(f"Raw prediction output: '{stdout_content}'")
        
        # Better error handling
        if process.returncode != 0:
            error_msg = stderr_content or "Unknown prediction error"
            logger.error(f"Prediction process failed with code {process.returncode}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
        # More robust parsing - strip all whitespace and non-numeric content
        try:
            # Filter the output to only get numeric characters plus decimal point
            import re
            numeric_part = re.search(r'\d+\.\d+', stdout_content)
            if numeric_part:
                prediction = float(numeric_part.group(0))
                logger.info(f"Successfully parsed prediction: {prediction}")
                return {"prediction": prediction}
            else:
                logger.error(f"No numeric value found in output: '{stdout_content}'")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Could not find prediction value in output"
                )
        except Exception as e:
            logger.error(f"Error parsing prediction output: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse prediction: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/last-close')
async def get_last_closing_price(request: ModelRequest):
    """Get the last closing price for a specific symbol"""
    try:
        # Map the symbol to BSE symbol
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        # Use the data_loader functionality to get recent data for the symbol
        cmd = [
            sys.executable,
            'tft/get_last_price.py',
            '--symbol', bse_symbol
        ]
        
        logger.info(f"Fetching last close price for {bse_symbol}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error fetching last close: {stderr.decode()}")
            raise HTTPException(status_code=500, detail="Failed to fetch last closing price")
            
        last_close = float(stdout.decode().strip())
        logger.info(f"Last close for {bse_symbol}: {last_close}")
        
        return {"symbol": display_symbol, "lastClose": last_close}
        
    except Exception as e:
        logger.error(f"Error fetching last close: {str(e)}")
        # Return a reasonable fallback if we can't get the actual value
        # In a production app, you'd want to handle this more gracefully
        return {"symbol": input_symbol, "lastClose": None}

@app.get('/progress')
async def get_progress():
    try:
        # Check for progress file from trainer
        progress_file = os.path.join(os.path.dirname(__file__), 'tft', 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                progress = data.get('progress', training_progress)
                logger.info(f"Progress from file: {progress}%")
                return {"progress": progress}
    except Exception as e:
        logger.error(f"Error reading progress file: {e}")
    
    # Return the global progress as fallback
    logger.info(f"Using global progress: {training_progress}%")
    return {"progress": training_progress}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Explicitly set port to 8000