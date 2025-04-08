import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel, validator
from typing import Optional
import os
import sys
import logging
import asyncio
import json
import re  # Add regex module for better pattern matching
from datetime import datetime
from tft.config import Config  # Add this import

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

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

# Define model request schema with improved date validation
class ModelRequest(BaseModel):
    symbol: str
    start_date: Optional[str] = None  # Optional start date 
    model_symbol: Optional[str] = None  # Optional model override (e.g., "TCS")
    
    # Add a validator for start_date format
    @validator('start_date')
    def validate_start_date(cls, v):
        if v is None:
            return v
        try:
            # Validate the date format
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('start_date must be in YYYY-MM-DD format')

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

# API endpoints - define first before static routes
@app.get('/progress')
async def get_progress():
    from fastapi.responses import JSONResponse
    
    try:
        # Initialize with a default response
        response_data = {"progress": training_progress}
        
        # Get path to progress file
        progress_file = os.path.join(os.path.dirname(__file__), 'tft', 'progress.json')
        
        # Check if file exists with more detailed logging
        if os.path.exists(progress_file):
            try:
                # Check file age
                file_age = time.time() - os.path.getmtime(progress_file)
                if file_age > 60:
                    logger.warning(f"Progress file is {file_age:.1f} seconds old")
                
                # Read file content with extra error handling
                try:
                    with open(progress_file, 'r') as f:
                        file_content = f.read().strip()
                        logger.debug(f"Raw progress file content: '{file_content}'")
                except Exception as read_error:
                    logger.error(f"Error reading progress file: {str(read_error)}")
                    file_content = ""
                    
                if file_content:  # Only try to parse if file has content
                    try:
                        # Parse JSON with detailed error handling
                        try:
                            data = json.loads(file_content)
                        except json.JSONDecodeError as json_error:
                            logger.error(f"Failed to parse progress file: {str(json_error)}, content: '{file_content}'")
                            # Try to sanitize and repair JSON in case of common formatting issues
                            if '{' in file_content and '}' in file_content:
                                repaired_content = file_content.replace("'", '"')
                                # Extract content between { and }
                                start_idx = repaired_content.find('{')
                                end_idx = repaired_content.rfind('}') + 1
                                if start_idx >= 0 and end_idx > start_idx:
                                    try:
                                        data = json.loads(repaired_content[start_idx:end_idx])
                                        logger.info("Successfully repaired and parsed JSON content")
                                    except:
                                        # If repair fails, use empty dict
                                        data = {}
                                else:
                                    data = {}
                            else:
                                data = {}
                        
                        # Ensure progress is a number (or can be converted to one)
                        if 'progress' in data:
                            try:
                                progress_value = int(float(data['progress']))
                                response_data["progress"] = progress_value
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid progress value: {data['progress']}")
                        
                        # Include any error message with safer string handling
                        if 'error' in data:
                            error_value = data.get('error')
                            if error_value is not None:
                                # Safely convert error message to string, avoiding formatting issues
                                try:
                                    error_str = str(error_value)
                                    # Replace any format specifiers to avoid issues with f-strings
                                    error_str = error_str.replace('{', '{{').replace('}', '}}')
                                    response_data["error"] = error_str
                                except Exception as str_error:
                                    logger.error(f"Error converting error message to string: {str_error}")
                                    response_data["error"] = "Error message conversion failed"
                            else:
                                response_data["error"] = "Unknown error"
                    except Exception as e:
                        logger.error(f"Error processing progress data: {str(e)}")
            except Exception as e:
                logger.error(f"Error reading progress file: {str(e) if e is not None else 'Unknown error'}")
        else:
            logger.info("Progress file not found, using global progress value")
    except Exception as e:
        logger.error(f"Error in progress endpoint: {str(e) if e is not None else 'Unknown error'}")
        # On any error, return a safe default
        response_data = {"progress": 0}
    
    # Return with explicit no-cache headers
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache"
    }
    
    # Log the response we're sending with safe string handling
    try:
        # Create a safe copy of the response data for logging
        log_data = response_data.copy()
        if 'error' in log_data:
            # Truncate error message to prevent log issues
            error_msg = log_data['error']
            if len(error_msg) > 100:
                log_data['error'] = error_msg[:100] + "..." 
        logger.info(f"Sending progress response: {log_data}")
    except Exception as log_error:
        logger.error(f"Error logging response: {str(log_error) if log_error is not None else 'Unknown error'}")
    
    return JSONResponse(content=response_data, headers=headers)

@app.get('/api-test')
async def api_test():
    return JSONResponse(
        content={"status": "ok", "message": "API is working correctly"},
        headers={
            "Content-Type": "application/json",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        }
    )

@app.post('/train')
async def train_model(request: ModelRequest):
    try:
        # Map the symbol to BSE symbol and get display symbol
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        logger.info(f"Starting training for display:{display_symbol} using data:{bse_symbol}")
        
        # Initialize progress file to ensure it exists with clean state
        progress_file = os.path.join(os.path.dirname(__file__), 'tft', 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump({"progress": 0}, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Set up the command with optional start date
        cmd = [
            sys.executable,
            'tft/train.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
        ]
        
        # Add start date if provided - add format debugging
        if request.start_date:
            cmd.extend(['--start-date', request.start_date])
            logger.info(f"Using custom start date: {request.start_date} (format validation passed)")
        
        # Update global progress tracking
        global training_progress
        async with training_lock:
            training_progress = 0
        
        # Execute the training process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Don't wait for completion - return immediately so UI can start polling progress
        return {"status": "training_started"}
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/update-model')
async def update_model(request: ModelRequest):
    """Incrementally update an existing model with latest data"""
    try:
        # Map the symbol to BSE symbol and get display symbol
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        logger.info(f"Starting incremental update for display:{display_symbol} using data:{bse_symbol}")
        
        # Initialize progress file to show processing
        progress_file = os.path.join(os.path.dirname(__file__), 'tft', 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump({"progress": 0, "update_type": "incremental"}, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Set up the command
        cmd = [
            sys.executable,
            'tft/incremental_update.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
            '--epochs', '10'  # Use fewer epochs for incremental updates
        ]
        
        # Execute the update process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start process but don't wait
        with open(progress_file, 'w') as f:
            json.dump({"progress": 10, "update_type": "incremental"}, f)
            
        # Return immediately so UI can show processing
        return {"status": "update_started"}
            
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
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
        
        # Run prediction process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        # Clean and log outputs
        stdout_content = stdout.decode().strip() if stdout else ""
        stderr_content = stderr.decode().strip() if stderr else ""
        
        logger.info(f"Raw prediction output: '{stdout_content}'")
        if stderr_content:
            logger.info(f"Stderr output: {stderr_content}")
        
        # Check for process error
        if process.returncode != 0:
            error_msg = stderr_content or "Unknown prediction error"
            logger.error(f"Prediction process failed with code {process.returncode}: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Try to directly convert to float
        try:
            # First check if the output is empty
            if not stdout_content.strip():
                raise ValueError("Empty output from prediction script")
                
            prediction = float(stdout_content)
            logger.info(f"Successfully parsed prediction: {prediction}")
            return {"prediction": prediction}
        except ValueError:
            # If direct conversion fails, try regex as fallback
            logger.warning(f"Could not directly convert output to float: '{stdout_content}'")
            clean_output = ''.join(c for c in stdout_content if c.isprintable())
            
            # Try to match any decimal number
            match = re.search(r'(\d+\.\d+)', clean_output)
            if match:
                prediction = float(match.group(1))
                logger.info(f"Extracted prediction via regex: {prediction}")
                return {"prediction": prediction}
                
            # No match found
            logger.error(f"No valid prediction value found in output")
            raise HTTPException(status_code=500, detail="Could not extract prediction from model output")
                
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Ensure we return a proper JSON error
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
            
        # Get clean output
        stdout_content = stdout.decode().strip()
        logger.info(f"Raw last close output: '{stdout_content}'")
        
        # Parse the result more robustly
        try:
            # Try to convert directly to float first
            try:
                last_close = float(stdout_content)
                logger.info(f"Last close for {bse_symbol}: {last_close}")
                return {"symbol": display_symbol, "lastClose": last_close}
            except ValueError:
                # If direct conversion fails, try regex
                decimal_match = re.search(r'(\d+\.\d+)', stdout_content)
                if decimal_match:
                    last_close = float(decimal_match.group(0))
                    logger.info(f"Last close for {bse_symbol}: {last_close}")
                    return {"symbol": display_symbol, "lastClose": last_close}
                    
                # Try integer pattern as fallback
                int_match = re.search(r'(\d+)', stdout_content)
                if int_match:
                    last_close = float(int_match.group(0))
                    logger.info(f"Last close for {bse_symbol}: {last_close}")
                    return {"symbol": display_symbol, "lastClose": last_close}
                    
                # If we reach here, couldn't parse the output
                raise ValueError(f"Could not parse last close value from: {stdout_content}")
        except Exception as e:
            logger.error(f"Error parsing last close: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse last close price: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fetching last close: {str(e)}")
        # Return a reasonable fallback if we can't get the actual value
        return {"symbol": input_symbol, "lastClose": None}

# Add a debug endpoint to test model behavior
@app.post('/debug-predict')
async def debug_predict(request: ModelRequest):
    """Get detailed diagnostic information about predictions"""
    try:
        input_symbol = request.symbol
        bse_symbol, display_symbol = map_symbol_to_bse(input_symbol)
        
        # Use debug flag for more information
        cmd = [
            sys.executable,
            'tft/predict.py',
            '--symbol', bse_symbol,
            '--display-symbol', display_symbol,
            '--debug',
        ]
        
        # Set up environment with explicit PYTHONIOENCODING
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run prediction process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        
        # Gather all debug info
        stdout_content = stdout.decode().strip() if stdout else ""
        stderr_content = stderr.decode().strip() if stderr else ""
        
        # Try to extract the prediction value
        prediction = None
        try:
            # Look for a decimal number in stdout
            match = re.search(r'(\d+\.\d+)', stdout_content)
            if match:
                prediction = float(match.group(1))
        except Exception:
            pass
        
        # Parse debug information
        debug_info = {}
        for line in stderr_content.split('\n'):
            if line.startswith('DEBUG:'):
                parts = line[6:].split(':')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    debug_info[key] = value
        
        # Return detailed information
        return {
            "prediction": prediction,
            "stdout": stdout_content,
            "stderr": stderr_content,
            "debug_info": debug_info
        }
            
    except Exception as e:
        logger.error(f"Debug prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a more detailed debug endpoint to inspect model metadata and data structures
@app.get('/debug-model/{symbol}')
async def debug_model(symbol: str):
    """Endpoint to inspect model data for debugging"""
    try:
        # Import needed functions
        from tft.utils.model_utils import load_model
        from tft.utils.data_loader import fetch_data
        
        # Clean symbol
        clean_symbol = symbol
        if symbol.endswith('.BO'):
            clean_symbol = symbol[:-3]
        
        # Load model info
        model, scaler_params, metadata = load_model(clean_symbol)
        
        # Check model structure
        model_info = {
            "model_exists": model is not None,
            "input_size": metadata.get("input_size") if metadata else "unknown",
            "hidden_size": metadata.get("hidden_size") if metadata else "unknown", 
            "scaler_params_type": str(type(scaler_params)),
            "mean_length": len(scaler_params.get("mean_", [])) if isinstance(scaler_params, dict) else 0,
            "scale_length": len(scaler_params.get("scale_", [])) if isinstance(scaler_params, dict) else 0,
            "metadata": metadata
        }
        
        # Get recent data shape
        bse_symbol, _ = map_symbol_to_bse(symbol)
        try:
            data = fetch_data(bse_symbol, days=30)
            data_info = {
                "data_found": data is not None,
                "data_length": len(data) if data is not None else 0,
                "columns": list(data.columns) if data is not None else [],
                "last_close": float(data["Close"].iloc[-1]) if data is not None else None
            }
        except Exception as data_err:
            data_info = {
                "data_error": str(data_err)
            }
        
        return {
            "symbol": symbol,
            "clean_symbol": clean_symbol,
            "bse_symbol": bse_symbol,
            "model_info": model_info,
            "data_info": data_info
        }
    except Exception as e:
        logger.error(f"Debug model error: {str(e)}")
        return {
            "error": str(e)
        }

# Replace the get_model_metadata endpoint for better diagnostics and handling
@app.get('/model-metadata/{symbol}')
async def get_model_metadata(symbol: str):
    """Get metadata about a trained model"""
    try:
        # Log the incoming request
        logger.info(f"Received metadata request for symbol: {symbol}")
        
        # Clean symbol more thoroughly
        clean_symbol = symbol
        if symbol.endswith('.BO'):
            clean_symbol = symbol[:-3]
        elif symbol.startswith('^'):
            clean_symbol = symbol[1:]
            
        # Get path to metadata file
        metadata_path = os.path.join(Config.MODEL_DIR, clean_symbol, 'metadata.json')
        
        # Debug info about paths
        logger.info(f"MODEL_DIR: {Config.MODEL_DIR}")
        logger.info(f"Looking for metadata at: {metadata_path}")
        logger.info(f"File exists: {os.path.exists(metadata_path)}")
        
        # Check if metadata exists
        if not os.path.exists(metadata_path):
            logger.info(f"Metadata file not found for {clean_symbol}")
            raise HTTPException(status_code=404, detail=f"No model found for {clean_symbol}")
        
        # Check file size for sanity
        file_size = os.path.getsize(metadata_path)
        logger.info(f"Metadata file size: {file_size} bytes")
        
        if file_size == 0:
            logger.warning(f"Metadata file for {clean_symbol} is empty")
            raise HTTPException(status_code=404, detail=f"Metadata file is empty")
        
        # Read metadata with better error handling
        try:
            with open(metadata_path, 'r') as f:
                file_content = f.read()
                logger.info(f"Raw metadata content length: {len(file_content)}")
                
                try:
                    metadata = json.loads(file_content)
                    
                    # Validate the metadata has required fields
                    if not isinstance(metadata, dict):
                        raise ValueError(f"Metadata is not a dictionary: {type(metadata)}")
                        
                    # Log the keys found in metadata
                    logger.info(f"Metadata keys: {', '.join(metadata.keys())}")
                    
                    return metadata
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parse error: {str(json_err)}, content: {file_content[:100]}...")
                    raise HTTPException(status_code=500, detail=f"Invalid metadata format: {str(json_err)}")
        except Exception as read_error:
            logger.error(f"Error reading metadata file: {str(read_error)}")
            raise HTTPException(status_code=500, detail=f"Error reading model metadata: {str(read_error)}")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static assets (CSS, JS, images) - AFTER API routes
@app.get('/assets/{filepath:path}')
async def serve_static(filepath: str):
    return FileResponse(f"../front-end/dist/assets/{filepath}")

# Finally, the catch-all route for SPA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse("../front-end/dist/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Explicitly set port to 8000