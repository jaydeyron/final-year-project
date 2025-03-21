# This file contains shared data structures used by multiple modules

# Stock data with BSE symbols
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
    # ... other stocks ...
    { 
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS",
        "tradingViewSymbol": "BSE:TCS",
        "bseSymbol": "TCS.BO",
        "startDate": "2004-08-25"
    }
]

def get_symbol_info(symbol_to_find, field_name='symbol'):
    """
    Find stock information by symbol
    
    Args:
        symbol_to_find: Symbol to search for
        field_name: Field to match against (symbol, bseSymbol, etc.)
    
    Returns:
        Dictionary with stock information or None if not found
    """
    for stock in NIFTY_STOCKS:
        if field_name in stock and stock[field_name] == symbol_to_find:
            return stock
    return None
