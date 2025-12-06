import requests
import json
import os
from collections import defaultdict

FOLDER_NAME = 'nn_directory'
os.makedirs(FOLDER_NAME, exist_ok=True)

def get_all_markets(series_ticker, inp_market):
    """Fetch all markets for a series, handling pagination"""
    all_markets = []
    cursor = None
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"

    while True:
        # Build URL with cursor if we have one
        url = f"{base_url}?series_ticker={series_ticker}&limit=100"
        if cursor:
            url += f"&cursor={cursor}"

        response = requests.get(url)
        data = response.json()
        
        # Add markets from this page
        all_markets.extend(data['markets'])

        # Check if there are more pages
        cursor = data.get('cursor')
        if not cursor:
            break

        print(f"Fetched {len(data['markets'])} markets, total: {len(all_markets)}")

    # Group markets by date and write separate files
    markets_by_date = defaultdict(list)
    for market in all_markets:
        # Extract date from event_ticker (e.g., "KXEARNINGSMENTIONNVDA-26MAR31" -> "26MAR31")
        event_ticker = market['event_ticker']
        date_part = event_ticker.split('-')[-1]
        markets_by_date[date_part].append(market)
    
    # Create subdirectory for this market
    market_folder = os.path.join(FOLDER_NAME, inp_market)
    os.makedirs(market_folder, exist_ok=True)
    
    # Write separate files for each date
    for date, markets in markets_by_date.items():
        file_path = os.path.join(market_folder, f'kalshi_response_{inp_market}_{date}.json')
        with open(file_path, "w") as f:
            json.dump({'cursor': '', 'markets': markets}, f, indent=4)
        print(f"Wrote {len(markets)} markets to {file_path}")

    return all_markets

def get_market_input(inp_market):
    markets = get_all_markets(f"KXEARNINGSMENTION{inp_market.upper()}", inp_market)
    print(f"Total markets found: {len(markets)}")

    # Initialize the output string
    x = {}
    
    # Read and display from the split files in the market subdirectory
    market_folder = os.path.join(FOLDER_NAME, inp_market)
    for filename in os.listdir(market_folder):
        if filename.startswith(f'kalshi_response_{inp_market}_'):
            file_path = os.path.join(market_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                x[filename] = {}
                for market in data['markets']:
                    x[filename][market['custom_strike']['Word']] = 1 if market['result'].upper() == 'YES' else 0
    return x
