from kalshi_api_tester import get_market_input
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from defeat_data_api import Ticker
import pandas as pd
from kalshi_mentions_prob import analyze_keyword_mentions

inp_market = input("Enter earnings mentions series ticker (stock ticker): ")
# Create a ticker object
ticker = Ticker(inp_market)
# Get the transcripts object
transcripts = ticker.earning_call_transcripts()
# Get the transcripts list as a dataframe
transcripts_df = transcripts.get_transcripts_list()

# Create the missing columns that the function expects
transcripts_df['date'] = pd.to_datetime(transcripts_df['report_date'])
transcripts_df['year_quarter'] = (
    transcripts_df['fiscal_year'].astype(str) + 
    '-Q' + 
    transcripts_df['fiscal_quarter'].astype(str)
)
transcripts_df['transcript'] = transcripts_df['transcripts']

# Sort by report_date
transcripts_df = transcripts_df.sort_values('date')

print(transcripts_df.head())

# Get market data and keywords
k = get_market_input(inp_market)
inp_date = input("Enter date (e.g., 26MAR31): ")

# Prepare keywords
keywords = []
for keyword in k[f'kalshi_response_{inp_market}_{inp_date}.json']:
    keywords.append(keyword)
print(f"\nKeywords to analyze: {keywords}")

# Convert inp_date to datetime for filtering
from datetime import datetime
year = "20" + inp_date[:2]
month_map = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 
             'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
             'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
month = month_map[inp_date[2:5]]
day = inp_date[5:7]
target_date = pd.to_datetime(f"{year}-{month}-{day}")

# Filter to only keep transcripts from the last 2 years before target date
two_years_before = target_date - pd.DateOffset(years=2)
transcripts_df = transcripts_df[transcripts_df['date'] >= two_years_before]
transcripts_df = transcripts_df[transcripts_df['date'] < target_date]

print(f"\nFiltered to {len(transcripts_df)} transcripts from last 2 years")

# Analyze keyword mentions using Kalshi rules
mention_df, probability_df = analyze_keyword_mentions(transcripts_df, keywords, verbose=True)

# Save results
print("\n" + "="*100)
print("SAVING RESULTS")
print("="*100)
mention_df.to_csv(f'nn_directory/{inp_market}/mention_analysis_{inp_date}.csv', index=False)
probability_df.to_csv(f'nn_directory/{inp_market}/probabilities_{inp_date}.csv', index=False)
print(f" Saved mention analysis to nn_directory/{inp_market}/mention_analysis_{inp_date}.csv")
print(f" Saved probabilities to nn_directory/{inp_market}/probabilities_{inp_date}.csv")
