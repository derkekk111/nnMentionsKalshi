import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
input_data_dir = os.path.join(parent_dir, 'input_data')
sys.path.insert(0, parent_dir)
sys.path.insert(0, input_data_dir)

from kalshi_api_tester import get_market_input
from kalshi_mentions_prob import analyze_keyword_mentions
from LLM_Likelihood_DefeatBeta import get_keywords_probabilities_batch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from defeatbeta_api.data.ticker import Ticker

class KeywordPredictor(nn.Module):
    def __init__(self, n_features):
        super(KeywordPredictor, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def create_features(keyword, past_mentions, llm_prob):
    """Simple 3-feature vector: past mentions, LLM prob, keyword length"""
    return np.array([
        float(past_mentions),
        float(llm_prob),
        len(keyword) / 20.0
    ])

def prepare_training_data(ticker_symbol, dates_to_analyze):
    ticker = Ticker(ticker_symbol)
    X_list, y_list = [], []
    
    for date_str in dates_to_analyze:
        print(f"\nProcessing {date_str}...")
        
        try:
            # Get keywords
            k = get_market_input(ticker_symbol)
            key = f'kalshi_response_{ticker_symbol}_{date_str}.json'
            if key not in k:
                print(f"  Skipping - no data")
                continue
            keywords = list(k[key].keys())
            
            # Parse date
            year = "20" + date_str[:2]
            month_map = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
                        'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
            month, day = month_map[date_str[2:5]], date_str[5:7]
            target_date = pd.to_datetime(f"{year}-{month}-{day}")
            
            # Get transcripts
            transcripts_df = ticker.earning_call_transcripts().get_transcripts_list()
            transcripts_df['date'] = pd.to_datetime(transcripts_df['report_date'])
            transcripts_df['transcript'] = transcripts_df['transcripts']
            transcripts_df['year_quarter'] = (transcripts_df['fiscal_year'].astype(str) + 
                                             '-Q' + transcripts_df['fiscal_quarter'].astype(str))
            
            # Filter last 8 transcripts before target date
            two_years_before = target_date - pd.DateOffset(years=2)
            transcripts_df = transcripts_df[
                (transcripts_df['date'] >= two_years_before) & 
                (transcripts_df['date'] < target_date)
            ].tail(8)
            
            # Analyze mentions
            mention_df, _ = analyze_keyword_mentions(transcripts_df, keywords, verbose=False)
            
            # Get LLM probabilities
            keyword_batch_data = get_keywords_probabilities_batch(
                keywords=keywords, ticker=ticker, 
                transcripts_df=transcripts_df, context_date=target_date
            )
            
            # Create features for each keyword
            for keyword in keywords:
                # Get past mentions - handle different return formats
                past_mentions = 0
                if mention_df is not None and not mention_df.empty:
                    if 'Keyword' in mention_df.columns:
                        matches = mention_df[mention_df['Keyword'] == keyword]
                        if not matches.empty and 'Total_Mentions' in mention_df.columns:
                            past_mentions = matches['Total_Mentions'].values[0]
                
                # Get LLM probability
                llm_prob = 0.0
                if keyword_batch_data and isinstance(keyword_batch_data, dict) and keyword in keyword_batch_data:
                    llm_prob = keyword_batch_data[keyword].get('probability', 0.0)
                
                features = create_features(keyword, past_mentions, llm_prob)
                
                # Mock outcome (replace with actual Kalshi data)
                outcome = np.random.choice([0.0, 1.0])
                
                X_list.append(features)
                y_list.append(outcome)
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return np.array(X_list), np.array(y_list)

def train_model(X_train, y_train, X_test, y_test, epochs=1000, lr=0.01):
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)
    
    model = KeywordPredictor(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Train
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Evaluate
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                train_acc = (y_pred.round() == y_train).float().mean().item()
                y_test_pred = model(X_test)
                test_loss = criterion(y_test_pred, y_test).item()
                test_acc = (y_test_pred.round() == y_test).float().mean().item()
                print(f'Epoch {epoch+1}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}')
    
    return model

def predict_keywords(model, scaler, ticker_symbol, date_str):
    print(f"\n{'='*80}\nPREDICTING FOR {ticker_symbol} on {date_str}\n{'='*80}")
    
    ticker = Ticker(ticker_symbol)
    
    # Get keywords and date
    k = get_market_input(ticker_symbol)
    keywords = list(k[f'kalshi_response_{ticker_symbol}_{date_str}.json'].keys())
    
    year = "20" + date_str[:2]
    month_map = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
                'JUL':'07','AUG':'08','SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
    month, day = month_map[date_str[2:5]], date_str[5:7]
    target_date = pd.to_datetime(f"{year}-{month}-{day}")
    
    # Get transcripts
    transcripts_df = ticker.earning_call_transcripts().get_transcripts_list()
    transcripts_df['date'] = pd.to_datetime(transcripts_df['report_date'])
    transcripts_df['transcript'] = transcripts_df['transcripts']
    transcripts_df['year_quarter'] = (transcripts_df['fiscal_year'].astype(str) + 
                                     '-Q' + transcripts_df['fiscal_quarter'].astype(str))
    
    two_years_before = target_date - pd.DateOffset(years=2)
    transcripts_df = transcripts_df[
        (transcripts_df['date'] >= two_years_before) & 
        (transcripts_df['date'] < target_date)
    ].tail(8)
    
    # Analyze
    mention_df, _ = analyze_keyword_mentions(transcripts_df, keywords, verbose=False)
    keyword_batch_data = get_keywords_probabilities_batch(
        keywords=keywords, ticker=ticker, 
        transcripts_df=transcripts_df, context_date=target_date
    )
    
    # Predict
    results = []
    for keyword in keywords:
        # Get past mentions - handle different return formats
        past_mentions = 0
        if mention_df is not None and not mention_df.empty:
            if 'Keyword' in mention_df.columns:
                matches = mention_df[mention_df['Keyword'] == keyword]
                if not matches.empty and 'Total_Mentions' in mention_df.columns:
                    past_mentions = matches['Total_Mentions'].values[0]
        
        # Get LLM probability
        llm_prob = 0.0
        if keyword_batch_data and isinstance(keyword_batch_data, dict) and keyword in keyword_batch_data:
            llm_prob = keyword_batch_data[keyword].get('probability', 0.0)
        
        features = create_features(keyword, past_mentions, llm_prob)
        features_scaled = scaler.transform([features])
        features_tensor = torch.from_numpy(features_scaled.astype(np.float32))
        
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        results.append({
            'Keyword': keyword,
            'NN_Probability': prediction,
            'LLM_Probability': llm_prob,
            'Past_Mentions': past_mentions
        })
    
    results_df = pd.DataFrame(results).sort_values('NN_Probability', ascending=False)
    print(results_df)
    
    os.makedirs(f'nn_directory/{ticker_symbol}', exist_ok=True)
    results_df.to_csv(f'nn_directory/{ticker_symbol}/predictions_{date_str}.csv', index=False)
    
    return results_df

# Main
if __name__ == "__main__":
    ticker_symbol = input("Enter ticker symbol: ")
    
    # Get available dates
    ticker = Ticker(ticker_symbol)
    transcripts_df = ticker.earning_call_transcripts().get_transcripts_list()
    transcripts_df['date'] = pd.to_datetime(transcripts_df['report_date'])
    transcripts_df = transcripts_df.sort_values('date', ascending=False)
    
    print(f"\nAvailable earnings dates:")
    for i, row in transcripts_df.head(10).iterrows():
        date = row['date']
        date_str = f"{str(date.year)[2:]}{date.strftime('%b').upper()}{date.strftime('%d')}"
        print(f"  {date.strftime('%Y-%m-%d')} -> {date_str}")
    
    # Get training dates
    train_dates = input("\nEnter training dates (comma-separated, e.g., 25OCT30,25JUL31): ").strip().split(',')
    train_dates = [d.strip() for d in train_dates]
    
    # Prepare and train
    print(f"\n{'='*80}\nPREPARING TRAINING DATA\n{'='*80}")
    X, y = prepare_training_data(ticker_symbol, train_dates)
    
    print(f"\nTotal samples: {len(X)}")
    
    if len(X) == 0:
        print("ERROR: No training samples collected. Check if dates have valid data.")
        exit(1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"\n{'='*80}\nTRAINING MODEL\n{'='*80}")
    model = train_model(X_train, y_train, X_test, y_test, epochs=1000, lr=0.01)
    
    # Predict
    pred_date = input("\nEnter date for prediction: ").strip()
    predictions = predict_keywords(model, scaler, ticker_symbol, pred_date)
    
    # Save model
    torch.save(model.state_dict(), f'nn_directory/{ticker_symbol}/model.pth')
    print(f"\nModel saved to nn_directory/{ticker_symbol}/model.pth")