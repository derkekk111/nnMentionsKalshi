from openai import OpenAI
import os
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import defeatbeta_api
from defeatbeta_api.data.ticker import Ticker

load_dotenv()

# Initialize Jetstream2 LLM client
JETSTREAM_BASE_URL = os.getenv('JETSTREAM_BASE_URL', 'https://llm.jetstream-cloud.org/api')
JETSTREAM_API_KEY = os.getenv('JETSTREAM_API_KEY')
PRIMARY_MODEL = os.getenv('PRIMARY_MODEL', 'gpt-oss-120b')
FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'llama-4-scout')

jetstream_client = OpenAI(
    base_url=JETSTREAM_BASE_URL,
    api_key=JETSTREAM_API_KEY
)

# System prompt - Updated to include news
system_prompt = """
You analyze company documents with a strict focus on **earnings call transcripts for historical mentions**, while allowing **investor or special presentations** for management context, and considering **recent news articles** for current relevance.

Users will provide text of past earnings call transcripts, investor presentations, and recent news articles. Extract information directly from those files.

### Instructions (optimized for API efficiency)

- Input: an **array of keywords**.
- For each keyword:
1. Apply the Kalshi word-matching criteria (below) independently.
2. Search only **earnings call transcripts** for historical counts and dates.
3. Use **investor/special presentations** only for management or strategy context.
4. Include **news articles** to assess recent relevance and trends.
5. Each earnings call counts once per keyword (deduplicate mentions).
6. Extract short quotes (1–2 sentences) from any relevant document.
7. Summarize the last five earnings calls (chronological) with date and presence (Yes/No).
8. Compute probability (0–100) for future mention based on:
- Historical frequency
- Recency weighting
- Trend direction
- Strategic context
- News coverage intensity
9. Assign confidence level (High/Medium/Low) based on data availability.
10. Provide a two-sentence rationale grounded in frequency, recency, news, and context.

### Kalshi Word-Matching Criteria

**Include:** plural/possessive forms, compound/hyphenated words, ordinal numbers, transliterations, homonyms, homographs. 

**Exclude:** tense/grammatical inflections, closed compounds, other languages, homophones, synonyms.

### Additional Rules

- Only count earnings call mentions for statistics.
- Factor in news coverage when estimating probability.
- Treat ticker/name changes as the same entity if continuity is clear.
- If data is sparse, lower confidence accordingly.

### Output Format (strict JSON)

{
"results": [
{
"keyword": "the keyword analyzed",
"historical_mentions": "number of times mentioned in earnings calls",
"probability": "0-100 probability of future mention",
"confidence_level": "High/Medium/Low",
"rationale": "Two-sentence explanation for probability",
"summary_table": [
{"date": "YYYY-MM-DD", "presence": "Yes/No"}
],
"earnings_call_mentions": [
{"date": "YYYY-MM-DD", "transcript": "relevant quote"}
],
"quotes": [
{"date": "YYYY-MM-DD", "quote": "exact quote with context"}
]
}
]
}

### Output Rules

- Each keyword must have its own entry in the "results" array.
- No prose or commentary outside JSON.
- Start output with '{' and end with '}'.
"""

def get_keywords_probabilities_batch(keywords, ticker, transcripts_df, context_date, return_full_response=False):
    """
    Analyzes the probability of multiple keywords being mentioned in the next earnings call.
    Now includes news articles from Defeat-Beta API before the context date and processes all keywords in a single API call.
    Args:
    keywords (list): List of keywords to analyze
    ticker (str): Company ticker symbol
    transcripts_df (pd.DataFrame): DataFrame with columns ['year_quarter', 'date', 'transcript']
    context_date (str): Date string (YYYY-MM-DD) to filter news articles up to this date
    return_full_response (bool): If True, returns (probabilities, full_response), otherwise just probabilities
    Returns:
    dict or tuple: Dictionary mapping keywords to their probabilities, or (dict, str) if return_full_response=True
    """
    # Pre-process keywords to handle multi-keyword entries (e.g., "Regulation / Regulator / Regulatory")
    processed_keywords = []
    keyword_mapping = {}  # Maps processed keyword back to original keyword
    
    for keyword in keywords:
        if ' / ' in keyword:
            # Split multi-keyword entries
            split_keywords = [k.strip() for k in keyword.split(' / ')]
            processed_keywords.extend(split_keywords)
            # Map each split keyword back to the original for result mapping
            for split_kw in split_keywords:
                keyword_mapping[split_kw] = keyword
        else:
            processed_keywords.append(keyword)
            keyword_mapping[keyword] = keyword
    
    # Sort by date
    transcripts_df = transcripts_df.sort_values('report_date')
    # Build content string with transcripts
    content_str = f"""Analyze the following keywords for company {ticker}: {', '.join(processed_keywords)}\n\n"""
    content_str += "=== EARNINGS CALL TRANSCRIPTS ===\n\n"
    for idx, row in transcripts_df.iterrows():
        content_str += f"""Transcript ({row['year_quarter']}, Date: {row['date']}):
{row['transcript']}
---
"""
    # Fetch news from Defeat-Beta API
    try:
        ticker_obj = Ticker(ticker)
        news_obj = ticker_obj.news()
        news_df = news_obj.get_news_list()
        # Filter news by context_date and take only articles from 3 weeks and two days to two days
        if news_df is not None and not news_df.empty:
            # Convert context_date string to datetime for comparison
            context_datetime = pd.to_datetime(context_date) - timedelta(days=2)
            start_dt = context_datetime - timedelta(days=23)
            # Convert report_date to datetime
            news_df['report_date'] = pd.to_datetime(news_df['report_date'])
            # Filter news: only articles on or before context_date
            filtered_news = news_df[news_df['report_date'] <= context_datetime]
            filtered_news = news_df[start_dt <= news_df['report_date']]
            print(filtered_news)
            # Sort by date descending (most recent first) and take top 50
            filtered_news = filtered_news.sort_values('report_date', ascending=False).head(25)
            # Add news to context
            if not filtered_news.empty:
                content_str += "\n\n=== RECENT NEWS ARTICLES ===\n\n"
                # Iterate through filtered news articles
                for idx, article in filtered_news.iterrows():
                    # Extract relevant fields from the news article
                    title = article.get('title', '')
                    content = article.get('news', '')
                    date = article.get('report_date', '')
                    source = article.get('publisher', '')
                    if title or content:
                        content_str += f"""Date: {date}
Source: {source}
Title: {title}
Content: {content[:1000]}
---
"""
    except Exception as e:
        print(f"Warning: Could not fetch news for {ticker}: {e}")
        # Continue without news if it fails
    # Get likelihood estimate from LLM using Jetstream2
    try:
        response = jetstream_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_str}
            ],
            temperature=0.1
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error with primary model {PRIMARY_MODEL}, trying fallback {FALLBACK_MODEL}: {e}")
        try:
            response = jetstream_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_str}
                ],
                temperature=0.1
            )
            response_text = response.choices[0].message.content
        except Exception as fallback_error:
            print(f"Error with fallback model: {fallback_error}")
            return None
    # Parse the JSON response
    try:
        response_text = response_text.strip()
        # Strip markdown formatting if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()
        result = json.loads(response_text)
        # Extract probabilities for each keyword and map back to original keywords
        keyword_probabilities = {}
        results = result.get('results', [])
        print(f"Debug: Found {len(results)} results in JSON response")
        for keyword_result in results:
            processed_keyword = keyword_result.get('keyword')
            probability = keyword_result.get('probability')
            if processed_keyword and probability is not None:
                # Map back to original keyword
                original_keyword = keyword_mapping.get(processed_keyword, processed_keyword)
                keyword_probabilities[original_keyword] = int(probability)
        print(f"Debug: Extracted probabilities for {len(keyword_probabilities)} keywords")
        if return_full_response:
            return keyword_probabilities, response_text
        else:
            return keyword_probabilities
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing response for {ticker}: {e}")
        print(f"Raw response:\n{response_text}")
        if return_full_response:
            return None, response_text
        else:
            return None


def get_keyword_probability(keyword, ticker, transcripts_df):
    """
    Analyzes the probability of a keyword being mentioned in the next earnings call.
    Now includes news articles from Defeat-Beta API.

    Args:
        keyword (str): The keyword to analyze
        ticker (str): Company ticker symbol
        transcripts_df (pd.DataFrame): DataFrame with columns ['year_quarter', 'date', 'transcript']

    Returns:
        int: Probability (0-100) that the keyword will be mentioned in the next earnings call
    """
    # Sort by date
    transcripts_df = transcripts_df.sort_values('date')

    # Build content string with transcripts
    content_str = f"""Analyze the keyword "{keyword}" for company {ticker}.\n\n"""
    content_str += "=== EARNINGS CALL TRANSCRIPTS ===\n\n"

    for idx, row in transcripts_df.iterrows():
        content_str += f"""Transcript ({row['year_quarter']}, Date: {row['date']}):
{row['transcript']}
---
"""

    # Fetch news from Defeat-Beta API
    try:
        ticker_obj = Ticker(ticker)
        news_obj = ticker_obj.news()
        news_df = news_obj.get_news_list()

        # Add news to context (check if DataFrame is not empty)
        if news_df is not None and not news_df.empty:
            content_str += "\n\n=== RECENT NEWS ARTICLES ===\n\n"

            # Iterate through news articles (limit to 50)
            for idx, article in news_df.head(50).iterrows():
                # Extract relevant fields from the news article
                title = article.get('title', '')
                content = article.get('content', '') or article.get('description', '')
                date = article.get('published_date', '') or article.get('date', '')
                source = article.get('source', '')

                if title or content:
                    content_str += f"""Date: {date}
Source: {source}
Title: {title}
Content: {content[:1000]}
---
"""

    except Exception as e:
        print(f"Warning: Could not fetch news for {ticker}: {e}")
        # Continue without news if it fails

    # Get likelihood estimate from LLM using Jetstream2
    try:
        response = jetstream_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_str}
            ],
            temperature=0.1
        )
        response_text = response.choices[0].message.content
        return response_text

    except Exception as e:
        print(f"Error with primary model {PRIMARY_MODEL}, trying fallback {FALLBACK_MODEL}: {e}")
        try:
            response = jetstream_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_str}
                ],
                temperature=0.1
            )
            response_text = response.choices[0].message.content
        except Exception as fallback_error:
            print(f"Error with fallback model: {fallback_error}")
            return None

    # Parse the JSON response
    try:
        response_text = response_text.strip()

        # Strip markdown formatting if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()

        result = json.loads(response_text)
        probability = int(result['probability'])

        return probability

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing response for {ticker} - {keyword}: {e}")
        print(f"Raw response:\n{response_text}")
        return None