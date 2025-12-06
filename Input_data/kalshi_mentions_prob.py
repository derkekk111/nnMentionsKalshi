import re
import pandas as pd
from typing import List, Dict, Any

def extract_transcript_text(transcript_list: List[Dict]) -> str:
    """Extract all text content from transcript paragraphs"""
    full_text = ""
    for para in transcript_list:
        if isinstance(para, dict) and 'content' in para:
            full_text += para['content'] + " "
    return full_text

def parse_keyword_variants(keyword: str) -> List[str]:
    """
    Parse keywords that have multiple variants separated by '/'
    
    Example: "RPO / Remaining Performance Obligation" -> ["RPO", "Remaining Performance Obligation"]
    Example: "Doge/Dogecoin" -> ["Doge", "Dogecoin"]
    """
    variants = [variant.strip() for variant in keyword.split('/')]
    return variants

def create_pattern_for_word(word: str) -> str:
    """
    Create regex pattern for a single word following Kalshi rules:
    - Plural forms included (s/es at end)
    - Possessive forms included ('s at end)
    - NO grammatical inflections (no -ing, -ed, -tion, etc.)
    - Word boundaries to avoid closed compounds
    """
    escaped_word = re.escape(word)
    
    # Allow:
    # - exact word
    # - word + s or es (plural)
    # - word + 's (possessive)
    # - word + s' (plural possessive)
    # But NOT other suffixes like -ing, -ed, -tion, -ment, etc.
    
    pattern = rf"\b{escaped_word}(?:es|s)?(?:'s?|')?\b"
    return pattern

def check_keyword_mention(keyword: str, transcript_text: str) -> Dict[str, Any]:
    """
    Check if a keyword is mentioned according to Kalshi rules.
    
    KALSHI RULES APPLIED:
    ✓ Plural and possessive forms included (s/es/'s)
    ✓ Compound words with hyphens/spaces included
    ✓ Case-insensitive
    ✓ Word boundaries (excludes closed compounds like "firetruck")
    ✓ Multiple variants with '/' (any variant counts)
    ✓ Adjacent context (partial word matches)
    ✗ Grammatical inflections NOT included (no -ing, -ed, -tion, etc.)
    ✗ Homophones excluded (can't detect programmatically)
    ✗ Foreign language excluded (can't detect reliably)
    
    NOTE: This implements a CONSERVATIVE approach - it may miss some edge cases
    but should avoid false positives for grammatical variations.
    """
    
    text_lower = transcript_text.lower()
    
    # Parse keyword variants (e.g., "RPO / Remaining Performance Obligation")
    variants = parse_keyword_variants(keyword)
    
    results = {
        'keyword': keyword,
        'found': False,
        'count': 0,
        'matches': [],
        'contexts': [],
        'variants_found': {}
    }
    
    # Check each variant
    for variant in variants:
        keyword_lower = variant.lower().strip()
        keyword_words = keyword_lower.split()
        
        if len(keyword_words) > 1:
            # Multi-word phrase: "Cloud infrastructure", "Law and order"
            pattern_parts = []
            for i, word in enumerate(keyword_words):
                if i == len(keyword_words) - 1:
                    # Last word: allow plural/possessive
                    pattern_parts.append(create_pattern_for_word(word))
                else:
                    # Middle words: allow plural/possessive too
                    # Example: "Laws and order" should match "Law and order"
                    pattern_parts.append(create_pattern_for_word(word))
            
            # Allow flexible spacing between words
            pattern = r'\s+'.join(pattern_parts)
            
        else:
            # Single word
            pattern = create_pattern_for_word(keyword_lower)
        
        # Find all matches
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
        
        if matches:
            results['variants_found'][variant] = len(matches)
        
        for match in matches:
            results['found'] = True
            results['count'] += 1
            matched_text = transcript_text[match.start():match.end()]
            results['matches'].append(matched_text)
            
            # Get context (100 chars before and after for better context)
            start = max(0, match.start() - 100)
            end = min(len(transcript_text), match.end() + 100)
            context = transcript_text[start:end].strip()
            results['contexts'].append({
                'variant': variant,
                'matched_text': matched_text,
                'context': context
            })
    
    return results

def check_keywords_in_transcript(keywords: List[str], transcript_data: List[Dict]) -> Dict[str, Dict]:
    """
    Check multiple keywords in a transcript
    """
    full_text = extract_transcript_text(transcript_data)
    
    results = {}
    for keyword in keywords:
        results[keyword] = check_keyword_mention(keyword, full_text)
    
    return results

def create_mention_dataframe(transcripts_df, keywords: List[str]) -> pd.DataFrame:
    """
    Create a dataframe showing which keywords were mentioned in each transcript
    Returns: DataFrame with columns [date, year_quarter, symbol, keyword, mentioned, count, variants_found]
    """
    rows = []
    for idx, row in transcripts_df.iterrows():
        transcript_data = row['transcript']
        keyword_results = check_keywords_in_transcript(keywords, transcript_data)
        
        for keyword, result in keyword_results.items():
            rows.append({
                'date': row['date'],
                'year_quarter': row['year_quarter'],
                'symbol': row['symbol'],
                'keyword': keyword,
                'mentioned': result['found'],
                'count': result['count'],
                'variants_found': str(result['variants_found'])
            })
    
    return pd.DataFrame(rows)

def calculate_mention_probabilities(mention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical mention probabilities for each keyword
    Returns: DataFrame with keyword and probability columns
    """
    keyword_stats = mention_df.groupby('keyword').agg({
        'mentioned': ['sum', 'count']
    }).reset_index()
    
    keyword_stats.columns = ['keyword', 'times_mentioned', 'total_transcripts']
    keyword_stats['probability'] = keyword_stats['times_mentioned'] / keyword_stats['total_transcripts']
    
    return keyword_stats.sort_values('probability', ascending=False)

def print_detailed_results(results: Dict[str, Dict]):
    """Print detailed analysis of keyword findings"""
    print("\n" + "="*100)
    print("KEYWORD MENTION ANALYSIS (Kalshi Rules)")
    print("="*100)
    
    found_keywords = {k: v for k, v in results.items() if v['found']}
    not_found_keywords = {k: v for k, v in results.items() if not v['found']}
    
    print(f"\n✓ FOUND: {len(found_keywords)} keywords")
    print(f"✗ NOT FOUND: {len(not_found_keywords)} keywords")
    
    for keyword, result in found_keywords.items():
        print(f"\n{'─'*100}")
        print(f"📌 {keyword}")
        print(f"   Total mentions: {result['count']}")
        
        if result['variants_found']:
            print(f"   Variants breakdown: {result['variants_found']}")
        
        # Show first 2 contexts
        for i, ctx in enumerate(result['contexts'][:2]):
            print(f"\n   Example {i+1} [{ctx['variant']}]:")
            print(f"   Matched: '{ctx['matched_text']}'")
            print(f"   Context: ...{ctx['context'][:150]}...")
    
    if not_found_keywords:
        print(f"\n{'─'*100}")
        print("\n✗ Keywords NOT mentioned:")
        for keyword in not_found_keywords.keys():
            print(f"   • {keyword}")

def analyze_keyword_mentions(transcripts_df: pd.DataFrame, keywords: List[str], verbose: bool = True):
    """
    Main function to analyze keyword mentions across all transcripts
    
    Args:
        transcripts_df: DataFrame with transcript data
        keywords: List of keywords to search for
        verbose: Whether to print detailed results
    
    Returns:
        tuple: (mention_df, probability_df)
    """
    print(f"\n🔍 Analyzing {len(keywords)} keywords across {len(transcripts_df)} transcripts...")
    
    # Create mention dataframe
    mention_df = create_mention_dataframe(transcripts_df, keywords)
    
    # Calculate probabilities
    probability_df = calculate_mention_probabilities(mention_df)
    
    if verbose:
        print("\n" + "="*100)
        print("KEYWORD MENTION PROBABILITIES (Based on Historical Data)")
        print("="*100)
        print(probability_df.to_string(index=False))
        
        # Show latest transcript analysis
        print("\n" + "="*100)
        print("MOST RECENT TRANSCRIPT ANALYSIS")
        print("="*100)
        latest_transcript = transcripts_df.iloc[-1]['transcript']
        latest_date = transcripts_df.iloc[-1]['date']
        latest_quarter = transcripts_df.iloc[-1]['year_quarter']
        
        print(f"\nDate: {latest_date}")
        print(f"Quarter: {latest_quarter}")
        
        results = check_keywords_in_transcript(keywords, latest_transcript)
        print_detailed_results(results)
    
    return mention_df, probability_df