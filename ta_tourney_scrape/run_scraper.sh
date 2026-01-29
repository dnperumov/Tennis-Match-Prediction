#!/bin/bash
# Complete scraping script for ATP 2025 matches

cd /Users/dennisperumov/Tennis-Match-Prediction/ta_tourney_scrape

echo "=========================================="
echo "ATP 2025 Match Scraper - Complete Pipeline"
echo "=========================================="
echo ""

# Step 1: Discover tournaments (already done, but can re-run)
echo "Step 1: Discovering tournaments..."
python3 -m ta_tourney_scrape discover
echo ""

# Step 2: Scrape tournament pages (with increased rate limiting)
echo "Step 2: Scraping tournament pages (this will take ~2 minutes with 2.5s rate limit)..."
python3 -m ta_tourney_scrape scrape_tournaments
echo ""

# Step 3: Check results
echo "Step 3: Analyzing scraped data..."
python3 << 'EOF'
import pandas as pd
import json

# Load tournaments
with open('data/intermediate/ta_tournaments_2025.json') as f:
    data = json.load(f)
    total_tournaments = data['count']

print(f"Tournaments discovered: {total_tournaments}")

# Load matches
try:
    df = pd.read_parquet('data/intermediate/ta_tournament_matches_2025.parquet')
    print(f"\n✅ Successfully scraped {len(df)} matches")
    print(f"   - With stats: {df['has_stats'].sum()}")
    print(f"   - Without stats: {(~df['has_stats']).sum()}")
    print(f"   - Tournaments covered: {df['tourney_name'].nunique()}/{total_tournaments}")
    
    # Check for tournaments needing profile fallback
    import os
    if os.path.exists('data/intermediate/tournaments_needing_profiles_2025.json'):
        with open('data/intermediate/tournaments_needing_profiles_2025.json') as f:
            need_profiles = json.load(f)
        print(f"\n⚠️  {len(need_profiles)} tournaments need profile fallback")
        for t in need_profiles[:5]:
            print(f"   - {t['tourney_name']}: {t['match_count']} matches")
    
    print("\n✅ Tournament scraping complete!")
    print("\nNext step: Run profile enrichment for tournaments lacking stats")
    print("Command: python3 -m ta_tourney_scrape scrape_profiles --year 2025")
    
except FileNotFoundError:
    print("\n❌ No matches file found. Scraping may have failed.")
    print("Check for errors above.")
EOF

echo ""
echo "=========================================="
echo "Scraping complete! Check output above."
echo "=========================================="

