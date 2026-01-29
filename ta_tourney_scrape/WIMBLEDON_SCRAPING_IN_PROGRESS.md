# Wimbledon 2025 Scraping - IN PROGRESS

## Status: RUNNING 🚀

The scraper is currently extracting Wimbledon 2025 matches from player profiles using the TennisAbstract "classic" player pages.

### What's Happening

- **Source**: `player-classic.cgi` pages on TennisAbstract
- **Method**: Browser automation (Playwright) to handle JavaScript-loaded content
- **Players to Process**: 128 Wimbledon 2025 participants
- **Estimated Time**: ~5-7 minutes total
- **Current Progress**: Check `wimbledon_scrape_final.log` for real-time updates

### Why This Works

The regular player pages (`player.cgi`) don't have Wimbledon 2025 data, but the **classic player pages** (`player-classic.cgi`) do! 

Key discovery: The classic pages use **non-breaking spaces** (Unicode 0xa0) instead of regular spaces in match results, which is why the initial scraper didn't work. This has been fixed by normalizing the text before parsing.

### What Will Be Scraped

For each of the 128 players:
- Match date
- Tournament name (Wimbledon)
- Surface (Grass)
- Round (R128, R64, R32, R16, QF, SF, F)
- Winner and loser names
- Score
- Optional: Match statistics if available

### Expected Output

1. **`data/atp_matches_2025_wimbledon_only.csv`**
   - Raw Wimbledon 2025 matches (before deduplication)
   - Estimated: ~127 unique matches after deduplication

2. **`data/atp_matches_2025_WITH_WIMBLEDON.csv`**
   - Combined dataset: existing 4,667 matches + Wimbledon 2025
   - Estimated total: ~4,794 matches from 83 tournaments
   - Grand Slam coverage: 3/4 (Australian Open, Roland Garros, Wimbledon)

### Progress Check

To monitor progress in real-time:

```bash
tail -f /Users/dennisperumov/Tennis-Match-Prediction/ta_tourney_scrape/wimbledon_scrape_final.log
```

Or check the terminal file:

```bash
tail -f /Users/dennisperumov/.cursor/projects/Users-dennisperumov-Tennis-Match-Prediction/terminals/13.txt
```

### Next Steps After Completion

1. **Verify Wimbledon matches count** (~127 expected)
2. **Merge with existing dataset**
3. **Check for duplicates**
4. **Update final dataset statistics**
5. **Optional**: Scrape US Open 2025 using the same method

---

**Last Updated**: 2026-01-14 (Scraper started)

