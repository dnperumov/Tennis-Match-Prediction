#!/usr/bin/env python3
"""
Debug script to inspect TennisAbstract page structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ta2025.utils import fetch_page, get_session
from bs4 import BeautifulSoup
import re

URL = "https://www.tennisabstract.com/current/2026ATPHongKong.html"

print("=" * 60)
print("Debugging TennisAbstract Page Structure")
print("=" * 60)
print(f"URL: {URL}\n")

html = fetch_page(URL)
if not html:
    print("❌ Failed to fetch page")
    sys.exit(1)

print("✅ Page fetched successfully\n")

# Check for "Completed Matches" text
print("Searching for 'Completed Matches' text...")
completed_matches_found = False
for element in html.find_all(string=re.compile(r'Completed', re.I)):
    print(f"  Found: {element.strip()[:100]}")
    completed_matches_found = True
    parent = element.find_parent()
    if parent:
        print(f"  Parent tag: {parent.name}")
        print(f"  Parent classes: {parent.get('class', [])}")

if not completed_matches_found:
    print("  ❌ 'Completed Matches' text not found")

print("\n" + "=" * 60)
print("All tables on page:")
print("=" * 60)

tables = html.find_all('table')
print(f"Found {len(tables)} table(s)\n")

for i, table in enumerate(tables, 1):
    print(f"Table {i}:")
    rows = table.find_all('tr')
    print(f"  Rows: {len(rows)}")
    
    # Show first few rows
    for j, row in enumerate(rows[:5], 1):
        cells = row.find_all(['td', 'th'])
        row_text = row.get_text()[:100].replace('\n', ' ')
        print(f"  Row {j}: {len(cells)} cells - {row_text}")
    
    if len(rows) > 5:
        print(f"  ... ({len(rows) - 5} more rows)")
    print()

print("=" * 60)
print("JavaScript-embedded tables:")
print("=" * 60)

for script in html.find_all('script'):
    script_text = script.string
    if not script_text:
        continue
    
    # Look for table HTML in JavaScript
    table_match = re.search(r"var\s+proj\d+\s*=\s*['\"](<table[^>]*>.*?</table>)['\"]", script_text, re.DOTALL | re.IGNORECASE)
    if table_match:
        table_html = table_match.group(1)
        print(f"Found embedded table in script (first 500 chars):")
        print(table_html[:500])
        print("\nTrying to parse...")
        
        # Try to parse it
        table_html_clean = table_html.replace('&nbsp;', ' ').replace('&amp;', '&').replace('\\"', '"')
        try:
            embedded_table = BeautifulSoup(table_html_clean, 'lxml')
            table = embedded_table.find('table')
            if table:
                rows = table.find_all('tr')
                print(f"✅ Parsed table with {len(rows)} rows")
                for i, row in enumerate(rows[:5], 1):
                    print(f"  Row {i}: {row.get_text()[:100]}")
                break
        except Exception as e:
            print(f"  ❌ Error parsing: {e}")

print("\n" + "=" * 60)
print("Looking for match patterns in page text...")
print("=" * 60)

page_text = html.get_text()
def_pattern = re.compile(r'([A-Z][a-zA-Z\s\-\.]+?)\s+(?:def\.?|d\.?|defeated)\s+([A-Z][a-zA-Z\s\-\.]+?)', re.IGNORECASE)
matches = def_pattern.findall(page_text[:5000])  # First 5000 chars

if matches:
    print(f"Found {len(matches)} potential match patterns:\n")
    for i, (winner, loser) in enumerate(matches[:10], 1):
        print(f"  {i}. {winner.strip()} def. {loser.strip()}")
    if len(matches) > 10:
        print(f"  ... ({len(matches) - 10} more)")
else:
    print("  ❌ No match patterns found")

print("\n" + "=" * 60)
print("Page title and headers:")
print("=" * 60)

title = html.find('title')
if title:
    print(f"Title: {title.get_text()}")

for header in html.find_all(['h1', 'h2', 'h3', 'h4']):
    text = header.get_text().strip()
    if text:
        print(f"{header.name}: {text[:80]}")

