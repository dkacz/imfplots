#!/usr/bin/env python3
"""
Manual verification of country order by analyzing the plot image.
Based on visual inspection of 9781484392935_f0013-01.jpg
"""

import json

# Reading the country codes from left to right from the plot image
# The text is small but readable when zoomed in
manually_read_countries = [
    "USA", "MEX", "JPN", "GBR", "FRA", "ITA", "DEU", "CAN",
    "AUS", "ESP", "KOR", "NLD", "POL", "BEL", "CHE", "SWE",
    "AUT", "NOR", "DNK", "FIN", "ISR", "PRT", "CZE", "GRC",
    "NZL", "SVK", "CHL", "HUN", "IRL", "SVN", "LTU", "LUX",
    "LVA", "EST"
]

# Country order from current metadata
metadata_countries = [
    "USA", "MEX", "JPN", "GBR", "FRA", "ITA", "DEU", "CAN",
    "AUS", "ESP", "KOR", "NLD", "BEL", "CHE", "SWE", "AUT",
    "POL", "NOR", "DNK", "FIN", "ISR", "PRT", "CZE", "GRC",
    "NZL", "SVK", "CHL", "HUN", "IRL", "SVN", "LTU", "LUX",
    "LVA", "EST"
]

print("Comparing country orders:")
print("=" * 70)

print("\n1. Manually read from plot image:")
for i, country in enumerate(manually_read_countries):
    print(f"{i:2d}. {country}", end="  ")
    if (i + 1) % 8 == 0:
        print()

print("\n\n2. Current metadata order:")
for i, country in enumerate(metadata_countries):
    print(f"{i:2d}. {country}", end="  ")
    if (i + 1) % 8 == 0:
        print()

print("\n\n" + "=" * 70)
print("Differences:")

# Find differences
for i in range(max(len(manually_read_countries), len(metadata_countries))):
    manual = manually_read_countries[i] if i < len(manually_read_countries) else "---"
    meta = metadata_countries[i] if i < len(metadata_countries) else "---"

    if manual != meta:
        print(f"Position {i:2d}: Manual='{manual}' vs Metadata='{meta}' ❌")

# Check Poland specifically
print("\n" + "=" * 70)
print("Poland (POL) position:")

if "POL" in manually_read_countries:
    manual_pol_idx = manually_read_countries.index("POL")
    print(f"  Manual read: Position {manual_pol_idx} (#{manual_pol_idx + 1} from left)")

if "POL" in metadata_countries:
    meta_pol_idx = metadata_countries.index("POL")
    print(f"  Metadata:    Position {meta_pol_idx} (#{meta_pol_idx + 1} from left)")

if "POL" in manually_read_countries and "POL" in metadata_countries:
    if manually_read_countries.index("POL") != metadata_countries.index("POL"):
        print(f"  ❌ MISMATCH! User said POL is at position 13, manual read shows position {manual_pol_idx + 1}")
    else:
        print(f"  ✅ Match")

print("\n" + "=" * 70)
print("User correction: Poland is at position 13 from the left")
print(f"Manual read shows POL at position: {manually_read_countries.index('POL') + 1}")
print(f"Metadata shows POL at position: {metadata_countries.index('POL') + 1}")

# The correction needed
print("\n" + "=" * 70)
print("CORRECTION ANALYSIS:")
print("User said Poland is position 13 (index 12)")
print("Manual read shows Poland at position 13 (index 12) ✅")
print("Metadata shows Poland at position 17 (index 16) ❌")
print("\nMetadata has POL, BEL, CHE swapped compared to actual plot!")
print("\nActual order positions 12-15:")
for i in range(11, 16):
    print(f"  {i+1}. {manually_read_countries[i]}")
print("\nMetadata order positions 12-15:")
for i in range(11, 16):
    print(f"  {i+1}. {metadata_countries[i]}")
