#!/usr/bin/env python3
"""Verify Poland's position in corrected metadata."""

import json

with open('plot_metadata.json', 'r') as f:
    metadata = json.load(f)

countries = metadata['9781484392935_f0013-01.jpg']['x_axis']['countries']

print("Corrected country order (positions 10-18):")
print("=" * 50)
for i in range(10, 18):
    marker = " ← POLAND" if countries[i] == "POL" else ""
    print(f"Position {i+1:2d} (index {i:2d}): {countries[i]}{marker}")

poland_idx = countries.index('POL')
print(f"\n{'='*50}")
print(f"Poland (POL) is at:")
print(f"  Index: {poland_idx}")
print(f"  Position from left: {poland_idx + 1}")
print(f"\nUser said Poland is at position 13: {'✅ CORRECT' if poland_idx == 12 else '❌ WRONG'}")
