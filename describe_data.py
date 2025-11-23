import os
import json

import pandas as pd
import numpy as np

from pdb import set_trace as pst

def print_percentiles(sorted_array, percentiles=[0, 25, 50, 75, 100]):
    arr = np.array(sorted_array)
    n = len(arr)
    if n == 0:
        print("Error: Empty array!")
        return

    # Validate and sort if needed
    if not np.array_equal(arr, np.sort(arr)):
        print("Warning: Array not sorted - auto-sorted.")
        arr = np.sort(arr)

    # Filter valid percentiles
    valid_percents = [p for p in percentiles if 0 <= p <= 100]
    invalid_percents = [p for p in percentiles if not 0 <= p <= 100]
    if invalid_percents:
        print(f"Warning: Ignored invalid percentiles: {invalid_percents} (must be 0-100)")

    # Calculate percentiles
    values = np.percentile(arr, valid_percents, method='linear')
    results = list(zip(valid_percents, np.round(values, 4)))

    # Print table
    print("\n" + "-" * 35)
    print(f"{'Percentile':<12} {'Value':<10}")
    print("-" * 35)
    for p, val in results:
        print(f"{p:<12.1f} {val:<10.4f}")
    print("-" * 35)
    print(f"Array Info: Length={n}, Min={arr.min():.4f}, Max={arr.max():.4f}")
    print("-" * 35 + "\n")

ORIGIN_DIR = "./origin_data/recipes_cleaned.json"

with open(ORIGIN_DIR, "r") as infile:
    data = json.load(infile)

with open("./origin_data/recipes_cleaned_sample10000.json", "w") as outfile:
    json.dump(data[:10000], outfile, ensure_ascii=False, indent=2)

str_len = []
for di in data:
    str_len.append(tuple((len(di["text"]) + len(di["name"]), di["id"])))

str_len.sort()

percentilies = [i*5 for i in range(20)]
percentilies.extend([1,2,3,4,96, 97, 98, 99])
percentilies.sort()

print(percentilies)

lens = [di[0] for di in str_len]

print_percentiles(lens, percentilies)

pst()
