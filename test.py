import pandas as pd
import json
import pdb

file_name = "true-10sec.json"
# file_name = "debug-zero-cards.json"

with open(file_name, "r") as f:
    data = json.loads(f.read())

print(data.keys())
query = data["join-order-benchmark/10a.sql"]

for k in query.keys():
    if "movie " in k:
        print(k)
assert " movie_companies title" in query.keys()
pdb.set_trace()

