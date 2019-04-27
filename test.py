import pandas as pd
import json
import pdb

# file_name = "true-30sec-1-5.json"
# file_name = "true-30sec-6-9.json"
file_name = "true.json"

with open(file_name, "r") as f:
    data = json.loads(f.read())

print(data.keys())
print(len(data))

# for k in query.keys():
    # if "movie " in k:
        # print(k)
# assert " movie_companies title" in query.keys()
pdb.set_trace()

