import pickle
from pprint import pprint

# Change this to the path of your .pkl file
pkl_file = "test_mutag240k_query_subgraphs.pkl"

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
try:
    print(f"Length: {len(data)}")
except Exception:
    pass
print("\nFormatted content:\n")
pprint(data)
