import pickle
import pdb
# for reading a single query file
# f = open("features-wv100-all/20a.pickle", 'rb')
f = open("features-nowv/20a.pickle", 'rb')
dict = pickle.loads(f.read())
for k, v in dict.items():
    # key is of the form n.name. Same as it was in the parser you had I think -
    # but let me know if you want to change the format
    print(k, len(v))
    print("last feature: ", v[-1])
    print("first feature: ", v[0])

pdb.set_trace()
