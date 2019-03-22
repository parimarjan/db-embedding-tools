import argparse
import psycopg2 as pg
import itertools
import glob
from job_parse import JOBQuery
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from utils.net import ContrastiveLoss, SiameseNetwork
from utils.utils import *
from torch import optim
import pickle
import os
import hashlib

QUERY_TEMPLATE = "SELECT {ATTRIBUTES} FROM {TABLE} WHERE random() < {PROB}"

def read_flags():
    parser = argparse.ArgumentParser()
    # Note: we are using the parser imported from park in order to avoid
    # conflicts with park's argparse
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--data_dir", type=str, required=False,
            default="./data/")
    parser.add_argument("--query_match", type=str, required=False,
            default="sql", help="sql will match all queries")
    parser.add_argument("--net_suffix", type=str, required=False,
            default="", help="sql will match all queries")

    parser.add_argument("--use_saved", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_id", action="store_true")
    parser.add_argument("--dictionary_encoding", action="store_false")
    parser.add_argument("--dictionary_encoding_size", type=int, required=False,
            default=100000)
    parser.add_argument("--hidden_layer_size", type=int, required=False,
            default=10000)
    parser.add_argument("--embedding_layer_size", type=int, required=False,
            default=300)
    parser.add_argument("--epochs", type=int, required=False,
            default=10000)
    parser.add_argument("--minibatch_size", type=int, required=False,
            default=128)
    parser.add_argument("--contrastive_loss_margin", type=float, required=False,
            default=2.0)

    parser.add_argument("--sample_prob", type=float, required=False,
            default=0.1)
    parser.add_argument("--lr", type=float, required=False,
            default=0.0005)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.3)
    return parser.parse_args()

def find_relevant_attributes():
    queries = []
    # key: table_name, value: list of attributes
    tables = defaultdict(set)
    for fn in glob.glob("job/*.sql"):
        if (args.query_match not in fn):
            continue
        queries.append(JOBQuery(fn))
    for q in queries:
        attrs = q.attrs_with_predicate()
        for a in attrs:
            assert len(a) == 2
            if ("gender=" in a[1]):
                continue
            if args.no_id:
                if ("id" in a[1]):
                    continue

            tables[a[0]].add(a[1])

        joins = q.joins()
        for j in joins:
            assert len(j) == 4
            if ("gender=" in j[1] or "gender=" in j[3]):
                continue
            if args.no_id:
                if ("id" in j[1] or "id" in j[3]):
                    continue
            tables[j[0]].add(j[1])
            tables[j[2]].add(j[3])

    return tables

def main():
    print("starting embedding")
    training_fname = get_saved_data_name("training")
    test_fname = get_saved_data_name("test")
    encoding_fname = get_saved_data_name("encoding"+str(args.dictionary_encoding_size))
    print(training_fname)
    print(test_fname)
    training = None
    test = None
    if (not (args.use_saved and os.path.exists(training_fname))):
        tables = find_relevant_attributes()
        positive_samples = []
        for table in tables:
            attributes = str(tables[table])
            # convert attributes to a string
            attributes = attributes.replace("{", "")
            attributes = attributes.replace("}", "")
            # messes up the selects
            attributes = attributes.replace("'", "")
            attrs, rows = sample_rows(table, attributes, prob=args.sample_prob)

            # TODO: maybe train on each sample separately?
            positive_samples += gen_positive_samples(rows, attrs)

        print("gen positive samples done!")
        training, test = gen_training_samples(positive_samples)
        # let us save this data
        with open(training_fname, "wb") as f:
            f.write(pickle.dumps(training))
        with open(test_fname, "wb") as f:
            f.write(pickle.dumps(test))
        with open(encoding_fname, "wb") as f:
            f.write(pickle.dumps(encoding_dict))
        print("written data out!")

    else:
        print("going to use saved data")
        with open(training_fname, "rb") as f:
            training = pickle.loads(f.read())
        with open(test_fname, "rb") as f:
            test = pickle.loads(f.read())
        # encoding must be already read at the start

    train(training)

def train(training_samples):
    print("going to start training. Number of training samples: ",
            len(training_samples))
    net = SiameseNetwork(args.dictionary_encoding_size, args.hidden_layer_size,
            args.embedding_layer_size)
    criterion = ContrastiveLoss(margin=args.contrastive_loss_margin)
    # FIXME: use adaptive learning rate??
    optimizer = optim.Adam(net.parameters(), lr = args.lr)

    # for epoch, data in enumerate(training_samples):
    epochs = max(args.epochs, len(training_samples))
    for epoch in range(epochs):
        input1_mb = []
        input2_mb = []
        label_mb = []
        minibatch_samples = random.sample(training_samples, args.minibatch_size)
        for m in minibatch_samples:
            word1, word2 , label = m
            label = np.array(label, dtype=np.float32)
            input1_mb.append(encode(word1))
            input2_mb.append(encode(word2))
            label_mb.append(label)
        label_mb = to_variable(label_mb).float()

        output1,output2 = net(to_variable(input1_mb).float(),
                to_variable(input2_mb).float())
        optimizer.zero_grad()
        loss_contrastive = criterion(output1,output2,label_mb)
        assert loss_contrastive.dim() == 0
        loss_contrastive.backward()
        optimizer.step()
        if (epoch % 10 == 0):
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.data[0]))

        if (epoch % 100 == 0):
            net_name = get_name(args.net_suffix) + ".net"
            print("saving network with name: ", net_name)
            save_network(net, net_name, epoch, args.data_dir,delete_old=True)

def test(test_samples):
    print("in test")

def get_random_sample(samples):
    if args.dictionary_encoding:
        if random.random() % 2 == 0:
            random_sample = random.sample(samples, 1)[0][0]
        else:
            random_sample = random.sample(samples, 1)[0][1]
    return random_sample

def gen_training_samples(positive_samples):
    print("going to gen training samples, num positive: ",
            len(positive_samples))
    negative_samples = []
    for i, sample in enumerate(positive_samples):
        if (i % 10000 == 0):
            print("gen training samples: ", i)
        # Note: we follow the convention in
        #   https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
        # thus, the positive samples are labeled 0 (the loss function is
        # defined in such a way...)
        positive_samples[i] = tuple((sample[0], sample[1], 0.00))
        # choose one of these, and randomly put in a value there
        # FIXME: should we ensure that the random value is of the same type as
        # the other?
        random_sample = get_random_sample(positive_samples)

        if random.random() % 2 == 0:
            neg_sample = tuple((sample[0], random_sample, 1.00))
        else:
            neg_sample = tuple((random_sample, sample[1], 1.00))
        negative_samples.append(neg_sample)

    train, test = train_test_split(positive_samples + negative_samples,
            test_size=args.test_size)
    return train, test

def sample_rows(table_name, attributes, prob):
    conn = pg.connect(host=args.db_host, database=args.db_name)
    if args.verbose: print("connection succeeded")
    cur = conn.cursor()
    query = QUERY_TEMPLATE.format(ATTRIBUTES = attributes,
                                  TABLE = table_name,
                                  PROB = prob)
    if args.verbose: print("going to execute query: ", query)
    try:
        cur.execute(query)
    except:
        import pdb
        pdb.set_trace()

    # for small table, sample all the data
    # TODO: need to do this somehow.
    if cur.rowcount < 10:
        query = QUERY_TEMPLATE.format(ATTRIBUTES = attributes,
                                      TABLE = table_name,
                                      PROB = 1.00)
        cur.execute(query)

    if args.verbose: print("rowcount: ", cur.rowcount)

    # TODO: extract only list of attributes from descr
    descr = cur.description
    attrs = []
    for d in descr:
        attrs.append(d[0])
    rows = cur.fetchall()

    # just to explore stuff. Particularly, some single character attributes had
    # seemed suspicious, but turned out to be missing values, or gender values
    # (m or f)
    if args.debug:
        for row in rows:
            for column, val in enumerate(row):
                column_name = attrs[column]
                if isinstance(val, str) and not val.isdigit() and len(val)==1 \
                    and val != "m" and val != "f" and val != "-":
                    import pdb
                    pdb.set_trace()

    cur.close()
    conn.close()
    return attrs, rows

def gen_positive_samples(rows, attrs=None):
    '''
    FIXME: how should we be using attributes
    gen every pair in rows to create positive samples
    '''
    positive_samples = []
    for row in rows:
        # gen every subset of length 2
        pairs = list(itertools.combinations(row, 2))
        for p in pairs:
            if not None in p:
                positive_samples.append(p)
                if args.dictionary_encoding:
                    if p[0] not in encoding_dict:
                        encoding_dict[p[0]] = gen_hashed_index(p[0])

                    if p[1] not in encoding_dict:
                        encoding_dict[p[1]] = gen_hashed_index(p[1])

    return positive_samples

def get_saved_data_name(name_suffix):
    return args.data_dir + "/" + get_name(name_suffix) + ".json"

def get_name(name_suffix):
    # 1:4 to avoid the annoying negative sign that comes up sometimes (can't
    # delete those files easily...)
    key_vals = args.query_match + str(args.sample_prob) + str(args.no_id)
    name = str(deterministic_hash(str(key_vals)))[1:4] +"-"+name_suffix
    return name

def gen_hashed_index(val):
    # FIXME: deal with collision etc.
    # return hash(val) % args.dictionary_encoding_size
    return deterministic_hash(val) % args.dictionary_encoding_size

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def encode(val):
    if args.dictionary_encoding:
        if val not in encoding_dict:
            encoding_dict[val] = gen_hashed_index(val)

        # convert this into a np array with only val as 1
        idx = encoding_dict[val]
        one_hot_encoding = np.zeros(args.dictionary_encoding_size,
                dtype=np.float32)
        one_hot_encoding[idx] = 1.00
        return one_hot_encoding

# just keep these global for now
args = read_flags()
if args.dictionary_encoding:
    fname = get_saved_data_name("encoding" + str(args.dictionary_encoding_size))
    if args.use_saved and os.path.exists(fname):
        with open(fname, "rb") as f:
            encoding_dict = pickle.loads(f.read())
        print("unique values in encoding dict are: ",
                len(encoding_dict.keys()))
    else:
        encoding_dict = {}

main()
