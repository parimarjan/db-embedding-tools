from gensim.models import Word2Vec
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from bokeh.io import push_notebook, show, output_notebook
# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, LabelSet
# output_notebook()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from job_parse import JOBQuery
import glob
from collections import defaultdict
from utils.utils import *
import pdb
import psycopg2 as pg
import pickle

SANITY_CHECK_COMICS = False
HONGZI = False

def find_vector_containing(string):
    keys = model.keys()

def get_all_rows(sql_query):
    '''
    execute the given sql, and return the results in a list.
    '''
    conn = pg.connect(host=args.db_host, database=args.db_name)
    cur = conn.cursor()
    if args.verbose: print("going to execute query: ", sql_query)
    if args.random_prob is not None:
        sql_query += " AND random() < {}".format(args.random_prob)

    cur.execute(sql_query)

    # TODO: extract only list of attributes from descr
    descr = cur.description
    attrs = []
    for d in descr:
        attrs.append(d[0])
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return attrs, rows

# def wv_metric(model):
    # def metric(w1, w2):
        # return model.similarity(w1, w2)
    # return metric

def make_tsne(model, sql_queries):

    def plot_projections(name, labels, X_embedded):
        df = pd.DataFrame(X_embedded, columns = ["x", "y"])
        df["labels"] = labels
        ax = df.plot.scatter(x="x", y="y", c="labels", colormap="viridis",
                alpha=0.3)
        fname = args.sql_dir + name + ".png"
        print("saving ", fname)
        plt.savefig(fname)

    # need to fill up these vectors for arbitrary sql_queries
    X = []
    labels = []
    label_names = []
    names = []

    # word -> (vec, 0/1/2)
    word_vecs = defaultdict(tuple)
    # movie title -> 0/1/2 (label)
    titles = defaultdict(str)

    for label, sql in enumerate(sql_queries):
        label_name = sql[0]
        if "dc" in label_name:
            word_vecs["dccomics"] = (label, wv["dccomics"])
        else:
            word_vecs["marvelcomics"] = (label, wv["marvelcomics"])

        query = sql[1]
        _, rows = get_all_rows(query)
        matched_words = []
        for row in rows:
            if args.sel_label_class:
                label = row[0]
                word = row[1]
            else:
                word = row[0]
            print(label)
            print(word)
            pdb.set_trace()

            pr_word = preprocess_word(word, exclude_words=EXCLUDE_LIST,
                    min_len=4)
            if len(pr_word) <= 0:
                continue

            # FIXME: heuristic
            # w = max(pr_word.split(), key=len)
            titles[pr_word] = label
            for w in pr_word.split():
                matched_words.append(w)

        matched_words = set(matched_words)
        print("num matched words: ", len(matched_words))
        for k in matched_words:
            if k not in model:
                continue
            if SANITY_CHECK_COMICS:
                msim = model.similarity(k, "marvelcomics")
                dsim = model.similarity(k, "dccomics")
                # FIXME: temporary sanity check
                if msim < dsim and "marvel" in label_name:
                    continue
                if msim > dsim and "dc" in label_name:
                    continue
                if "marvel" in label_name and msim < 0.00:
                    continue
                if "dc" in label_name and dsim < 0.00:
                    continue

            # add stuff for the dim reduction step
            X.append(wv[k])
            labels.append(label)
            label_names.append(label_name)
            names.append(k)

            # for hongzi
            word_vecs[k] = (label, wv[k])

    if SANITY_CHECK_COMICS:
        # compute marvel vs dc for marvel movies
        correct_marvel = []
        incorrect_marvel = []
        correct_dc = []
        incorrect_dc = []
        for i, name in enumerate(names):
            msim = model.similarity(name, "marvelcomics")
            dsim = model.similarity(name, "dccomics")
            if "marvel" in label_names[i]:
                if (msim > dsim):
                    correct_marvel.append(name)
                else:
                    incorrect_marvel.append(name)
            elif "dc" in label_names[i]:
                if (msim < dsim):
                    correct_dc.append(name)
                else:
                    incorrect_dc.append(name)
        print("correct marvel: {}, correct dc: {}".format(len(correct_marvel),
            len(correct_dc)))
        print("incorrect marvel: {}, incorrect dc: {}".format(len(incorrect_marvel),
            len(incorrect_dc)))

    if HONGZI:
        with open("word_vecs.pickle", "wb") as f:
            f.write(pickle.dumps(word_vecs))
        with open("titles.pickle", "wb") as f:
            f.write(pickle.dumps(titles))

    X = np.array(X)

    # first do PCA
    X_embedded = PCA(n_components=2).fit_transform(np.copy(X))
    plot_projections("pca", labels, X_embedded)

    PERPLEXITIES = [10, 50, 100]
    for p in PERPLEXITIES:
        X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                verbose=2).fit_transform(X)
        print("Computed t-SNE", X_embedded.shape)
        plot_projections("tsne" + str(p), labels, X_embedded)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_prob", type=float, required=False,
            default=None)
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--sql_dir", type=str, required=False,
            default="tsne-queries")
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/word2vec/")
    parser.add_argument("--model_name", type=str, required=False,
            default="new-wv-nopairs25.bin")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sel_label_class", action="store_true")
    return parser.parse_args()


args = read_flags()
model_name = args.data_dir + args.model_name
model = Word2Vec.load(model_name)
print(model)
wv = model.wv
del model

sql_queries = []
for fn in glob.glob(args.sql_dir + "/*.sql"):
    with open(fn, "r") as f:
        sql_queries.append((fn, f.read()))

print("categories are: ")
for i, q in enumerate(sql_queries):
    print(i, q[0])

EXCLUDE_LIST = ["of", "in", "the", "and", "for", "not"]
make_tsne(wv, sql_queries)

for i, q in enumerate(sql_queries):
    print(i, q[0])
pdb.set_trace()

