from gensim.models import Word2Vec
import argparse
import numpy as np
import pandas as pd
import pandas.tools
import pandas.tools.plotting
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
import seaborn as sns
import os
import enchant
import random

KEYWORD_SELECT_TMP = """SELECT title.title
FROM movie_keyword,
keyword,
title
WHERE movie_keyword.keyword_id = keyword.id
AND movie_keyword.movie_id = title.id
AND keyword.keyword = '{keyword}'"""


def find_vector_containing(string):
    keys = model.keys()

def get_all_rows(sql_query):
    '''
    execute the given sql, and return the results in a list.
    '''
    conn = pg.connect(host=args.db_host, database=args.db_name)
    cur = conn.cursor()
    if args.random_prob is not None and "where" in sql_query.lower():
        sql_query += " AND random() < {}".format(args.random_prob)
    elif args.random_prob is not None:
        sql_query += " WHERE random() < {}".format(args.random_prob)
    print("going to execute query: ", sql_query)
    cur.execute(sql_query)

    # TODO: extract only list of attributes from descr
    descr = cur.description
    attrs = []
    for d in descr:
        attrs.append(d[0])
    rows = cur.fetchall()

    cur.close()
    conn.close()
    print("returning {} rows".format(len(rows)))
    return attrs, rows

def get_name(name):
    suffix = str(args.no_pca_tsne) + "prob-" + str(args.random_prob) \
            + "exclude-english-" + str(args.exclude_english)
    suffix = hash(suffix)[0:3]
    return name + suffix

def plot_subset(df, subset, name = "test", subset_label="subset"):
    '''
    df has the embedding. subset is a bunch of words from within the embedding
    which will be colored brightly, and the rest would be colored a drab gray.
    '''
    print("plot subset!")
    name = get_name(name)

    labels = []
    label_ints = []
    for w in df["words"]:
        if w in subset:
            labels.append(subset_label)
            label_ints.append(1)
        else:
            labels.append("others")
            label_ints.append(0)
    df["labels"] = labels
    df["label_ints"] = label_ints

    plot_projections(name, df)

def get_processed_words(model, query):
    _, rows = get_all_rows(query)
    X = []
    words = []
    print("going to process returned words now")
    # FIXME: decompose
    for row in rows:
        matched_words = []
        word = row[0]
        pr_word = preprocess_word(word, exclude_words=EXCLUDE_LIST,
                min_len=4)
        if len(pr_word) <= 0:
            continue

        # FIXME: heuristic
        # w = max(pr_word.split(), key=len)
        for w in pr_word.split():
            if args.exclude_english and english_dict.check(w):
                continue
            else:
                matched_words.append(w)

        matched_words = set(matched_words)
        for k in matched_words:
            if k in words:
                continue
            if k not in model:
                continue
            # add stuff for the dim reduction step
            X.append(wv[k])
            words.append(k)

    print(len(words))
    assert len(words) == len(X)
    return X, words

def make_projection(model, query):
    X_all = []
    if args.project_all:
        all_words = []
        for w in model.index2word:
            X_all.append(model[w])
            all_words.append(w)
        X_all = np.array(X_all)
        print("done adding all words to X")

    print("make projection")
    X, words = get_processed_words(model, query)

    # random titles
    rand_query = "select title from title where random() < 0.001"
    _, rand_titles = get_processed_words(model, rand_query)
    # marvel / dc titles
    marvel_query = KEYWORD_SELECT_TMP.format(keyword="marvel_comics")
    _, marvel_titles = get_processed_words(model, marvel_query)

    # sci-fi / genre based movies
    if args.project_all:
        X_all = PCA(n_components=2).fit_transform(X_all)
        df = pd.DataFrame(X_all, columns = ["x", "y"])
        df["words"] = all_words
        plot_subset(df, words, name="titles", subset_label="titles")

    if args.pca:
        X_embedded= PCA(n_components=2).fit_transform(X)
        df = pd.DataFrame(X_embedded, columns = ["x", "y"])
        df["words"] = words
        plot_subset(df, rand_titles, name="rand", subset_label="rand")
        plot_subset(df, marvel_titles, name="marvel", subset_label="marvel")
        pdb.set_trace()

    # FIXME: decompose
    # PERPLEXITIES = [10, 25, 50, 100, 500, 1000, 5000]
    PERPLEXITIES = [500, 1000, 5000, 100]
    for p in PERPLEXITIES:
        print("perplexity: ", p)
        name = "TSNE" + str(p)
        name = get_pickle_name(name)
        df = None
        if os.path.exists(name):
            with open(name, "rb") as f:
                df = pickle.loads(f.read())
        else:
            print("first do pca, then tsne!")
            X_PCA = PCA(n_components=args.pca_dim).fit_transform(X)
            print("going to do tsne with: ", X_PCA.shape)
            X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                    verbose=0).fit_transform(X_PCA)

            print("Computed t-SNE", X_embedded.shape)
            df = pd.DataFrame(X_embedded, columns = ["x", "y"])
            df["words"] = words

            with open(name, "wb") as f:
                f.write(pickle.dumps(df))

        # plot_projections(name, df)

def get_pickle_name(name):
    outdir = args.output_dir
    name += "pca-" + str(args.pca_dim)
    name += "-noeng-" + str(args.exclude_english)[0]
    if not args.random_prob is None:
        name += "-rand-"+ str(args.random_prob)
    return outdir + "/" + name + ".pickle"

def plot_projections(name, df):
    name = get_name(name)
    outdir = args.sql_dir

    fname = outdir + name + ".png"

    # plots each label too
    groups = df.groupby('labels')
    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        # alpha = 0.05
        # ms = 2
        if len(group) < 1000:
            alpha = 1.0
            ms = 4
        else:
            alpha = 0.05
            ms = 2

        ax.plot(group.x, group.y, marker='o', linestyle='', ms=ms,
                label=name, alpha=alpha)

    ax.legend(loc="best")
    print('saving file: ', fname)
    plt.savefig(fname)
    plt.close()

    # FIXME: try more.
    # print("plot projections with catplot!")
    # seaborn is really slow :|
    # sns_plot = sns.catplot(x="x", y="y", hue="labels", kind="swarm",
            # data=df)
    # sns_plot = sns.catplot(x="x", y="y", hue="labels",
            # data=df)
    # print("saving ", fname)
    # sns_plot.savefig(fname)

def make_embedding(model, sql_queries):
    '''
    should return df with embedding + word corresponding to each embedding.
    '''
    X = []
    names = []
    labels = []
    for query in sql_queries:
        _, rows = get_all_rows(query)
        for row in rows:
            matched_words = []
            assert len(row) == 1
            word = row[0]
            pr_word = preprocess_word(word, exclude_words=EXCLUDE_LIST,
                    min_len=4)
            if len(pr_word) <= 0:
                continue

            # FIXME: heuristic
            # w = max(pr_word.split(), key=len)
            for w in pr_word.split():
                if args.exclude_english and english_dict.check(w):
                    continue
                else:
                    matched_words.append(w)

            matched_words = set(matched_words)
            for k in matched_words:
                if k not in model:
                    continue
                # do not re-add stuff!
                if k in names:
                    continue
                # add stuff for the dim reduction step
                X.append(wv[k])
                names.append(k)
                labels.append("other")

    print("total data points: ", len(X))
    X = np.array(X)
    embeddings = {}

    X_embedded = PCA(n_components=2).fit_transform(np.copy(X))
    df = pd.DataFrame(X_embedded, columns = ["x", "y"])
    df["words"] = names
    df["labels"] = labels

    PERPLEXITIES = [25, 50, 100, 500, 1000, 5000]
    for p in PERPLEXITIES:
        if args.no_pca_tsne:
            X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                    verbose=0).fit_transform(X)
        else:
            print("first do pca, then tsne!")
            X_PCA = PCA(n_components=20).fit_transform(X)
            print("going to do tsne with: ", X_PCA.shape)
            X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                    verbose=0).fit_transform(X_PCA)

        print("Computed t-SNE", X_embedded.shape)
        df = pd.DataFrame(X_embedded, columns = ["x", "y"])
        df["words"] = names
        df["labels"] = labels
        embeddings["tsne" + str(p)] = df
        plot_projections("TSNE" + str(p) + "_no_labels", df)

    return embeddings

def make_tsne(model, sql_queries):

    # need to fill up these vectors for arbitrary sql_queries
    X = []
    labels = []
    names = []
    label_ints = []

    for label_int, sql in enumerate(sql_queries):
        label_name = sql[0]
        query = sql[1]
        _, rows = get_all_rows(query)
        for row in rows:
            matched_words = []
            if args.sel_label_class:
                assert len(row) == 2
                label_name = row[0]
                word = row[1]
            else:
                assert len(row) == 1
                word = row[0]

            pr_word = preprocess_word(word, exclude_words=EXCLUDE_LIST,
                    min_len=4)
            if len(pr_word) <= 0:
                continue

            # FIXME: heuristic
            # w = max(pr_word.split(), key=len)
            for w in pr_word.split():
                if args.exclude_english and english_dict.check(w):
                    continue
                else:
                    matched_words.append(w)

            matched_words = set(matched_words)
            for k in matched_words:
                if k not in model:
                    continue
                # do not re-add stuff!
                if k in names:
                    continue
                # add stuff for the dim reduction step
                if label_name == "others":
                    if random.random() < 0.01:
                        X.append(wv[k])
                        labels.append(label_name)
                        label_ints.append(label_int)
                        names.append(k)
                else:
                    X.append(wv[k])
                    labels.append(label_name)
                    label_ints.append(label_int)
                    names.append(k)

    print(set(labels))
    print("total data points: ", len(X))
    X = np.array(X)

    # first do PCA
    print("going to do PCA!")
    if args.pca:
        X_embedded = PCA(n_components=2).fit_transform(np.copy(X))
        df = pd.DataFrame(X_embedded, columns = ["x", "y"])
        df["labels"] = labels
        df["label_ints"] = label_ints
        plot_projections("PCA", df)

    # PERPLEXITIES = [10, 25, 50, 100, 500, 1000]
    PERPLEXITIES = [500, 1000]
    for p in PERPLEXITIES:
        if args.no_pca_tsne:
            X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                    verbose=0).fit_transform(X)
        else:
            print("first do pca, then tsne!")
            X_PCA = PCA(n_components=20).fit_transform(X)
            print("going to do tsne with: ", X_PCA.shape)
            X_embedded = TSNE(n_components=2, metric='cosine', perplexity=p,
                    verbose=0).fit_transform(X_PCA)

        print("Computed t-SNE", X_embedded.shape)
        df = pd.DataFrame(X_embedded, columns = ["x", "y"])
        df["labels"] = labels
        df["label_ints"] = label_ints
        plot_projections("TSNE" + str(p), df)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_prob", type=float, required=False,
            default=None)
    parser.add_argument("--pca_dim", type=int, required=False,
            default=5)
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--sql_dir", type=str, required=False,
            default="tsne-queries")
    parser.add_argument("--output_dir", type=str, required=False,
            default="new-tsne")
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/word2vec/")
    parser.add_argument("--model_name", type=str, required=False,
            default="new-wv-nopairs25.bin")

    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exclude_english", action="store_true")
    parser.add_argument("--sel_label_class", action="store_true")
    parser.add_argument("--no_pca_tsne", action="store_true")
    parser.add_argument("--no_make_single_projection", action="store_true")
    parser.add_argument("--project_all", action="store_true")

    return parser.parse_args()

args = read_flags()
model_name = args.data_dir + args.model_name
model = Word2Vec.load(model_name)
wv = model.wv
del(model)
english_dict = enchant.Dict("en_US")

EXCLUDE_LIST = ["of", "in", "the", "and", "for", "not"]

unlabeled_queries = []
label_queries = []

for fn in glob.glob(args.sql_dir + "/*.sql"):
    with open(fn, "r") as f:
        label = os.path.basename(fn).replace(".sql", "")
        if "all" in label:
            unlabeled_queries.append(f.read())
        else:
            label_queries.append((label, f.read()))

# as long as the queries are same + sample prob, take name
emb_fn = gen_name(args.sql_dir + "/embeddings")
emb_fn += ".pickle"
if not os.path.exists(emb_fn):
    embeddings = make_embedding(wv, unlabeled_queries)
    # save embeddings!
    with open(emb_fn, "wb") as f:
        f.write(pickle.dumps(embeddings))
else:
    with open(emb_fn, "rb") as f:
        embeddings = pickle.loads(f.read())

print("now time to generate labels!")

pdb.set_trace()

