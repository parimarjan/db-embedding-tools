from gensim.models import Word2Vec
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
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

def find_vector_containing(string):
    keys = model.keys()

def emit_job_features(wv):
    # first open all files
    print("emit job features...")
    queries = []
    # key: table_name, value: list of attributes
    # file_names = glob.glob("job/*.sql")
    file_names = glob.glob("job-ryan/*.sql")
    for fn in file_names:
        # if (args.query_match not in fn):
            # continue
        queries.append(JOBQuery(fn))

    for idx, q in enumerate(queries):
        attrs = list(q.attrs_with_predicate(values=True))
        joins = list(q.joins())
        print("query: ", file_names[idx])
        if "19a.sql" in file_names[idx]:
            print("19a.sql!")
            import pdb
            pdb.set_trace()

        print(attrs)
        print(joins)
        for a in attrs:
            print(a)
            assert len(a) == 4
            table_name = a[0]
            attribute = a[1]
            cmp_op = a[2]
            values = a[3]
            # can be multiple values for each predicate (e.g., when cmp op = IN)
            # FIXME: use conditining for different comparison operators
            for v in values:
                v = preprocess_word(v)
                print(v)

        import pdb
        pdb.set_trace()

def make_tsne(model):
    X = []
    words = []
    MAX_WORDS = 100
    for i, word in enumerate(model.wv.vocab):
        if i >= MAX_WORDS:
            break
        X.append(model.wv[word])
        words.append(word)

    X = np.array(X)
    print("Computed X: ", X.shape)
    X_embedded = TSNE(n_components=2, n_iter=250, verbose=2).fit_transform(X)
    print("Computed t-SNE", X_embedded.shape)

    # make a mapping from category to your favourite colors and labels
    # category_to_color = {0: 'lightgreen', 1: 'lawngreen', 2:'limegreen', 3: 'darkgreen'}
    # category_to_label = {0: 'A', 1:'B', 2:'C', 3:'D'}

    # plot each category with a distinct label
    # fig, ax = plt.subplots(1,1)
    # for category, color in category_to_color.items():
        # mask = y == category
        # ax.plot(X_reduced[mask, 0], X_reduced[mask, 1], 'o',
                # color=color, label=category_to_label[category])

    # ax.legend(loc='best')

    df = pd.DataFrame(columns=['x', 'y', 'word'])

    # df['x'], df['y'], df['word'] = X_embedded[:,0], X_embedded[:,1], model.wv.vocab[0:MAX_WORDS]
    # df['x'], df['y'], df['word'] = X_embedded[:,0], X_embedded[:,1], words
    df['x'], df['y'] = X_embedded[:,0], X_embedded[:,1]

    ax = plt.figure()
    df.plot(lw=2, colormap='jet', marker='.', markersize=10,
                     title='')

    for i, txt in enumerate(words):
        ax.annotate(txt, (df['x'][i], df['y'][i]))

    plt.savefig("test.png")

    # source = ColumnDataSource(ColumnDataSource.from_df(df))
    # labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                           # text_font_size="8pt", text_color="#555555",
                                             # source=source,
                                             # text_align='center')


    # plot = figure(plot_width=600, plot_height=600)
    # plot.circle("x", "y", size=12, source=source, line_color="black",
            # fill_alpha=0.8)
    # plot.add_layout(labels)

    # import pdb
    # pdb.set_trace()
    # show(plot, notebook_handle=True)

model_dir = "/data/pari/embeddings/word2vec/"
# model_name = model_dir + "all_attributes.bin"
model_name = model_dir + "all_attributes_split_words.bin"
# model_name = model_dir + "preprocessed-words-model.bin"
# model_name = model_dir + "joined-tables-half.bin"

model = Word2Vec.load(model_name)
print(model)
wv = model.wv
del model
emit_job_features(wv)
