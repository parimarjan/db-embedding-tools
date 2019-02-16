from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence
import argparse
from collections import defaultdict
from job_parse import JOBQuery
import glob
import psycopg2 as pg
import pickle
import hashlib
import os
import string
import pdb
from utils.utils import *
import itertools
from pg_iterator import PGIterator
import time

QUERY_TEMPLATE = "SELECT {ATTRIBUTES} FROM {TABLE} WHERE random() < {PROB}"

def read_flags():
    parser = argparse.ArgumentParser()
    # Note: we are using the parser imported from park in order to avoid
    # conflicts with park's argparse
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--sql_dir", type=str, required=False,
            default=None)
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/word2vec/")
    parser.add_argument("--model_name", type=str, required=False,
            default="model.bin")
    parser.add_argument("--query_match", type=str, required=False,
            default="sql", help="sql will match all queries")
    parser.add_argument("--sql", type=str, required=False,
            default=None, help="sql to execute for getting rows")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--all_attributes", action="store_true")
    parser.add_argument("--no_id", action="store_false")
    parser.add_argument("--sample_prob", type=float, required=False,
            default=0.1)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--split_words", action="store_false")

    parser.add_argument("--train_pairs", action="store_true")
    parser.add_argument("--exclude_the", action="store_false")

    # we don't usually want to do this (...)
    parser.add_argument("--exclude_nums", action="store_true")

    parser.add_argument("--regen_sentences", action="store_true")
    parser.add_argument("--sentence_gen", action="store_true")
    parser.add_argument("--relevant_selects", action="store_true")
    parser.add_argument("--no_pickle", action="store_false")
    parser.add_argument("--skipgram", type=int, required=False,
            default=0)

    parser.add_argument("--min_count", type=int, required=False,
            default=100)
    parser.add_argument("--embedding_size", type=int, required=False,
            default=100)
    parser.add_argument("--gensim_model_name", type=str, required=False,
            default="word2vec")

    return parser.parse_args()

def find_relevant_attributes():
    queries = []
    # key: table_name, value: list of attributes
    tables = defaultdict(set)
    for fn in glob.glob("job/*.sql"):
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
                print("gender= parsing error")
                continue
            if args.no_id:
                if ("id" in j[1] or "id" in j[3]):
                    continue
            tables[j[0]].add(j[1])
            tables[j[2]].add(j[3])

    return tables

def train(sentences):
    print("starting to train!")
    # train model
    if args.train_pairs and args.relevant_selects:
        min_count = args.min_count*5
    elif args.train_pairs:
        min_count = args.min_count*10
    else:
        min_count = args.min_count

    if "word2vec" in args.gensim_model_name:
        model = Word2Vec(sentences, size=args.embedding_size, window=20,
                sg=args.skipgram,
                workers=16, min_count=min_count)
    elif "fast" in args.gensim_model_name:
        model = FastText(sentences,
            size=args.embedding_size, window=20, sg=args.skipgram, workers=16,
            min_count=min_count)

    # summarize the loaded model
    print(model)
    # access vector for one word
    # save model
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(args.data_dir + args.model_name)
    # pdb.set_trace()
    # load model
    # new_model = Word2Vec.load(args.data_dir + args.model_name)
    # print(new_model)

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

    cur.close()
    conn.close()
    return attrs, rows

def main():
    sentences_fname = get_saved_data_name("sentences")

    print("sentences name is: ", sentences_fname)
    tables = find_relevant_attributes()
    sentences = []
    if args.sentence_gen:
        sql_queries = []
        for fn in glob.glob(args.sql_dir + "/*.sql"):
            with open(fn, "r") as f:
                sql_queries.append(f.read())
        sentences = PGIterator(sql_queries, args)
    elif (os.path.exists(sentences_fname + ".txt")  or
            os.path.exists(sentences_fname + ".pickle")) and not args.regen_sentences:
        if args.no_pickle:
            print("going to use saved pickle!")
            with open(sentences_fname + ".pickle", "rb") as f:
                sentences = pickle.loads(f.read())
            print("loaded the saved pickle into memory!")
        else:
            sentences = LineSentence(sentences_fname + ".txt")
    else:
        # just grab it from relevant tables
        if args.sql is None and args.sql_dir is None:
            for table in tables:
                attributes = ""
                if args.all_attributes:
                    attributes = "*"
                else:
                    attributes = str(tables[table])
                    # convert attributes to a string
                    attributes = attributes.replace("{", "")
                    attributes = attributes.replace("}", "")
                    # messes up the selects
                    attributes = attributes.replace("'", "")
                attrs, rows = sample_rows(table, attributes, prob=args.sample_prob)
                preprocess_rows(sentences, rows, attrs)
                print("table: ", table)
        elif args.sql_dir is not None:
            print("sql dir: ", args.sql_dir)
            sql_queries = []
            for fn in glob.glob(args.sql_dir + "/*.sql"):
                with open(fn, "r") as f:
                    sql_queries.append(f.read())
            sentence_it = PGIterator(sql_queries, args)
            # let us just try to do both
            sentences_list = list(sentence_it)
            f = open(sentences_fname + ".txt", "w")
            start = time.time()
            for sentence in sentences_list:
                # s is a list that should be converted to a sentence
                out = ""
                for word_i, word in enumerate(sentence):
                    out += word
                    if word_i != len(sentence)-1:
                        out += " "
                    else:
                        out += "\n"
                    f.write(out)
            f.close()
            print("writing out to file took: ", time.time() - start)

            # try to write out in pickle form as well. this seems to
            # arbitrarily fail sometimes?
            try:
                with open(sentences_fname+".pickle", "wb") as f:
                    f.write(pickle.dumps(sentences_list))
                sentences = sentences_list
            except:
                print("pickle.dumps failed:(")
                sentences = LineSentence(sentences_fname)
                pdb.set_trace()

            print("pickle.dumps successful!")
        else:
            assert False

    train(sentences)

def get_saved_data_name(name_suffix):
    name = args.data_dir + "/" + get_name(name_suffix)
    return name

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def get_name(name_suffix):
    # 1:4 to avoid the annoying negative sign that comes up sometimes (can't
    # delete those files easily...)
    key_vals = args.query_match + str(args.sample_prob) + str(args.no_id)
    if args.sql is not None:
        key_vals += args.sql
    if args.sql_dir is not None:
        key_vals += args.sql_dir
    name = str(deterministic_hash(str(key_vals)))[1:4] +"-"+name_suffix
    # name += args.model_name
    name += str(args.train_pairs)
    name += str(args.exclude_the)
    name += str(args.exclude_nums)
    return name

args = read_flags()
main()


