import psycopg2 as pg
import pdb
import itertools
from utils.utils import *
from collections import defaultdict
from job_parse import JOBQuery

MAX_WORD_LEN = 14

class PGIterator:
    """Class to implement an iterator
    of powers of two"""

    def __init__(self, sql_queries, args):
        self.sql_queries = sql_queries
        self.args = args
        self.cur_id = 0

    def __iter__(self):
        self.sentences = self.sentence_generator(self.sql_queries, self.args)
        return self

    def __next__(self):
        return next(self.sentences)

    def sentence_generator(self, sql_queries, args):
        '''
        Takes a bunch of sql queries, and generates one sentence at a time, using
        whatever rules have been specified in the args.
        '''
        def find_relevant_attributes():
            queries = []
            # key: table_name, value: list of attributes
            tables = defaultdict(set)
            for fn in glob.glob("job/*.sql"):
                # if (self.args.query_match not in fn):
                    # continue
                queries.append(JOBQuery(fn))
            for q in queries:
                attrs = q.attrs_with_predicate()
                for a in attrs:
                    assert len(a) == 2
                    if ("gender=" in a[1]):
                        continue
                    if self.args.no_id:
                        if ("id" in a[1]):
                            continue
                    tables[a[0]].add(a[1])

                joins = q.joins()
                for j in joins:
                    assert len(j) == 4
                    if ("gender=" in j[1] or "gender=" in j[3]):
                        print("gender= parsing error")
                        continue
                    if self.args.no_id:
                        if ("id" in j[1] or "id" in j[3]):
                            continue
                    tables[j[0]].add(j[1])
                    tables[j[2]].add(j[3])

            return tables

        def handle_sentence(sentence, attrs):
            sentence = []
            for i, word in enumerate(row):
                if "id" in attrs[i] and args.no_id:
                    # print("skipping ", attrs[i])
                    continue
                if word != None:
                    if args.split_words:
                        # This doesn't seem what we want for multi-word things as
                        # in keywords. Especially, if we need to match LIKE
                        # statements on that...
                        all_words = str(word).split()
                        if len(all_words) > 6:
                            continue
                        for w in all_words:
                            if (len(w) > MAX_WORD_LEN):
                                continue
                            if not args.no_preprocess_word:
                                w = preprocess_word(w, exclude_the=args.exclude_the,
                                        exclude_nums=args.exclude_nums)
                            sentence.append(w)
                    else:
                        if (len(word) > MAX_WORD_LEN):
                            continue
                        if not args.no_preprocess_word:
                            word = preprocess_word(str(word), exclude_the=args.exclude_the,
                                    exclude_nums=args.exclude_nums)
                        sentence.append(word)
            return sentence

        def get_relevant_select(tables, sql):
            # let us find the relevant attributes
            sel_attrs = []
            for table in tables:
                if table in sql:
                    attributes = str(tables[table])
                    # convert attributes to a string
                    attributes = attributes.replace("{", "")
                    attributes = attributes.replace("}", "")
                    # messes up the selects
                    attributes = attributes.replace("'", "")
                    attributes = attributes.replace(",", "")
                    # just add each of them
                    all_attrs = attributes.split()
                    # for each of these, need to add the name.alias guys
                    for idx, _ in enumerate(all_attrs):
                        all_attrs[idx] = table + "." + all_attrs[idx]
                    sel_attrs += all_attrs
            select = ""
            for i, sa in enumerate(sel_attrs):
                select += sa
                if (i != len(sel_attrs)-1):
                    select += ","
            return select

        def get_cursor(query):
            # format the query
            conn = pg.connect(host=args.db_host, database=args.db_name)
            cur = conn.cursor(name="named_cursor_more_efficient")
            print("going to execute: ", query)
            cur.execute(query)
            print("successfully executed query!")
            return cur, conn

        if args.relevant_selects:
            tables = find_relevant_attributes()

        print("going to go over sql queries")
        for sql in sql_queries:
            select = "*"
            if args.relevant_selects:
                select = get_relevant_select(tables, sql)
            query = sql.replace("*", select)
            cursor, conn = get_cursor(query)
            # go over every row in the cursor, and yield all valid sentences
            attrs = None
            for row_num, row in enumerate(cursor):
                if row_num % 1000000 == 0:
                    print("row_num: ", row_num)
                if attrs is None:
                    descr = cursor.description
                    attrs = []
                    for d in descr:
                        attrs.append(d[0])
                if args.train_pairs:
                   pair_words  = list(itertools.combinations(row, 2))
                   for pair in pair_words:
                       yield handle_sentence(pair, attrs)
                else:
                    yield handle_sentence(row, attrs)

            #preprocess_rows(sentences, rows, res_attrs)
            cursor.close()
            conn.close()
