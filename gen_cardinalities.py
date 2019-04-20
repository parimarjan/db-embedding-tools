import sqlparse
import glob
import argparse
import pdb
import itertools
from utils.utils import *
from utils.db_utils import *
import json
import os

QUERY_TMP_ANALYZE = "EXPLAIN SELECT COUNT(*) FROM {TABLES} {CONDS}"
QUERY_TMP = "SELECT COUNT(*) FROM {TABLES} {CONDS}"

def find_all_tables_till_keyword(token):
    tables = []
    # print("fattk: ", token)
    index = 0
    while (True):
        if (type(token) == sqlparse.sql.Comparison):
            left = token.left
            right = token.right
            if (type(left) == sqlparse.sql.Identifier):
                tables.append(left.get_parent_name())
            if (type(right) == sqlparse.sql.Identifier):
                tables.append(right.get_parent_name())
            break
        elif (type(token) == sqlparse.sql.Identifier):
            tables.append(token.get_parent_name())
            break
        try:
            index, token = token.token_next(index)
            if ("Literal" in str(token.ttype)) or token.is_keyword:
                break
        except:
            # print(token)
            # pdb.set_trace()
            break
    # print("returning tables: ", tables)
    return tables


def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    match = ""
    index, token = wheres.token_next(index)
    if token is None:
        return None, None
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres)
    in_between = False
    while True:
        if token is None:
            break
        _, next_token = token_list.token_next(index)

        if not in_between and token.value == "AND":
            break
        match += " " + token.value
        if (token.value == "BETWEEN"):
            in_between = True
        index, token = token_list.token_next(index)

    # print("match: ", match)
    for table in tables_in_pred:
        if table not in tables:
            return index, None

    if len(tables_in_pred) == 0:
        return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        if index is None:
            break
        if match is not None:
            matched.append(match)

    # print("tables: ", tables)
    # print("matched: ", matched)
    return matched

def handle_query(tables, wheres):
    all_data = {}
    conn = pg.connect(host=args.db_host, database=args.db_name,
            user=args.user_name,
            password=args.pwd)
    cur = conn.cursor()
    combs = []

    all_tables = [t[0] for t in tables]
    all_aliases = [t[1] for t in tables]

    for i in range(1, len(tables)+1):
        # combs += itertools.combinations(tables.keys(), i)
        # combs += itertools.combinations(table_names, i)
        combs += itertools.combinations(list(range(len(all_tables))), i)

    for comb in combs:
        tables_string = ""
        aliases = []
        used_tables = []
        for i, idx in enumerate(comb):
            aliases.append(all_aliases[idx])
            used_tables.append(all_tables[idx])
            tables_string += all_tables[idx] + " AS " + all_aliases[idx]
            if i != len(comb)-1:
                tables_string += ","

        matches = find_all_clauses(aliases, wheres)
        cond_string = ""

        for i, m in enumerate(matches):
            cond_string += m
            if i != len(matches)-1:
                cond_string += " AND "

        if cond_string != "":
            cond_string = "WHERE " + cond_string

        # used_tables = [c for c in comb]
        used_tables.sort()

        tkey = ""
        for usedt in used_tables:
            tkey += " " + usedt

        if args.exact_count:
            query = QUERY_TMP.format(TABLES=tables_string, CONDS=cond_string)
            cur.execute("SET statement_timeout = {}".format(args.query_timeout))
            try:
                cur.execute(query)
                exp_output = cur.fetchall()
                count = int(exp_output[0][0])
            # only catch the timeout error
            except pg.OperationalError:
                count = 100000000
                conn.close()
                cur.close()
                conn = pg.connect(host=args.db_host, database=args.db_name,
                        user=args.user_name,
                        password=args.pwd)
                cur = conn.cursor()
        else:
            query = QUERY_TMP_ANALYZE.format(TABLES=tables_string, CONDS=cond_string)
            cur.execute(query)
            exp_output = cur.fetchall()
            count = parse_explain(exp_output)

        all_data[tkey] = count
        print(query, tkey, count)
        if count == 0:
            print("COUNT WAS 0!!!!")
            # pdb.set_trace()

    conn.close()
    cur.close()
    return all_data

def main():
    all_file_names = glob.glob("join-order-benchmark/*.sql")
    queries = []
    file_names = []
    base_patterns = args.sql_file_pattern.split(",")
    patterns = []
    for p in base_patterns:
        patterns.append(p + "a")
        patterns.append(p + "b")
        patterns.append(p + "c")
        patterns.append(p + "d")
        patterns.append(p + "e")

    print(patterns)
    for fn in all_file_names:
        if len(patterns) > 0:
            for p in patterns:
                if "/" + p in fn:
                    print("reading: ", fn)
                    with open(fn, "rb") as f:
                        queries.append(f.read())
                    file_names.append(fn)
        else:
            print("reading everything!")
            # read everything
            file_names.append(fn)
            with open(fn, "rb") as f:
                queries.append(f.read())

    all_queries = {}
    for i, q in enumerate(queries):
        parsed = sqlparse.parse(q)[0]
        # let us go over all the where clauses
        where_clauses = None
        tables = []
        ident_seen = 0
        for token in parsed.tokens:
            if (type(token) == sqlparse.sql.Where):
                where_clauses = token
            if (type(token) == sqlparse.sql.IdentifierList):
                if (ident_seen == 0):
                    ident_seen += 1
                    continue
                # second time will be the tables
                all_tables = str(token).replace(",","").replace(" ","").split("\n")
                # for t in all_tables:
                    # tables[t.split("AS")[0]] = t.split("AS")[1]
                for t in all_tables:
                    tables.append((t.split("AS")[0], t.split("AS")[1]))

        # dumb, dumb hack
        if (len(tables) == 0):
            for token in parsed.tokens:
                if (type(token) == sqlparse.sql.IdentifierList):
                    all_tables = str(token).replace(",","").replace(" ","").split("\n")
                    # for t in all_tables:
                        # tables[t.split("AS")[0]] = t.split("AS")[1]
                    for t in all_tables:
                        tables.append((t.split("AS")[0], t.split("AS")[1]))

        assert where_clauses is not None
        # token_list = sqlparse.sql.TokenList(where_clauses)
        file_name = file_names[i]
        # if "22d.sql" not in file_name:
            # continue
        # print(tables)
        # pdb.set_trace()
        # where_clauses = str(where_clauses).replace("WHERE","").replace("AND","").replace(";","").split("\n")
        all_queries[file_name] = (tables, where_clauses)

    all_counts = {}
    num = 0
    old_saved_data = {}
    if os.path.exists(args.output_name):
        with open(args.output_name, "r") as f:
            old_saved_data = json.loads(f.read())
        all_counts.update(old_saved_data)

    for k, query in all_queries.items():
        # if "22d.sql" not in k:
            # continue
        print(num)
        print(k)
        if k in all_counts:
            print('already have the data!')
            continue
        data = handle_query(query[0], query[1])
        all_counts[k] = data
        with open(args.output_name, "w") as f:
            f.write(json.dumps(all_counts))
        print("written out json!")

        num += 1

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False,
            default="~/data/")
    parser.add_argument("--db_name", type=str, required=False,
            default="imdb")
    ## other vals
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--user_name", type=str, required=False,
            default="imdb")
    parser.add_argument("--pwd", type=str, required=False,
            default="")
    parser.add_argument("--output_name", type=str, required=False,
            default="test.json")
    parser.add_argument("--debug", type=int, required=False,
            default=0)
    parser.add_argument("--query_timeout", type=int, required=False,
            default=10000)
    parser.add_argument("--exact_count", type=int, required=False,
            default=1)
    parser.add_argument("--sql_file_pattern", type=str, required=False,
            default="")

    return parser.parse_args()

args = read_flags()
main()
