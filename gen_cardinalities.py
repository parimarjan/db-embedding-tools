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

def handle_token_list(token_list, data, identifier="", cmp_op=""):
    in_predicate = True
    index = 0
    connector = ""   # first one is always ""
    if identifier == "":
        in_predicate = False
    while (True):
        index, token = token_list.token_next(index)
        if args.debug: print("next token: ", token)
        if token is None:
            break
        # now we can deal differently with different types of tokens
        if args.debug: print(token.ttype, token)
        if (type(token) == sqlparse.sql.Comparison):
            # comparison seems to have the value within it as well
            # token.left / token.right should give us what we need
            identifier = token.left.value
            cmp_op = token.token_next(0)[1].value
            # FIXME: add comparison operator as well
            # data[identifier] = (cmp_op, [token.right.value])
            # data.append((identifier, cmp_op, [token.right.value]))
            data.append((identifier, cmp_op, [token.right.value], connector))
            identifier = ""
            in_predicate = False
            cmp_op = ""
        elif (type(token) == sqlparse.sql.Identifier):
            identifier = token.value
            in_predicate = True
        else:
            # parenthesis can start ANYWHERE
            if args.debug: print(type(token), str(token.ttype))
            if (type(token) == sqlparse.sql.Parenthesis):
                print(token)
                pdb.set_trace()
                # handle the stuff inside parenthesis separately
                handle_token_list(token, data, identifier, cmp_op)
                # this should always be true at the end
                in_predicate = False
                cmp_op = ""
                identifier = ""
                continue
            elif ("Punctuation" in str(token.ttype)):
                continue
            elif not in_predicate:
                # just valid for JOB
                assert (token.value == "AND" or token.value == "OR")
                connector = token.value
                continue
            elif (type(token) == sqlparse.sql.IdentifierList):
                # FIXME: deal with the identifier list!
                vals = token.get_identifiers()
                in_predicate = False
                vals = [v.value for v in vals]
                # data[identifier] = (cmp_op, vals)
                data.append((identifier, cmp_op, vals, connector))
                cmp_op = ""
                identifier = ""
                continue
            elif token.is_keyword:
                cmp_op += " " + token.value
            elif ("Literal" in str(token.ttype)):
                if args.debug: print(token.ttype, token)
                # deal with it.
                in_predicate = False
                values = []
                values.append(token.value)
                if ("BETWEEN" in cmp_op):
                    # this should be AND
                    index, token = token_list.token_next(index)
                    assert str(token.value) == "AND"
                    index, token = token_list.token_next(index)
                    # FIXME: handle this
                    values.append(token.value)

                data.append((identifier, cmp_op, values, connector))
                cmp_op = ""
                identifier = ""
                continue

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

        index, token = token.token_next(index)
        # print("find all tables next: ", token)
        if token.is_keyword:
            break
    # print("returning tables: ", tables)
    return tables


def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    # print("find next match ", print(index))
    match = ""
    index, token = wheres.token_next(index)
    if token is None:
        return None, None
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres)
    while True:
        if token is None:
            break
        if token.value == "AND":
            break
        # print("token.value: ", token.value)
        match += " " + token.value
        index, token = token_list.token_next(index)

    # print("match: ", match)
    for table in tables_in_pred:
        if table not in tables:
            return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        print(match, index)
        if index is None:
            break
        if match is not None:
            matched.append(match)

    # print(matched)
    return matched

# def find_all_clauses(tables, wheres):
    # matched = []
    # for left, op, right, connector in wheres:
        # left_match = False
        # right_match = False
        # # op = val[0]
        # # right = val[1]
        # assert left.count(".") == 1
        # # left match
        # for t in tables:
            # if " " + t + "." in " " + left:
                # left_match = True
        # if right[0].count(".") == 0:
            # right_match = True

        # elif right[0].count(".") == 1:
            # # join
            # for t in tables:
                # if " " + t + "." in " " + right[0]:
                    # right_match = True
        # else:
            # assert False
            # # print(left)
            # # print(op)
            # # print(right)
            # # pdb.set_trace()

        # if left_match and right_match:
            # if len(right) == 1:
                # cond = left + " " + op + " " + right[0]
            # elif " IN" == op:
                # # cond = left + " " + op + " " + "(" + str(right) + ")"
                # right_str = "("
                # for wi, w in enumerate(right):
                    # # right_str += "'" + w + "'"
                    # right_str += w
                    # if (wi != len(right) -1):
                        # right_str += ","

                # cond = left + " " + op + " " + right_str + ")"

            # elif "BETWEEN" in op:
                # cond = left + " " + " BETWEEN " + right[0] + " AND " + right[1]
            # else:
                # # FIXME:
                # continue

            # cond = connector + " " + cond
            # matched.append(cond)

    # return matched

def handle_query(tables, wheres):
    all_data = {}
    conn = pg.connect(host=args.db_host, database=args.db_name,
            user=args.user_name,
            password=args.pwd)
    cur = conn.cursor()

    combs = []
    for i in range(1, len(tables)+1):
        combs += itertools.combinations(tables.keys(), i)
    for comb in combs:
        tables_string = ""
        for i, c in enumerate(comb):
            tables_string += c + " AS " + tables[c]
            if i != len(comb)-1:
                tables_string += ","

        aliases = [tables[c] for c in comb]
        matches = find_all_clauses(aliases, wheres)
        cond_string = ""

        # need to ensure that each of the tables is prsent in atleast ONE join
        # if there are more than one tables
        # if len(aliases) > 1:
            # all_joins = True
            # for alias in aliases:
                # joined = False
                # for match in matches:
                    # if match.count(".") == 2:
                        # # FIXME: so hacky ugh.
                        # if (" " + alias + "." in " " + match):
                            # joined = True
                # if not joined:
                    # all_joins = False
                    # break
            # if not all_joins:
                # continue

        for i, m in enumerate(matches):
            cond_string += m
            if i != len(matches)-1:
                cond_string += " AND "

        if cond_string != "":
            cond_string = "WHERE " + cond_string

        used_tables = [c for c in comb]
        used_tables.sort()
        tkey = ""
        for usedt in used_tables:
            tkey += " " + usedt

        query = QUERY_TMP.format(TABLES=tables_string, CONDS=cond_string)
        print(query)
        pdb.set_trace()
        cur.execute("SET statement_timeout = {}".format(args.query_timeout))
        try:
            cur.execute(query)
            exp_output = cur.fetchall()
            count = int(exp_output[0][0])
        except:
            count = 100000000
            conn.close()
            cur.close()
            conn = pg.connect(host=args.db_host, database=args.db_name,
                    user=args.user_name,
                    password=args.pwd)
            cur = conn.cursor()

        # query = QUERY_TMP_ANALYZE.format(TABLES=tables_string, CONDS=cond_string)
        # cur.execute(query)
        # exp_output = cur.fetchall()
        # count = parse_explain(exp_output)

        all_data[tkey] = count
        print(query, tkey, count)
        if count == 0:
            pdb.set_trace()

    conn.close()
    cur.close()
    return all_data

def main():
    file_names = glob.glob("join-order-benchmark/*.sql")
    queries = []
    for fn in file_names:
        with open(fn, "rb") as f:
            queries.append(f.read())

    all_queries = {}
    for i, q in enumerate(queries):
        parsed = sqlparse.parse(q)[0]
        # let us go over all the where clauses
        where_clauses = None
        tables = {}
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
                for t in all_tables:
                    tables[t.split("AS")[0]] = t.split("AS")[1]

        # dumb, dumb hack
        if (len(tables) == 0):
            for token in parsed.tokens:
                if (type(token) == sqlparse.sql.IdentifierList):
                    all_tables = str(token).replace(",","").replace(" ","").split("\n")
                    for t in all_tables:
                        tables[t.split("AS")[0]] = t.split("AS")[1]

        assert where_clauses is not None
        # token_list = sqlparse.sql.TokenList(where_clauses)
        file_name = file_names[i]
        if "20a.sql" not in file_name:
            continue
        # where_clauses = str(where_clauses).replace("WHERE","").replace("AND","").replace(";","").split("\n")
        data = []
        # handle_token_list(token_list, data)

        all_queries[file_name] = (tables, where_clauses)

    all_counts = {}
    num = 0
    old_saved_data = {}
    if os.path.exists(args.output_name):
        with open(args.output_name, "r") as f:
            old_saved_data = json.loads(f.read())
        all_counts.update(old_saved_data)

    for k, query in all_queries.items():
        if "20a.sql" not in k:
            continue
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
    pdb.set_trace()

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

    return parser.parse_args()

args = read_flags()
main()
