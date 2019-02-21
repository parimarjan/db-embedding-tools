import numpy as np
import pdb
import argparse
import psycopg2 as pg
import psycopg2.extras
import random
import time
import subprocess as sp
from gensim.models import Word2Vec
import re
from utils.utils import *

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from utils.net import SimpleRegression
import torch

SAMPLE_VALS = [0, 5, 10, 12, 15]
CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})";
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s";

SAMPLES_SINGLE_TEMPLATE = "EXPLAIN ANALYZE SELECT COUNT(*) FROM {table} WHERE {column} {cmp_op} {val}"
SAMPLES_CORR_TEMPLATE = "EXPLAIN ANALYZE SELECT COUNT(*) FROM {table} WHERE {column1} {cmp_op1} \
{val1} AND {column2} {cmp_op2} {val2}"

# FIXME: pass in query directly here.
# should we be doing train_pairs?
WORDVEC_TEMPLATE = "python3 word2vec_embedding.py --sql \"{sql}\" \
--db_name {db_name} --data_dir {dir} \
--regen_sentences --no_preprocess_word --model_name {model_name} \
--embedding_size {emb_size} --min_count {min_count}"

EPSILON_SELECTIVITY = 0.001

# STR_CMP_OPS = ["=", "IN", "LIKE"]
STR_CMP_OPS = ["="]
NUM_CMP_OPS = ["=", "<=", ">=", "BETWEEN"]

class SelectivitySample():
    def __init__(self, column_names, vals, cmp_ops, sel, count, pg_sel=None):
        self.sel = sel
        self.count = count
        self.cmp_ops = cmp_ops
        self.column_names = column_names
        self.vals = []
        self.pg_sel = pg_sel
        # we don't need the apostrophe's - just needed it for executing on
        # postgres
        for v in vals:
            self.vals.append(v.replace("'", ""))

    def __str__(self):
        rep = ""
        for i, v in enumerate(self.column_names):
            rep += str(v) + self.cmp_ops[i] + str(self.vals[i])
            if (i != len(self.column_names)-1):
                rep += " AND "
            else:
                rep += " = "
        rep += str(self.sel) + ", "
        rep += str(self.count)
        return rep

    def get_features(self, model):
        features = []
        for i, v in enumerate(self.column_names):
            cmp_op = self.cmp_ops[i]
            for i in range(len(STR_CMP_OPS)):
                if STR_CMP_OPS[i] == cmp_op:
                    features.append(1.00)
                else:
                    features.append(0.00)

            val = self.vals[i]
            if val in model:
                features.extend(model[val])
            else:
                features.extend([0]*args.embedding_size)
        return features

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_len", type=int, required=False,
            default=100)
    parser.add_argument("--std_scale", type=int, required=False,
            default=3)
    parser.add_argument("--max_iter", type=int, required=False,
            default=100000)
    parser.add_argument("--min_count", type=int, required=False,
            default=10)
    parser.add_argument("--minibatch_size", type=int, required=False,
            default=32)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--embedding_size", type=int, required=False,
            default=10)
    parser.add_argument("--num_vals", type=int, required=False,
            default=100)
    parser.add_argument("--num_samples_per_op", type=int, required=False,
            default=100)
    parser.add_argument("--num_columns", type=int, required=False,
            default=4)
    parser.add_argument("--table_name", type=str, required=False,
            default="synth1")

    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--db_name", type=str, required=False,
            default="syntheticdb")
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/synthdb")
    parser.add_argument("--model_name", type=str, required=False,
            default="test.bin")
    parser.add_argument("--no_gen_data", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--corr", action="store_true")

    return parser.parse_args()

def get_columns(num_columns, column_type = "varchar"):
    col_header = ""
    for i in range(num_columns):
        col_name = "col" + str(i)
        col_header += col_name + " " + column_type
        if i != num_columns-1:
            # add comma
            col_header += ", "
    return col_header

# Generalized from:
#https://stackoverflow.com/questions/18683821/generating-random-correlated-x-and-y-points-using-numpy
def gen_data(range_ends, std_scale, corrs):
    # we needs means, ranges for each columns. corr for n-1 columns.
    # correlations for the first column with all others
    ranges = []
    means = []
    stds = []
    for r in range_ends:
        ranges.append(np.array(r))
    for r in ranges:
        means.append(r.mean())
        stds.append(r.std() / std_scale)

    # corr = 0.8         # correlation
    covs = np.zeros((len(ranges), len(ranges)))
    for i in range(len(ranges)):
        for j in range(len(ranges)):
            if i == j:
                covs[i][j] = stds[i]**2
            elif i == 0:
                # add the symmetric correlations according to corr variable
                corr = corrs[j]
                covs[i][j] = corr*stds[i]*stds[j]
                covs[j][i] = corr*stds[i]*stds[j]
            # just leave it as 0.
    print(covs)
    vals = np.random.multivariate_normal(means, covs, args.num_vals).T

    for i, v in enumerate(vals):
        vals[i] = [int(x) for x in v]

    return list(zip(*vals))

def gen_samples(cursor, dtype="str", single_col=True, corr=False):
    print("gen samples!")
    def parse_explain_analyze(output):
        est_vals = None
        num_vals = None
        # FIXME: generic for corr part as well.
        for line in output:
            line = line[0]
            if "Seq Scan" in line:
                for w in line.split():
                    if "rows" in w and est_vals is None:
                        est_vals = int(re.findall("\d+", w)[0])
                    elif "rows" in w:
                        num_vals = int(re.findall("\d+", w)[0])
        return est_vals, num_vals

    def sample_row_values(op, idx1, idx2, rows):
        req_rows = None
        if dtype == "str":
            vals = []
            for i in range(args.num_samples_per_op):
                row = random.choice(rows)
                val1 = row[idx1+1]
                val2 = row[idx2+1]
                val1 = "'" + val1.replace("'", "''") + "'"
                val2 = "'" + val2.replace("'", "''") + "'"
                vals.append((val1, val2))
            return vals
        else:
            assert False
            # req_rows = [r[idx+1] for r in rows]

        if op == "=":
            vals = []
            for i in range(args.num_samples_per_op):
                vals.append(random.choice(req_rows))
            return vals
        elif op == "IN":
            # need to sample N guys every time, and then put them in a list
            pass
        elif op == "LIKE":
            # need to sample a substring from one sample for each
            pass
        else:
            assert False

    def sample_values(op, idx, rows):
        req_rows = None
        if dtype == "str":
            req_rows = []
            for r in rows:
                val = r[idx+1]
                if val != None:
                    formatted_str = "'" + val.replace("'", "''") + "'"
                    req_rows.append(formatted_str)
            # req_rows = ["'"+ r[idx+1].replace("'","''") + "'" for r in rows]
        else:
            req_rows = [r[idx+1] for r in rows]

        if op == "=":
            vals = []
            for i in range(args.num_samples_per_op):
                vals.append(random.choice(req_rows))
            return vals
        elif op == "IN":
            # need to sample N guys every time, and then put them in a list
            pass
        elif op == "LIKE":
            # need to sample a substring from one sample for each
            pass
        else:
            assert False

    # get a random sample of values from each column so we can use it to
    # generate sensible queries
    Q = "SELECT * FROM {table} WHERE random() < 0.1".format(table=args.table_name)
    cursor.execute(Q)
    rows = cursor.fetchall()
    # can get column names from the cursor as well
    columns = [c.name for c in cursor.description]

    Q = "SELECT COUNT(*) FROM {table}".format(table=args.table_name)
    cursor.execute(Q)
    total_rows = float(cursor.fetchone()[0])
    print("got rows info")

    samples = []
    cmp_ops = None
    if dtype == "str":
        cmp_ops = STR_CMP_OPS
    else:
        cmp_ops = NUM_CMP_OPS

    start = time.time()
    if single_col:
        for i in range(args.num_columns):
            # col = "col" + str(i)
            col = columns[i+1]
            for op in cmp_ops:
                vals = sample_values(op, i, rows)
                start_vals = time.time()
                for vali, val in enumerate(vals):
                    if (vali % 100 == 0):
                        print(vali)
                    Q = SAMPLES_SINGLE_TEMPLATE.format(table=args.table_name,
                            column=col, cmp_op = op, val = val)
                    cursor.execute(Q)
                    exp_output = cursor.fetchall()
                    est_vals, num_vals = parse_explain_analyze(exp_output)

                    # add the sample for training
                    sample = SelectivitySample([i], [val], [op],
                            num_vals / float(total_rows), num_vals,
                            pg_sel=est_vals/float(total_rows))
                    samples.append(sample)

                print("executing vals took ", time.time() - start_vals)

    print("generating single col samples took ", time.time() - start)

    if corr:
        i = 0
        # col1 = "col" + str(i)
        col1 = columns[i+1]
        for j in range(args.num_columns):
            # allow j = 0 as well.
            if j == 0:
                continue
            # col2 = "col" + str(j)
            col2 = columns[j+1]
            for op in cmp_ops:
                # TODO: different operators on both sides
                # maybe make choosing the operator a random choice
                # vals1 = sample_values(op, i, rows)
                # vals2 = sample_values(op, j, rows)
                vals = sample_row_values(op, i, j, rows)
                print("sample row values done!")
                # FIXME: find a way to execute each of these together?
                # for k, val1 in enumerate(vals1):
                for k, (val1, val2) in enumerate(vals):
                    if k % 1000 == 0:
                        print(k)

                    # val2 = vals2[k]
                    Q = SAMPLES_CORR_TEMPLATE.format(table=args.table_name,
                            column1=col1, cmp_op1 = op, val1 = val1,
                            column2=col2, cmp_op2 = op, val2 = val2)
                    cursor.execute(Q)

                    exp_output = cursor.fetchall()
                    est_vals, num_vals = parse_explain_analyze(exp_output)
                    # add the sample for training
                    # samples.append((i, op, val, num_vals / total_rows, num_vals))
                    columns = [col1, col2]
                    vals = [val1, val2]
                    ops = [op, op]
                    sel = float(num_vals) / total_rows
                    pg_sel = float(est_vals) / total_rows
                    sample = SelectivitySample(columns, vals, ops, sel,
                            num_vals, pg_sel=pg_sel)
                    samples.append(sample)

    sels = [s.sel for s in samples]
    zeros = [s for s in samples if s.sel == 0.00]
    # print("max selectivity: ", max(sels))
    print("num zeros: ", len(zeros))

    return samples

def compute_relative_loss(yhat, ytrue):
    '''
    as in the quicksel paper.
    '''
    error = 0.00
    for i, y in enumerate(ytrue):
        yh = yhat[i]
        # diff = abs(y - yh)
        # if (diff > y):
            # print("true y: {}, yh: {}".format(y, yh))
            # pdb.set_trace()
        error += abs(y - yh) / (max(EPSILON_SELECTIVITY, y))
    error = error / len(yhat)
    return round(error * 100, 2)

def train_and_predict(train, test, wv):
    def get_preds(samples, net):
        # after training
        x = [s.get_features(wv) for s in samples]
        y = [s.sel for s in samples]
        x = to_variable(x).float()
        pred = net(x)
        pred = pred.squeeze(1).detach().numpy()
        return pred

    inp_len = len(train[0].get_features(wv))
    net = SimpleRegression(inp_len, inp_len*2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()
    num_iter = 0

    while True:
        # select samples from xtrain
        batch = random.sample(train, args.minibatch_size)
        xbatch = [s.get_features(wv) for s in batch]
        xbatch = to_variable(xbatch).float()
        ybatch = [s.sel for s in batch]
        ybatch = to_variable(ybatch).float()
        pred = net(xbatch)
        pred = pred.squeeze(1)
        loss = loss_func(pred, ybatch)

        if (num_iter % 1000 == 0):
            print("iter: {}, loss: {}".format(num_iter, loss.item()))
        if (num_iter > args.max_iter or loss <= 0.00001):
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_iter += 1

    ytrain = get_preds(train, net)
    yhat = get_preds(test, net)
    return ytrain, yhat

def main():
    # TODO: generalize this further.
    # create a table, if it doesn't exist
    columns = get_columns(args.num_columns)
    if args.verbose: print(columns)
    conn = pg.connect(host=args.db_host, database=args.db_name)
    if args.verbose: print("connection succeeded")
    # FIXME: if it already exists, delete it.
    cur = conn.cursor()
    # always creates the table and fills it up with data.
    cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=args.table_name))
    create_sql = CREATE_TABLE_TEMPLATE.format(name = args.table_name,
                                             columns = columns)
    cur.execute(create_sql)
    # RANGES = [[0, 100], [100, 200], [200, 300], [300, 400]]
    RANGES = []
    for i in range(args.num_columns):
        RANGES.append([i*args.period_len, i*args.period_len+args.period_len])

    CORR = [0.00, 0.9, 0.2, 0.00]
    assert args.num_columns == len(RANGES)
    if not args.no_gen_data:
        assert "synth" in args.db_name
        data = gen_data(RANGES, args.std_scale, CORR)
        print("generated data!")
        # insert statement
        insert_sql = INSERT_TEMPLATE.format(name = args.table_name,
                                            columns = columns.replace("varchar",
                                                ""))
        pg.extras.execute_values(cur, insert_sql, data, template=None,
                page_size=100)
        conn.commit()
        # let's run vacuum

    old_isolation_level = conn.isolation_level
    conn.set_isolation_level(0)
    query = "VACUUM ANALYZE"
    cur.execute(query);
    # self._doQuery(query)
    conn.set_isolation_level(old_isolation_level)
    conn.commit()

    # let us train the wordvec model on this data
    print("going to train model!")
    select_sql = "SELECT * FROM {};".format(args.table_name)
    wv_train = WORDVEC_TEMPLATE.format(db_name = args.db_name,
                            model_name = args.model_name,
                            dir = args.data_dir,
                            sql = select_sql,
                            emb_size = args.embedding_size,
                            min_count = args.min_count)
    p = sp.Popen(wv_train, shell=True)
    p.wait()
    print("finished training model!")
    # load model
    time.sleep(2)
    model = Word2Vec.load(args.data_dir + args.model_name)
    wv = model.wv
    del model

    # testing phase
    samples = None
    if args.corr:
        samples = gen_samples(cur, single_col=False, corr=True)
    else:
        samples = gen_samples(cur, single_col=True, corr=False)

    features = samples[0].get_features(wv)
    print(len(features), features)
    train, test = train_test_split(samples, test_size=args.test_size)
    xtrain = [s.get_features(wv) for s in train]
    ytrain = [s.sel for s in train]
    xtest = [s.get_features(wv) for s in test]
    ytest = [s.sel for s in test]

    # Note: don't think we need to use a scaler here?
    # scaler = StandardScaler()
    # scaler.fit(xtrain)
    # xtrain = scaler.transform(xtrain)
    # xtest = scaler.transform(xtest)

    ## need to generate yhat
    # clf = SVR(kernel='rbf', gamma='scale')
    # classifiers = [LinearRegression(), MLPRegressor()]
    # for clf in classifiers:
    # clf = LinearRegression()
    # clf = MLPRegressor(max_iter=20000, validation_fraction=0.0)
    # print(clf)
    # clf.fit(xtrain, ytrain)
    # yhat = clf.predict(xtest)

    yhat_train, yhat = train_and_predict(train, test, wv)

    ypg = np.array([s.pg_sel for s in test])
    pg_loss = compute_relative_loss(ypg, ytest)
    train_loss = compute_relative_loss(yhat_train, ytrain)
    test_loss = compute_relative_loss(yhat, ytest)

    print("abs loss: ", sum(abs(yhat-ytest)))
    print("abs loss pg: ", sum(abs(ypg-ytest)))

    print("samples: {} loss training: {} loss test: {} loss pg: {}".\
            format(len(yhat), train_loss, test_loss, pg_loss))

    pdb.set_trace()
    cur.close()
    conn.close()

args = read_flags()
main()
