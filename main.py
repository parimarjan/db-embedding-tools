import numpy as np
# import pdb
import ipdb as pdb
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
import hashlib
# import pickle
import dill as pickle

from scipy import integrate
from scipy.stats import multivariate_normal as mvn
from random import randint
import math
from utils.tf_summaries import TensorboardSummaries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

# PROB_ALG_NAMES = ["independent", "independent_pg", "wv_kahnemann",
        # "wv_kahnemann_pg", "corr_kahnemann", "corr_kahnemann_pg", "rho1"]
# PROB_ALG_NAMES = ["debug_est_correl", "debug_est_correl_pg"]
PROB_ALG_NAMES = []

ALGS = ["independent", "postgres", "analytic", "wv_rho_est",
            "nn-true-marg-onehot", "nn-pg-marg-onehot", "nn-sel",
            "nn-true-marg-wv"]

# ALGS = ["independent", "postgres", "analytic", "wv_rho_est",
            # "nn-true-marg-onehot", "nn-pg-marg-onehot", "nn-sel"]

# ALGS = ["postgres", "analytic", "wv_rho_est",
            # "nn-true-marg"]

# FIXME: data generation script
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
--embedding_size {emb_size} --min_count {min_count} \
{extra_flags}"

EPSILON = 0.001

# STR_CMP_OPS = ["=", "IN", "LIKE"]
STR_CMP_OPS = ["="]
NUM_CMP_OPS = ["=", "<=", ">=", "BETWEEN"]

# FIXME: general class for sample, table etc. with various precomputed
# statistical measures
class SelectivitySample():
    def __init__(self, column_names, vals, cmp_ops, sel, count,
            pg_sel=None, pg_marginal_sels=None, marginal_sels=None):
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
        self.marginal_sels = marginal_sels
        self.pg_marginal_sels = pg_marginal_sels

        # FIXME: how does this generalize to nd
        # if args.use_true_marginals:
            # pa = self.marginal_sels[0]
            # pb = self.marginal_sels[1]
        # else:
            # pa = self.pg_marginal_sels[0]
            # pb = self.pg_marginal_sels[0]

        # normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
        # sel = self.sel
        # self.true_rho = ((sel) - (pa*pb)) / normalizer

    def __str__(self):
        rep = ""
        rep += "columns: " + str(self.column_names) + "\n"
        rep += "operator: " + " = " + "\n"
        rep += "values: " + str(self.vals) + "\n"
        rep += "selectivity: " + str(self.sel) + "\n"
        rep += "pg estimate: " + str(self.pg_sel) + "\n"
        rep += "count: " + str(self.count) + "\n"
        rep += "marginal_sel: " + str(self.marginal_sels) + "\n"
        rep += "pg_marginal_sel: " + str(self.pg_marginal_sels)
        return rep

    def get_marginals(self, true_marginals=True):
        if true_marginals:
            return self.marginal_sels
        else:
            return self.pg_marginal_sels

    def get_features(self, model=None, featurization_scheme="wordvec"):

        features = []
        if featurization_scheme == "wordvec":
            assert model is not None
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
        elif featurization_scheme == "onehot":
            # FIXME: estimate total some other way.
            total = len(model.index2word)
            for i in range(len(self.column_names)):
                val = self.vals[i]
                onehot_vec = [0.00]*total
                if val in model:
                    idx = hash(val) % total
                    onehot_vec[idx] = 1.00
                # else, we just estimate it as the 0 vec
                features.extend(onehot_vec)

        return features


# FIXME: combine in utils
def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def gen_args_hash():
    data_params = str(args.period_len) + str(args.num_vals)
    data_params += str(args.num_samples_per_op)
    data_params += str(args.num_columns)
    data_params += str(args.table_name)
    data_params += str(args.db_name)
    data_params += str(args.correlations)
    data_params += str(args.use_corr)
    hsh = str(deterministic_hash(data_params))[0:4]
    return args.suffix + hsh

def get_correlations():
    return [float(item) for item in args.correlations.split(',')]

def gen_results_name():
    # FIXME:
    prefix = args.data_dir + "/" + args.db_name + args.table_name
    return prefix + "-results.pd"

def read_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument('--suffix', help='',
            type=str, required=False, default="")
    parser.add_argument("--featurization_scheme",
            type=str, required=False, default="onehot")

    parser.add_argument('--correlations', help='delimited list correlations',
            type=str, required=False, default="1.0,0.0")
    # values for data generation.
    parser.add_argument("--period_len", type=int, required=False,
            default=10)
    parser.add_argument("--seed", type=int, required=False,
            default=1234)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--lr", type=float, required=False,
            default=0.00001)

    parser.add_argument("--num_vals", type=int, required=False,
            default=1000)
    parser.add_argument("--num_samples_per_op", type=int, required=False,
            default=100)
    parser.add_argument("--num_columns", type=int, required=False,
            default=2)
    parser.add_argument("--table_name", type=str, required=False,
            default="synth1")
    parser.add_argument("--gen_data_distr", type=str, required=False,
            default="gaussian")

    ## neural net training vals
    parser.add_argument("--std_scale", type=int, required=False,
            default=3)
    parser.add_argument("--max_iter", type=int, required=False,
            default=1000000)
    parser.add_argument("--min_count", type=int, required=False,
            default=10)
    parser.add_argument("--minibatch_size", type=int, required=False,
            default=32)
    parser.add_argument("--embedding_size", type=int, required=False,
            default=10)
    parser.add_argument("--db_name", type=str, required=False,
            default="syntheticdb")

    ## other vals
    parser.add_argument("--db_host", type=str, required=False,
            default="localhost")
    parser.add_argument("--data_dir", type=str, required=False,
            default="/data/pari/embeddings/synthdb/")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--gen_data", type=int, required=False,
            default=0)
    parser.add_argument("--use_true_marginals", type=int, required=False,
            default=1)
    parser.add_argument("--train_wv", type=int, required=False,
            default=0)
    # parser.add_argument("--rho_est", type=int, required=False,
            # default=1)
    parser.add_argument("--adaptive_lr", type=int, required=False,
            default=1)
    parser.add_argument("--algs", type=str, required=False,
            default=None)

    parser.add_argument("--store_results", action="store_true")
    parser.add_argument("--use_corr", action="store_true")

    return parser.parse_args()

# TODO: write general purpose integrand function for more than 2 variables
# def integrand(xs, mean, cov):
    # '''
    # @ xs: array of xs.
    # '''
    # len(xs) == len(mean) == len((cov)[0])
    # return multivariate_normal.pdf(xs, mean=mean, cov=cov)

def integrand2(x1,x2, mean, cov):
    '''
    @ xs: array of xs.
    '''
    return mvn.pdf([x1,x2], mean=mean, cov=cov)

def solve_analytically(mean, cov, ranges):

    # https://stackoverflow.com/questions/49433077/integration-of-multivariate-normal-distribution-in-python
    # Note: scipy's integration only does integration over bounded domains
    res, err = integrate.nquad(integrand2,
                               ranges,
                               args=(mean, cov))
    return res, err

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
def gen_data(means, covs):
    if args.gen_data_distr == "gaussian":
        vals = np.random.multivariate_normal(means, covs, args.num_vals).T
        for i, v in enumerate(vals):
            vals[i] = [int(x) for x in v]
    elif args.gen_data_distr == "uniform_linearly_dependent":
        vals = []
        x = [randint(0, args.period_len) for p in range(0, args.num_vals)]
        y = []
        for xi in x:
            y.append(xi * 10 + 50)

        vals.append(x)
        vals.append(y)
    elif args.gen_data_distr == "uniform_noise":
        vals = []
        x = [randint(0, args.period_len) for p in range(0, args.num_vals)]
        y = []
        for xi in x:
            # y.append(xi * 10 + random.randint(0,5)*5)
            y.append(xi * 10 + random.randint(0,5))

        vals.append(x)
        vals.append(y)

    return list(zip(*vals))

def gen_samples(cursor, dtype="str", single_col=True, corr=False):
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

    def get_selectivity_single_predicate(col, val, op):
        if col+val in selectivity_lookup_table:
            est_vals, num_vals = selectivity_lookup_table[col+val]
            return est_vals, num_vals

        Q = SAMPLES_SINGLE_TEMPLATE.format(table=args.table_name,
                column=col, cmp_op = op, val = val)
        cursor.execute(Q)
        exp_output = cursor.fetchall()
        est_vals, _ = parse_explain_analyze(exp_output)
        EXACT_COUNT_Q = Q.replace("EXPLAIN ANALYZE ", "")
        cursor.execute(EXACT_COUNT_Q)
        num_vals = cursor.fetchall()[0][0]

        # for future!
        selectivity_lookup_table[col+val] = (est_vals, num_vals)

        return est_vals, num_vals

    # FIXME: generalize to n-predicates
    def get_selectivity_two_predicate(col1, col2, val1, val2, op):
        # FIXME: generalize to n predicates
        key = col1+val1+col2+val2
        if key in selectivity_lookup_table:
            est_vals, num_vals = selectivity_lookup_table[key]
            return est_vals, num_vals

        Q = SAMPLES_CORR_TEMPLATE.format(table=args.table_name,
                column1=col1, cmp_op1 = op, val1 = val1,
                column2=col2, cmp_op2 = op, val2 = val2)
        cursor.execute(Q)
        exp_output = cursor.fetchall()
        est_vals, _ = parse_explain_analyze(exp_output)
        EXACT_COUNT_Q = Q.replace("EXPLAIN ANALYZE ", "")
        cursor.execute(EXACT_COUNT_Q)
        num_vals = cursor.fetchall()[0][0]
        selectivity_lookup_table[key] = (est_vals, num_vals)
        return est_vals, num_vals

    selectivity_lookup_table = {}
    # get a random sample of values from each column so we can use it to
    # generate sensible queries
    Q = "SELECT * FROM {table} WHERE random() < 0.001".format(table=args.table_name)
    cursor.execute(Q)
    rows = cursor.fetchall()
    # can get column names from the cursor as well
    columns = [c.name for c in cursor.description]

    Q = "SELECT COUNT(*) FROM {table}".format(table=args.table_name)
    cursor.execute(Q)
    total_rows = float(cursor.fetchone()[0])

    samples = []
    cmp_ops = None
    if dtype == "str":
        cmp_ops = STR_CMP_OPS
    else:
        cmp_ops = NUM_CMP_OPS

    start = time.time()
    # FIXME: not sure if single_col stuff even works now
    if single_col:
        for i in range(args.num_columns):
            # col = "col" + str(i)
            # +1 to avoid the id column
            col = columns[i+1]
            for op in cmp_ops:
                vals = sample_values(op, i, rows)
                for vali, val in enumerate(vals):
                    if (vali % 100 == 0):
                        print(vali)

                    est_vals, num_vals = get_selectivity_single_predicate(col,
                            val, op)

                    # add the sample for training
                    sample = SelectivitySample([col], [val], [op],
                            num_vals / float(total_rows), num_vals,
                            pg_sel=est_vals/float(total_rows))
                    samples.append(sample)

    print("generating single col samples took ", time.time() - start)

    ## FIXME: generalize this to multiple columns
    if not corr:
        return samples

    start = time.time()
    i = 0
    col1 = columns[i+1]
    for j in range(args.num_columns):
        # allow j = 0 as well.
        if j == 0:
            continue
        col2 = columns[j+1]
        for op in cmp_ops:
            # TODO: different operators on both sides
            # maybe make choosing the operator a random choice
            # vals2 = sample_values(op, j, rows)
            vals = sample_row_values(op, i, j, rows)
            for k, (val1, val2) in enumerate(vals):
                if k % 100 == 0:
                    print("generated {} samples".format(k))
                # first, do it for each of the single values
                est_val1, num_val1 = get_selectivity_single_predicate(col1,
                        val1, op)
                est_val2, num_val2 = get_selectivity_single_predicate(col2,
                        val2, op)
                pgsel1 = est_val1 / float(total_rows)
                pgsel2 = est_val2 / float(total_rows)
                sel1 = num_val1 / float(total_rows)
                sel2 = num_val2 / float(total_rows)
                pg_marginal_sels = [pgsel1, pgsel2]
                marginal_sels = [sel1, sel2]

                # both predicates together
                est_vals, num_vals = get_selectivity_two_predicate(col1,
                        col2, val1, val2, op)
                columns = [col1, col2]
                vals = [val1, val2]
                ops = [op, op]
                sel = float(num_vals) / total_rows
                pg_sel = float(est_vals) / total_rows

                sample = SelectivitySample(columns, vals, ops, sel,
                        num_vals, pg_sel=pg_sel,
                        pg_marginal_sels=pg_marginal_sels,
                        marginal_sels=marginal_sels)
                samples.append(sample)

    print("generating two col samples took ", time.time() - start)

    sels = [s.sel for s in samples]
    zeros = [s for s in samples if s.sel == 0.00]
    print("max selectivity: ", max(sels))
    print("num zeros: ", len(zeros))

    return samples

def training_loss_selectivity(pred, ytrue, batch=None, rho_est=True,
        use_true_marginals=True):
    '''
    Loss function for neural network training. Should use the
    compute_relative_loss formula, but deal with appropriate pytorch types.
    '''
    if rho_est:
        # here pred are the rho values
        assert batch is not None
        pa = to_variable([s.get_marginals(use_true_marginals)[0] \
                    for s in batch]).float()
        pb = to_variable([s.get_marginals(use_true_marginals)[1] \
                    for s in batch]).float()
        rho = pred
        normalizer = torch.sqrt(pa*(1-pa)*pb*(1-pb))
        pred = rho*normalizer + pa*pb

    # this part is the same for both rho_est, or directly selectivity
    # estimation cases
    epsilons = to_variable([EPSILON]*len(pred)).float()
    errors = torch.abs(pred-ytrue) / (torch.max(epsilons, ytrue))
    error = (errors.sum() * 100.00) / len(pred)
    # FIXME: should we just return error, or average error. does it even
    # matter?
    return error

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
        error += abs(y - yh) / (max(EPSILON, y))
    error = error / len(yhat)
    return round(error * 100, 3)

def predict_prob(X, wv, alg="independent", ytrue=None):
    '''
    just for X having samples with 2 variables
    @alg:
        - independent
        - independent_pg
        - wv_pg_kahnemann
        - wv_true_kahnemann
    '''
    yhat = []
    losses = []
    for i, sample in enumerate(X):
        marginals = sample.get_marginals(args.use_true_marginals)
        if alg == "independent":
            # res = marginals[0] * marginals[1]
            res = sample.marginal_sels[0] * sample.marginal_sels[1]
        elif alg == "wv_rho_est":
            pa = marginals[0]
            pb = marginals[1]
            try:
                rho = wv.similarity(sample.vals[0], sample.vals[1])
            except:
                # if we don't have it in the model, then assume 0
                rho = 0
            normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
            res = rho*normalizer + pa*pb
        elif alg == "debug_est_correl":
            pa = marginals[0]
            pb = marginals[1]
            normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
            val1 = float(sample.vals[0])
            val2 = float(sample.vals[1])
            if (val1 + 10.0 == val2):
                # rho = get_correlations()[1]
                rho = get_correlations()[1] / 3.0
            else:
                rho = 0.0
            res = rho*normalizer + pa*pb
        elif alg == "corr_rho":
            pa = marginals[0]
            pb = marginals[1]
            rho = get_correlations()[1]
            normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
            res = rho*normalizer + pa*pb
        elif alg == "debug":
            pa = marginals[0]
            pb = marginals[1]
            normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
            sel = sample.sel
            assert sel == ytrue[i]
            rho = ((sel) - (pa*pb)) / normalizer
            print(sample.vals[0], sample.vals[1])
            print("rho: ", rho)
            print(sample)
            print("similarity: ", wv.similarity(str(sample.vals[0]), str(sample.vals[1])))
            pdb.set_trace()
            res = rho*normalizer + pa*pb
        else:
            assert False, "must be one of these algs"
        yhat.append(res)

        if ytrue is not None:
            loss = abs(res - ytrue[i])
            losses.append(loss)
            # print(sample)
            # print("estimate: ", res)
            # print("sample {}, loss: {} ".format(i, loss))
            # print("pred: {}, true: {}, loss: {}".format(yhat[i],
                # ytrue[i], loss))

    print("max loss: ", sorted(losses)[-1])
    print("max loss2: ", sorted(losses)[-2])
    return np.array(yhat)

def predict_analytically(X, ytrue=None):
    means, covs = get_gaussian_data_params()
    yhat = []
    losses = []
    # FIXME: generalize this to handle multiple dimensions
    for i, sample in enumerate(X):
        ranges = [None]*len(means)
        for vali, column in enumerate(sample.column_names):
            column = int(column.replace("col", ""))
            val = float(sample.vals[vali])
            ranges[column] = [val, val+1.00]

        for vali, r in enumerate(ranges):
            if r is None:
                lower_bound = 0.0
                upper_bound = lower_bound + 2*(vali+1)*args.period_len
                ranges[vali] = [lower_bound, upper_bound]

        res, err = solve_analytically(means, covs, ranges)

        # FIXME: this doesn't seem to work.
        # dist = mvn(mean=means, cov=covs)
        # lower = list(zip(*ranges))[0]
        # upper = list(zip(*ranges))[1]
        # res_cdf = dist.cdf(upper) - dist.cdf(lower)
        # print("old res: ", res_old)
        # print("res: ", res)

        # FIXME: can we use err better?
        yhat.append(res)
        if ytrue is not None:
            loss = abs(res - ytrue[i])
            losses.append(loss)
            # print("pred: {}, true: {}, loss: {}, err: {}".format(yhat[i],
                # ytrue[i], loss, err))
            # print(sample)

    print("max loss: ", sorted(losses)[-1])
    print("max loss2: ", sorted(losses)[-2])
    return np.array(yhat)

# FIXME: shouldn't need wv here.
def train_and_predict_pytorch(train, test, wv, featurization_scheme="onehot",
        use_true_marginals=True, rho_est=True, alg_name="pytorch"):
    '''
    TODO:
        - print test loss periodically too.
        -
    '''
    def get_preds(samples, net):
        # after training, get predictions for selectivities
        x = [s.get_features(wv, featurization_scheme) for s in samples]
        x = to_variable(x).float()
        pred = net(x)
        pred = pred.squeeze(1).detach().numpy()
        if rho_est:
            # pred's represent the rho predictions. convert these to
            # predictions for selectivities
            pa = np.array([s.get_marginals(use_true_marginals)[0] \
                        for s in samples])
            pb = np.array([s.get_marginals(use_true_marginals)[1] \
                        for s in samples])
            rho = pred
            normalizer = np.sqrt(pa*(1-pa)*pb*(1-pb))
            pred = rho*normalizer + pa*pb

        return pred

    print("number of training examples: ", len(train))
    name = args.db_name + args.table_name + "-" + alg_name
    tfboard = TensorboardSummaries(args.data_dir + "/tflogs/" + name +
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    # tfboard.add_variables([
        # 'mb-loss'], 'train')

    tfboard.add_variables([
        'train-loss', 'test-loss', 'lr'], 'test')

    tfboard.init()

    # used for calculating loss periodically
    ytrain = [s.sel for s in train]
    ytest = [s.sel for s in test]

    inp_len = len(train[0].get_features(wv, featurization_scheme))
    net = SimpleRegression(inp_len, inp_len*2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    # update learning rate
    if args.adaptive_lr:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5,
                        verbose=True, eps=1e-10)

    # loss_func = torch.nn.MSELoss()
    loss_func = training_loss_selectivity
    num_iter = 0

    last_checkpoint_loss = 100000.00
    while True:
        # select samples from xtrain
        batch = random.sample(train, args.minibatch_size)
        # FIXME: debug
        # batch = train

        xbatch = [s.get_features(wv, featurization_scheme) for s in batch]
        xbatch = to_variable(xbatch).float()

        ybatch = [s.sel for s in batch]
        ybatch = to_variable(ybatch).float()
        pred = net(xbatch)
        pred = pred.squeeze(1)
        loss = loss_func(pred, ybatch, batch=batch,
                use_true_marginals=use_true_marginals, rho_est=rho_est)

        # tfboard.report(num_iter,
            # [loss.item()], 'train')

        if (num_iter % 1000 == 0):
            ytrainhat = get_preds(train, net)
            ytesthat = get_preds(test, net)
            # if (num_iter > 5000):
                # for si, sample in enumerate(train):
                    # print(sample.vals)
                    # print("true: {}, pred: {}".format(sample.sel, ytrainhat[si]))
                    # print("abs diff: ", abs(ytrain[si]-ytrainhat[si]))
                    # print("loss: ", compute_relative_loss([ytrainhat[si]], [ytrain[si]]))

            train_loss = compute_relative_loss(ytrainhat, ytrain)
            test_loss = compute_relative_loss(ytesthat, ytest)
            if (num_iter % 100 == 0):
                print("alg: {}, table: {}, iter: {}, mb-loss: {}, train-loss: {}, test-loss: {}" \
                        .format(alg_name, args.table_name, num_iter, loss.item(),
                            train_loss, test_loss))
            if args.adaptive_lr:
                # FIXME: should we do this for minibatch / or for train loss?
                scheduler.step(train_loss)

            cur_lr = optimizer.param_groups[0]['lr']
            tfboard.report(num_iter,
                [train_loss, test_loss, cur_lr], 'test')

            if train_loss < 2.00:
                print("breaking because training loss less than 2")
                break

            # if you haven't improved in 50,000 iterations, then stop trying
            if (num_iter % 50000 == 0):
                if (abs(last_checkpoint_loss - train_loss) < 3.00):
                    print("breaking because improvement plateaued")
                    break
                last_checkpoint_loss = train_loss

        if (num_iter > args.max_iter):
            print("breaking because max iter done")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_iter += 1

    ytrainhat = get_preds(train, net)
    ytesthat = get_preds(test, net)
    return ytrainhat, ytesthat

def get_gaussian_data_params():
    RANGES = []
    for i in range(args.num_columns):
        RANGES.append([i*args.period_len, i*args.period_len+args.period_len])

    corrs = get_correlations()

    # part 1: generate real data
    assert args.num_columns == len(RANGES)
    # we needs means, ranges for each columns. corr for n-1 columns.
    # correlations for the first column with all others
    ranges = []
    means = []
    stds = []
    for r in RANGES:
        ranges.append(np.array(r))
    for r in ranges:
        means.append(r.mean())
        stds.append(r.std() / args.std_scale)
    # corr = 0.8         # correlation
    covs = np.zeros((len(ranges), len(ranges)))
    for i in range(len(ranges)):
        for j in range(len(ranges)):
            if i == j:
                covs[i][j] = stds[i]**2
            elif i == 0 and j != 0:
                # add the symmetric correlations according to corr variable
                corr = corrs[j]
                covs[i][j] = corr*stds[i]*stds[j]
                covs[j][i] = corr*stds[i]*stds[j]
            # just leave it as 0.
    return means, covs

def main():
    columns = get_columns(args.num_columns)
    if args.verbose: print(columns)
    conn = pg.connect(host=args.db_host, database=args.db_name)
    if args.verbose: print("connection succeeded")
    cur = conn.cursor()

    # FIXME: this only makes sense in gaussian data case, so separate out
    # data-gen stuff / stats collection stuff etc.
    # check if table exists or not. if it does not exist, then create it, and
    # gen data.
    exists = check_table_exists(cur, args.table_name)

    if args.gen_data or not exists:
        means, covs = get_gaussian_data_params()
        # if gen_data isn't on, then presumably, the table SHOULD already
        # contain data.
        assert "synth" in args.db_name
        # always creates the table and fills it up with data.
        cur.execute("DROP TABLE IF EXISTS {TABLE}".format(TABLE=args.table_name))
        create_sql = CREATE_TABLE_TEMPLATE.format(name = args.table_name,
                                                 columns = columns)
        cur.execute(create_sql)

        # FIXME: again, only makes sense for gaussian case.
        data = gen_data(means, covs)
        # insert statement
        insert_sql = INSERT_TEMPLATE.format(name = args.table_name,
                                            columns = columns.replace("varchar",
                                                ""))
        pg.extras.execute_values(cur, insert_sql, data, template=None,
                page_size=100)
        conn.commit()
        print("generate and inserted new data!")


    # let's run vacuum to update stats.
    db_vacuum(conn, cur)

    # FIXME: not sure how much we even care about having the word2vec model.
    model_name = "wordvec" + gen_args_hash() + ".bin"
    # if the model exists already, then we can just load it.
    if not os.path.exists(args.data_dir + model_name) or args.gen_data or args.train_wv:
        # let us train the wordvec model on this data
        print("going to train model!")
        select_sql = "SELECT * FROM {};".format(args.table_name)
        wv_train = WORDVEC_TEMPLATE.format(db_name = args.db_name,
                                model_name = model_name,
                                dir = args.data_dir,
                                sql = select_sql,
                                emb_size = args.embedding_size,
                                min_count = args.min_count,
                                extra_flags = "")
                                # extra_flags = "--synthetic_db_debug")
        p = sp.Popen(wv_train, shell=True)
        p.wait()
        print("finished training model!")
        # load model
        time.sleep(2)
    else:
        print("going to load the saved word2vec model")
    model = Word2Vec.load(args.data_dir + model_name)
    wv = model.wv
    del model # frees memory

    # both correlated samples, or single column samples will be stored
    # differently.
    samples_fname = args.data_dir + "/samples-" + gen_args_hash() + ".pickle"
    print("samples fname: ", samples_fname)
    if os.path.exists(samples_fname) and not args.gen_data:
        with open(samples_fname, "rb") as f:
            samples = pickle.loads(f.read())
        # need to redo this stuff because dill doesn't generate the class
        # properly, even though it regenerates the data. So for things like the
        # __str__ method to work, we need to explicitly recreate the classes.
        for i, s in enumerate(samples):
            samples[i] = SelectivitySample(s.column_names, s.vals, s.cmp_ops,
                    s.sel, s.count, s.pg_sel, s.pg_marginal_sels, s.marginal_sels)

    else:
        samples = gen_samples(cur, single_col=(not args.use_corr),
                corr=args.use_corr)
        with open(samples_fname, "wb") as f:
            f.write(pickle.dumps(samples))

    train, test = train_test_split(samples, test_size=args.test_size,
            random_state=args.seed)

    # note: not all the algorithms we consider even use the training set. Thus,
    # algorithms are only evaluated on the test set.
    ytrain = [s.sel for s in train]
    ytest = [s.sel for s in test]

    result = {}
    result["dbname"] = args.db_name
    result["table_name"] = args.table_name
    result["samples"] = len(ytest)

    for alg in ALGS:
        if args.algs is not None:
            if alg not in args.algs:
                continue
        print("running ", alg)

        # FIXME: combine all the attempts below into single function
        if "nn" in alg:
            if alg == "nn-true-marg-onehot":
                feat_scheme = "onehot"
                marginals = True
                rho_est = True
            elif alg == "nn-pg-marg-onehot":
                feat_scheme = "onehot"
                marginals = False
                rho_est = True
            elif alg == "nn-true-marg-wv":
                feat_scheme = "wordvec"
                marginals = True
                rho_est = True
            elif alg == "nn-sel":
                feat_scheme = "onehot"
                marginals = True
                rho_est = False
            else:
                assert False, "not known nn alg"

            _, yhat = train_and_predict_pytorch(train, test, wv,
                    featurization_scheme=feat_scheme,
                    use_true_marginals=marginals,
                    rho_est=rho_est, alg_name=alg)

        elif "postgres" in alg:
            # already pre-computed
            yhat = np.array([s.pg_sel for s in test])
        elif "analytic" in alg:
            # FIXME: currently, this only makes sense for gaussian data
            if not "gaussian" == args.gen_data_distr:
                continue
            yhat = predict_analytically(test, ytrue=ytest)
        else:
            yhat = predict_prob(test, wv, alg=alg, ytrue=ytest)

        test_loss = compute_relative_loss(yhat, ytest)
        print("case: {}, alg: {}, samples: {}, abs_loss: {}, relative_loss: {}"\
                .format(args.table_name, alg, len(yhat), sum(abs(yhat-ytest)), test_loss))

        # store the results
        result[alg] = test_loss

    df = pd.DataFrame([result])
    if args.store_results:
        file_name = gen_results_name()
        # load file first
        orig_df = load_object(file_name)
        if orig_df is None:
            new_df = df
        else:
            new_df = orig_df.append(df, ignore_index=True)
        save_object(file_name, new_df)

    cur.close()
    conn.close()

args = read_flags()
main()
