import numpy as np
np.set_printoptions(suppress=True)
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
from utils.db_utils import *

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
from cardinality_estimation.algs import *

ALGS = ["independent", "postgres", "analytic", "wv_rho_est",
            "nn-true-marg-onehot", "nn-pg-marg-onehot", "nn-sel",
            "nn-true-marg-wv", "svd-sel", "svd-rho"]

# FIXME: data generation script
CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})";
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s";

# FIXME: pass in query directly here.
# should we be doing train_pairs?
WORDVEC_TEMPLATE = "python3 word2vec_embedding.py --sql \"{sql}\" \
--db_name {db_name} --data_dir {dir} \
--regen_sentences --no_preprocess_word --model_name {model_name} \
--embedding_size {emb_size} --min_count {min_count} \
{extra_flags}"

EPSILON = 0.001

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
    prefix = args.data_dir + "/" + args.db_name + args.table_name + args.algs
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
    parser.add_argument("--model_type", type=str, required=False,
            default="pmi-svd")

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
    parser.add_argument("--svd-size", type=int, required=False,
            default=5)

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
            default=1)
    # parser.add_argument("--rho_est", type=int, required=False,
            # default=1)
    parser.add_argument("--adaptive_lr", type=int, required=False,
            default=1)
    parser.add_argument("--algs", type=str, required=False,
            default=None)
    parser.add_argument("--column_names", type=str, required=False,
            default="col0,col1")

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
        error += abs(y - yh) / (max(EPSILON, y))
    error = error / len(yhat)
    return round(error * 100, 3)

def compute_abs_loss(yhat, ytrue):
    error = np.sum(np.abs(np.array(yhat) - np.array(ytrue)))
    error = error / len(yhat)
    error = error * 100
    return round(error, 3)

def compute_qerror(yhat, ytrue):
    error = 0.00
    for i, y in enumerate(ytrue):
        yh = yhat[i]
        cur_error = max((yh / y), (y / yh))
        # if cur_error > 1.00:
            # print(cur_error, y, yh)
            # pdb.set_trace()
        error += cur_error
    error = error / len(yhat)
    return round(error, 3)

def compute_weighted_error(yhat, ytrue, alpha=0.1, beta=0.1):
    pass

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

        # FIXME: can we use err better?
        yhat.append(res)
        if ytrue is not None:
            loss = abs(res - ytrue[i])
            losses.append(loss)
            # print("pred: {}, true: {}, loss: {}, err: {}".format(yhat[i],
                # ytrue[i], loss, err))
            # print(sample)

    return np.array(yhat)

# FIXME: shouldn't need embedding_model here.
def train_and_predict_pytorch(train, test, embedding_model, featurization_scheme="onehot",
        use_true_marginals=True, rho_est=True, alg_name="pytorch"):
    '''
    TODO:
        - print test loss periodically too.
        -
    '''
    def get_preds(samples, net):
        # after training, get predictions for selectivities
        x = [s.get_features(embedding_model, featurization_scheme) for s in samples]
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

    tfboard.add_variables([
        'train-loss', 'lr'], 'test')

    tfboard.init()

    # used for calculating loss periodically
    ytrain = [s.sel for s in train]
    ytest = [s.sel for s in test]

    inp_len = len(train[0].get_features(embedding_model, featurization_scheme))
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

        xbatch = [s.get_features(embedding_model, featurization_scheme) for s in batch]
        xbatch = to_variable(xbatch).float()

        ybatch = [s.sel for s in batch]
        ybatch = to_variable(ybatch).float()
        pred = net(xbatch)
        pred = pred.squeeze(1)
        loss = loss_func(pred, ybatch, batch=batch,
                use_true_marginals=use_true_marginals, rho_est=rho_est)

        if (num_iter % 1000 == 0):
            ytrainhat = get_preds(train, net)
            ytesthat = get_preds(test, net)

            train_loss = compute_relative_loss(ytrainhat, ytrain)
            # if (num_iter % 100 == 0):
                # print("alg: {}, table: {}, iter: {}, mb-loss: {}, train-loss: {}" \
                        # .format(alg_name, args.table_name, num_iter, loss.item(),
                            # train_loss))

            if args.adaptive_lr:
                # FIXME: should we do this for minibatch / or for train loss?
                scheduler.step(train_loss)

            cur_lr = optimizer.param_groups[0]['lr']

            tfboard.report(num_iter,
                [train_loss, cur_lr], 'test')

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

def get_alg(alg):
    if alg == "independent":
        return Independent()
    elif alg == "analytic":
        return Analytic()
    elif alg == "postgres":
        return Postgres()
    elif alg == "svd-sel":
         return SVDK(k=args.svd_size, model_type="svd-sel",
                 true_marginals=args.use_true_marginals)
    else:
        assert False

def main():
    columns = get_columns(args.num_columns)
    if args.verbose: print(columns)
    conn = pg.connect(host=args.db_host, database=args.db_name)
    if args.verbose: print("connection succeeded")
    cur = conn.cursor()
    db_vacuum(conn, cur)

    # FIXME: this only makes sense in gaussian data case, so separate out
    # data-gen stuff / stats collection stuff etc.
    # check if table exists or not. if it does not exist, then create it, and
    # gen data.
    exists = check_table_exists(cur, args.table_name)
    print(args.gen_data)
    print(exists)
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

    table_stats = TableStats(args.db_name, args.table_name, db_host =
            args.db_host, columns_string=args.column_names)

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
        samples = table_stats.get_samples(num_samples=args.num_samples_per_op,
                                          num_columns = 2)
        with open(samples_fname, "wb") as f:
            f.write(pickle.dumps(samples))

    if args.train_wv:
        model_name = "wordvec" + gen_args_hash() + ".bin"
        # if the model exists already, then we can just load it.
        # if not os.path.exists(args.data_dir + model_name) or args.gen_data or args.train_wv:
        if not os.path.exists(args.data_dir + model_name):
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
        model = Word2Vec.load(args.data_dir + model_name)
        embedding_model = model.wv
        del model # frees memory
    else:
        assert False

    train, test = train_test_split(samples, test_size=args.test_size,
            random_state=args.seed)

    # note: not all the algorithms we consider even use the training set. Thus,
    # algorithms are only evaluated on the test set.
    ytrain = [s.sel for s in train]
    ytest = np.array([s.sel for s in test])

    result = {}
    result["dbname"] = args.db_name
    result["table_name"] = args.table_name
    result["samples"] = len(ytest)

    algorithms = []
    for alg_name in args.algs.split(","):
        algorithms.append(get_alg(alg_name))

    for alg in algorithms:
        alg.train(table_stats, train)
        yhat = alg.test(test)
        test_rel_loss = compute_relative_loss(yhat, ytest)
        test_abs_loss = compute_abs_loss(yhat, ytest)
        test_qloss = compute_qerror(yhat, ytest)

        print("case: {}, alg: {}, samples: {}, abs_loss: {}, relative_loss: {}"\
                .format(args.table_name, alg, len(yhat), sum(abs(yhat-ytest)),
                    test_rel_loss))
        print("rel: {}, abs: {}, qerr: {}".format(test_rel_loss, test_abs_loss,
            test_qloss))

        # store the results
        result[str(alg) + "rel_loss"] = test_rel_loss
        result[str(alg) + "abs_loss"] = test_abs_loss
        result[str(alg) + "qloss"] = test_qloss

    # for alg in ALGS:
        # if args.algs is not None:
            # if alg not in args.algs:
                # continue
        # print("running ", alg)

        # # FIXME: combine all the attempts below into single function
        # if "nn" in alg:
            # if alg == "nn-true-marg-onehot":
                # feat_scheme = "onehot"
                # marginals = True
                # rho_est = True
            # elif alg == "nn-pg-marg-onehot":
                # feat_scheme = "onehot"
                # marginals = False
                # rho_est = True
            # elif alg == "nn-true-marg-wv":
                # feat_scheme = "wordvec"
                # marginals = True
                # rho_est = True
            # elif alg == "nn-sel":
                # feat_scheme = "onehot"
                # marginals = True
                # rho_est = False
            # else:
                # continue

            # _, yhat = train_and_predict_pytorch(train, test, embedding_model,
                    # featurization_scheme=feat_scheme,
                    # use_true_marginals=marginals,
                    # rho_est=rho_est, alg_name=alg)

        # elif "postgres" in alg:
            # # already pre-computed
            # yhat = np.array([s.pg_sel for s in test])
        # elif "analytic" in alg:
            # # FIXME: currently, this only makes sense for gaussian data
            # if not "gaussian" == args.gen_data_distr:
                # continue
            # yhat = predict_analytically(test, ytrue=ytest)
        # elif "svd-rho" == alg:
            # cur_alg = SVDK(k=args.svd_size, model_type="svd-rho", true_marginals=args.use_true_marginals)
            # cur_alg.train(table_stats, train)
            # yhat = cur_alg.test(test)
        # elif "svd-sel" == alg:
            # cur_alg = SVDK(k=args.svd_size, model_type="svd-sel")
            # cur_alg.train(table_stats, train)
            # yhat = cur_alg.test(test)
        # else:
            # yhat = predict_prob(test, embedding_model, alg=alg, ytrue=ytest)

        # test_rel_loss = compute_relative_loss(yhat, ytest)
        # test_abs_loss = compute_abs_loss(yhat, ytest)
        # test_qloss = compute_qerror(yhat, ytest)

        # print("case: {}, alg: {}, samples: {}, abs_loss: {}, relative_loss: {}"\
                # .format(args.table_name, alg, len(yhat), sum(abs(yhat-ytest)),
                    # test_rel_loss))
        # print("rel: {}, abs: {}, qerr: {}".format(test_rel_loss, test_abs_loss,
            # test_qloss))

        # store the results
        # result[alg + "rel_loss"] = test_rel_loss
        # result[alg + "abs_loss"] = test_abs_loss
        # result[alg + "qloss"] = test_qloss

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
