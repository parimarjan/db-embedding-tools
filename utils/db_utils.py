import psycopg2 as pg
from utils import *
import pdb
import time
import random
import re
import numpy as np
import scipy.sparse
from sparsesvd import sparsesvd
from sklearn.metrics.pairwise import cosine_similarity
import math

CARD_TMP = "SELECT {COLUMNS}, count(*) from {TABLE} group by {COLUMNS};"

SAMPLES_SINGLE_TEMPLATE = "EXPLAIN SELECT COUNT(*) FROM {table} WHERE {column}\
{cmp_op} '{val}'"
SAMPLES_CORR_TEMPLATE = "EXPLAIN SELECT COUNT(*) FROM {table} WHERE {column1} {cmp_op1} \
'{val1}' AND {column2} {cmp_op2} '{val2}'"

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
            embedding_size = len(model[model.index2word[0]])
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
                    features.extend([0]*embedding_size)

        elif featurization_scheme == "onehot":
            # FIXME: estimate total some other way.
            total = len(model.index2word)
            for i in range(len(self.column_names)):
                val = self.vals[i]
                onehot_vec = [0.00]*total
                try:
                    model[val]
                    idx = hash(val) % total
                    onehot_vec[idx] = 1.00
                except:
                    pass
                # else, we just estimate it as the 0 vec
                features.extend(onehot_vec)

        return features

# TODO: wrap around all models we use?
class SVDEmbedding():
    def __init__(self, ts, embedding_size=100, model_type=None):
        '''
        '''
        self.ts = ts
        self.embedding_size = embedding_size
        self.model_type = model_type
        # generate a word2index dict for optimized lookup
        ut, s, vt = self._gen_model(model_type, embedding_size=embedding_size)
        self.svd_mat = np.dot(ut * s, vt)
        self.word_features = ut
        self.s = s
        self.ut = ut
        self.vt = vt
        # self.word_features = vt

        # FIXME: wasteful...
        self.index2word = self.ts.index2word

    def similarity(self, word1, word2):
        '''
        TODO: how to handle case where column name not appended to word?
        '''
        word1_feats = self.__getitem__(word1)
        word2_feats = self.__getitem__(word2)
        # take their dot product, or cosine distance etc.
        sim = np.dot(word1_feats, word2_feats)
        # sim = np.dot(word1_feats, word2_feats) / \
                        # np.linalg.norm(word1_feats)*np.linalg.norm(word2_feats)

        return sim

        # if self.model_type == "pmi-svd":
            # word1_feats = self.__getitem__(word1)
            # word2_feats = self.__getitem__(word2)
            # # take their dot product, or cosine distance etc.
            # sim = np.dot(word1_feats, word2_feats) / \
                            # np.linalg.norm(word1_feats)*np.linalg.norm(word2_feats)

            # return sim
        # else:
            # idx1 = self._get_index(word1)
            # idx2 = self._get_index(word2)
            # return self.svd_mat[idx1][idx2]

    def _get_index(self, val):
        idx = None
        if val not in self.ts.word2index:
            for col in self.ts.columns:
                new_val = col + val
                if new_val in self.ts.word2index:
                    idx = self.ts.word2index[new_val]
                    break
        else:
            idx = self.ts.word2index[val]
        return idx

    def __getitem__(self, val):
        assert isinstance(val, str)
        idx = self._get_index(val)
        assert idx is not None
        features = self.word_features[:, idx]
        return features

    def _gen_model(self, model_type, embedding_size=100):
        '''
            - calculate pmi
            - use svd to reduce to given embedding size
            - generate embedding vectors for all inputs
        '''
        start = time.time()
        mat = self.ts.get_joint_mat(model_type)
        print("computing pmi took: {} seconds".format(time.time() - start))
        start = time.time()
        # smat = scipy.sparse.csc_matrix(mat) # convert to sparse CSC format
        # ut, sv, vt = sparsesvd(smat, embedding_size)
        ut, sv, vt = np.linalg.svd(mat)
        print("computing svd took: {} seconds".format(time.time() - start))

        # assert(np.allclose(mat, np.dot(ut * sv, vt)))
        # diff = np.subtract(mat, np.dot(ut * sv, vt))
        # diff = np.abs(diff)
        # print("diff after svd: ", diff.sum())
        print("allclose: ", np.allclose(mat, np.dot(ut * sv, vt)))

        # assert np.allclose(mat, np.dot(ut.T, np.dot(np.diag(sv), vt)))
        print(ut.shape, sv.shape, vt.shape)

        # let's check some basic things about ut, vt etc.
        it = ut.dot(vt)
        ev, evs = np.linalg.eig(mat)
        mat2 = np.matmul(np.matmul(evs, np.diag(ev)), evs.T)
        print("abs(eigenvalues) vs svds allclose: ", np.allclose(np.abs(ev), sv))
        print("eig reconstr, allclose: ", np.allclose(mat, mat2))
        print("abs(ut), abs(vt.T)): ", np.allclose(abs(ut), abs(vt.T)))
        # s_imp = sv[sv > 0.50]
        # print(len(sv))
        # print("num significant values > 0.50", len(s_imp))

        # print(ut.shape)
        # print(vt.shape)
        # pdb.set_trace()
        return ut, sv, vt
        # return ut

    def most_similar(self, word, n = 10):
        similarities = []
        for w2 in self.ts.index2word:
            similarities.append((w2, self.similarity(word, w2)))
            # print(w2, similarities[-1])
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[0:n]

class TableStats():

    def __init__(self, db_name, table_name, db_host="localhost",
            columns_string="col0,col1"):
        '''
        creates a conn to the db, and then will continue to reuse that.
        Provides the following:
            -
        '''
        self.db_host = db_host
        self.db_name = db_name
        self.table_name = table_name
        self.columns_string = columns_string
        self.columns = columns_string.split(",")
        # each index represents a value from the db
        self.index2word = []
        self.word2index = {}

        # FIXME: generalize this to handle numeric types too
        self.dtype = "str"

        conn = pg.connect(host=db_host, database=db_name)
        cur = conn.cursor()

        Q = "SELECT COUNT(*) FROM {table}".format(table=self.table_name)
        cur.execute(Q)
        self.total_rows = float(cur.fetchone()[0])

        # generate stats for each pair of columns
        self.cardinalities = {}
        self.joint_cardinalities = {}

        assert columns_string is not None
        # get for individual columns
        columns_list = columns_string.split(",")
        print("num columns: ", len(columns_list))
        for col in columns_list:
            cmd = CARD_TMP.format(COLUMNS = col, TABLE=table_name)
            cur.execute(cmd)
            for o in cur:
                # FIXME: handle the general case?
                key = col + o[0]
                card = o[1]
                self.cardinalities[key] = card
                self.index2word.append(key)
                self.word2index[key] = len(self.index2word)-1

        print("distinct values: ", len(self.index2word))

        # TODO: should this be for only 2 columns combined no matter what?
        # get for all columns combined
        cmd = CARD_TMP.format(COLUMNS = columns_string, TABLE=table_name)
        cur.execute(cmd)
        for o in cur:
            # FIXME: handle the general case?
            key = self.columns[0] + o[0] + "+" + self.columns[1] + o[1]
            card = o[2]
            self.joint_cardinalities[key] = card

        cur.close()
        conn.close()

    def get_joint_mat(self, model_type):
        '''
        using the collected stats, generate the pointwise mutual information
        matrix.
        TODO: decide types
        PMI formula:
        '''
        num_vals = len(self.index2word)
        mat = np.zeros((num_vals, num_vals), dtype=np.float32)
        # much faster than looping over O(n^2) elements, since most of those
        # are zero anyway.
        for key,pab in self.joint_cardinalities.items():
            pab = pab / self.total_rows
            ind_keys = key.split("+")
            key_a = ind_keys[0]
            key_b = ind_keys[1]
            i = self.word2index[key_a]
            j = self.word2index[key_b]
            idx1 = min(i,j)
            idx2 = max(i,j)
            pa = self.cardinalities[key_a] / self.total_rows
            pb = self.cardinalities[key_b] / self.total_rows
            # FIXME: temp
            mat[i][i] = pa*pa
            mat[j][j] = pb*pb
            if model_type == "pmi-svd":
                mat[idx1][idx2] = np.log((pab / (pa*pb)))
            elif model_type == "svd-sel":
                mat[idx1][idx2] = pab
            elif model_type == "svd-rho":
                normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
                rho = ((pab) - (pa*pb)) / normalizer
                mat[idx1][idx2] = rho
            else:
                assert False

        return mat

    def get_entropies(self):
        pass

    def get_samples(self, num_samples=100, num_columns=2):
        '''
        '''
        def parse_explain(output):
            '''
            '''
            est_vals = None
            for line in output:
                line = line[0]
                if "Seq Scan" in line:
                    for w in line.split():
                        if "rows" in w and est_vals is None:
                            est_vals = int(re.findall("\d+", w)[0])
            assert est_vals is not None
            return est_vals

        def sample_row_values(op, idx1, idx2, rows):
            '''
            FIXME: could probably directly just do a sql query here.
            '''
            print("sample row values!")
            req_rows = None
            if self.dtype == "str":
                vals = []
                for i in range(num_samples):
                    row = random.choice(rows)
                    val1 = row[idx1]
                    val2 = row[idx2]
                    # val1 = val1.replace("'", "")
                    # val2 = val2.replace("'", "")
                    # print(val1)
                    # val1 = "'" + val1.replace("'", "''") + "'"
                    # val2 = "'" + val2.replace("'", "''") + "'"
                    vals.append((val1, val2))
                return vals
            else:
                assert False

        def get_selectivity_single_predicate(col, val, op):
            key = col+val
            val = val.replace("'", "''")
            Q = SAMPLES_SINGLE_TEMPLATE.format(table=self.table_name,
                    column=col, cmp_op = op, val = val)
            cursor.execute(Q)
            exp_output = cursor.fetchall()
            est_vals = parse_explain(exp_output)

            # for true value
            num_vals = self.cardinalities[key]
            return est_vals, num_vals

        # FIXME: generalize to n-predicates
        def get_selectivity_two_predicate(col1, col2, val1, val2, op):
            # FIXME: generalize to n predicates
            key = col1+val1+ "+" +col2+val2
            val1 = val1.replace("'", "''")
            val2 = val2.replace("'", "''")
            Q = SAMPLES_CORR_TEMPLATE.format(table=self.table_name,
                    column1=col1, cmp_op1 = op, val1 = val1,
                    column2=col2, cmp_op2 = op, val2 = val2)
            cursor.execute(Q)
            exp_output = cursor.fetchall()
            est_vals = parse_explain(exp_output)
            num_vals = self.joint_cardinalities[key]
            return est_vals, num_vals

        # get a random sample of values from each column so we can use it to
        # generate sensible queries
        conn = pg.connect(host=self.db_host, database=self.db_name)
        cursor = conn.cursor()

        Q = "SELECT {COLUMNS} FROM {TABLE} WHERE random() < 0.001".\
                format(TABLE=self.table_name, COLUMNS=self.columns_string)
        cursor.execute(Q)
        rows = cursor.fetchall()
        test_columns = [c.name for c in cursor.description]
        for i, c in enumerate(test_columns):
            assert c == self.columns[i]

        # can get column names from the cursor as well
        samples = []
        cmp_ops = None
        if self.dtype == "str":
            cmp_ops = STR_CMP_OPS
        else:
            cmp_ops = NUM_CMP_OPS

        start = time.time()

        # FIXME: generalize to n-columns
        if num_columns == 2:
            start = time.time()
            col1 = self.columns[0]
            col2 = self.columns[1]

            for op in cmp_ops:
                # TODO: different operators on both sides
                # maybe make choosing the operator a random choice
                vals = sample_row_values(op, 0, 1, rows)
                for k, (val1, val2) in enumerate(vals):
                    if k % 100 == 0:
                        print("generated {} samples".format(k))

                    # first, do it for each of the single values
                    est_val1, num_val1 = get_selectivity_single_predicate(col1,
                            val1, op)
                    est_val2, num_val2 = get_selectivity_single_predicate(col2,
                            val2, op)

                    pgsel1 = est_val1 / float(self.total_rows)
                    pgsel2 = est_val2 / float(self.total_rows)
                    sel1 = num_val1 / float(self.total_rows)
                    sel2 = num_val2 / float(self.total_rows)
                    pg_marginal_sels = [pgsel1, pgsel2]
                    marginal_sels = [sel1, sel2]

                    # both predicates together
                    est_vals, num_vals = get_selectivity_two_predicate(col1,
                            col2, val1, val2, op)

                    columns = [col1, col2]
                    vals = [val1, val2]
                    ops = [op, op]
                    sel = float(num_vals) / self.total_rows
                    pg_sel = float(est_vals) / self.total_rows

                    sample = SelectivitySample(columns, vals, ops, sel,
                            num_vals, pg_sel=pg_sel,
                            pg_marginal_sels=pg_marginal_sels,
                            marginal_sels=marginal_sels)
                    samples.append(sample)

        else:
            assert False

        print("generating two col samples took ", time.time() - start)
        sels = [s.sel for s in samples]
        zeros = [s for s in samples if s.sel == 0.00]
        print("max selectivity: ", max(sels))
        print("num zeros: ", len(zeros))

        return samples


