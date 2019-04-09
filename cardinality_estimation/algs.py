import time
import numpy as np
import pdb
import math

# TODO: define a parent class to subclass these

class SVDK():

    def __init__(self, k=5, model_type="svd-sel", true_marginals=True):
        self.k = k
        self.model_type = model_type
        self.true_marginals = true_marginals

    def train(self, table_stats, training_samples):
        '''
        Note: going to ignore training samples here.
        '''
        self.word2index = table_stats.word2index
        self.index2word = table_stats.index2word
        self.columns = table_stats.columns

        start = time.time()
        mat = table_stats.get_joint_mat(self.model_type)
        print("computing pmi took: {} seconds".format(time.time() - start))
        start = time.time()
        # smat = scipy.sparse.csc_matrix(mat) # convert to sparse CSC format
        # ut, sv, vt = sparsesvd(smat, embedding_size)
        ut, sv, vt = np.linalg.svd(mat)
        print("computing svd took: {} seconds".format(time.time() - start))
        recon_mat = np.dot(ut * sv, vt)
        print("diff: ", np.sum(np.abs(mat-recon_mat)))
        print("mat, recon_mat allclose: ", np.allclose(mat, recon_mat))

        # TODO: save data for top k singular values
        K = min(self.k, len(sv))
        self.ut = ut[:,0:K]
        self.vt = vt[0:K]
        self.sv = sv[0:K]

        mat1 = np.dot(self.ut*self.sv, self.vt)
        print("diff: ", np.sum(np.abs(mat-mat1)))
        print(self.ut.shape, self.sv.shape, self.vt.shape)

    def test(self, test_samples):
        recon_mat = np.dot(self.ut*self.sv, self.vt)
        yhat = []
        for sample in test_samples:
            assert len(sample.vals) == 2
            val1 = sample.vals[0]
            col1 = sample.column_names[0]
            val2 = sample.vals[1]
            col2 = sample.column_names[1]
            i1 = self._get_index(col1+val1)
            i2 = self._get_index(col2+val2)
            idx1 = min(i1,i2)
            idx2 = max(i1,i2)

            if self.model_type == "svd-rho":
                rho = recon_mat[idx1][idx2]
                marginals = sample.get_marginals(true_marginals=self.true_marginals)
                pa = marginals[0]
                pb = marginals[1]
                normalizer = math.sqrt(pa*(1-pa)*pb*(1-pb))
                est = rho*normalizer + pa*pb
            elif self.model_type == "svd-sel":
                est = recon_mat[idx1][idx2]

            yhat.append(est)
        return np.array(yhat)

    def size(self):
        pass
    def __str__(self):
        pass

    def _get_index(self, val):
        idx = self.word2index[val]
        return idx
