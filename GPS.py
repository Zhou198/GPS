import numpy as np
from cvxopt import matrix, solvers


class GPSclassifier:
    def __init__(self, alpha=0.05, K=2, calibSet=None, candSigqtl=[50], candC1=[1]):
        """
        Initialize a GPS model.

        Parameters:
        :param alpha: non-coverage rate
        :param K: number of classes
        :param calibSet: calibration set for parameter tuning
        :param candSigqtl: quantiles of pair-wise distances for searching the optimal value of sigma in the Gaussian kernel
        :param candC1: a grid of hyperparameters C in eq. (5)
        """

        self.alpha = alpha
        self.K = K
        if calibSet is not None:
            self.caL, self.calibSet = calibSet[:, 0], calibSet[:, 1:]

        self.candSigqtl = candSigqtl
        self.candC1 = candC1
        self.classifierDict = {k: {"trsp12": None, "m": None, "optSig": None, "solab": None, "solr": None} for k in range(1, 1 + K)}
        self.threshList = np.zeros(K)

    def train(self, trainSet, trL):
        """
        Train the GPS model.

        Parameters:
        :param trainSet: feature array (X) of training set consisting of label and unlabeled observations
        :param trL: label array (Y) of training set where 0 refers to unlabeled observations
        """
        for l in range(1, self.K + 1):
            print(f"start to train class {l}")
            inSample, outSample = trainSet[trL == l], trainSet[trL == 0]

            trsp12 = np.vstack((inSample, outSample))
            m, n = len(inSample), len(outSample)
            Distsp12 = self.EucDist(trsp12, trsp12)

            # pairwise distance for finding sigma
            Distsp11 = Distsp12[:m, :m]
            duidx = np.triu_indices(len(Distsp11), k=1)
            candSig = np.percentile(Distsp11[duidx], self.candSigqtl) ** 0.5

            wobj, wcst = np.ones((n, 1)), np.ones((m, 1))
            bestclf = self.validCVXGPS(Distsp12, trsp12, self.calibSet, self.caL, l, m, n, self.candC1, candSig, self.alpha, wobj, wcst)

            self.classifierDict[l]["trsp12"], self.classifierDict[l]["m"], self.classifierDict[l]["optSig"], \
                self.classifierDict[l]["solab"], self.classifierDict[l]["solr"] = trsp12, m, bestclf["optSig"], bestclf["solab"], bestclf["solr"]

            fkcal = self.fcvx(self.calibSet[self.caL == l], trsp12, m, bestclf["optSig"], bestclf["solab"], bestclf["solr"])
            self.threshList[l - 1] = np.sort(fkcal, 0)[np.fmax((np.sum(self.caL == l) * self.alpha).astype(int), 1) - 1]

    def test(self, testSetX):
        """
        Compute scores for each test point.

        Parameters:
        :param testSetX: feature array (X) of testing set
        :return: score array (f_1, ..., f_K) for each test point
        """
        scoMat = np.zeros([len(testSetX), self.K])
        for l in range(1, 1 + self.K):
            trsp12, m, optsig, solab, solr = self.classifierDict[l]["trsp12"], self.classifierDict[l]["m"], \
                self.classifierDict[l]["optSig"], self.classifierDict[l]["solab"], self.classifierDict[l]["solr"]
            scoMat[:, l - 1] = self.fcvx(testSetX, trsp12, m, optsig, solab, solr)[:, 0]
        return scoMat

    def cvxoptfun(self, Kspsp, m, n, C, alpha, wobj, wcst):
        """
        Quadratic programming to solve problem (5).

        Parameters:
        :param Kspsp: kernel matrix
        :param m: sample size of (labeled) normal observations (differs from the paper's notation)
        :param n: sample size of unlabeled data
        :param C: hyperparameter C in problem (5)
        :param alpha: non-coverage rate
        :param wobj: just set it as an array of ones
        :param wcst: same as above
        :return: GPS model parameters in problem (5)
        """

        P = np.zeros((1 + m + n, 1 + m + n))
        P[1:, 1:] = np.block([[Kspsp[:m, :m], -Kspsp[:m, -n:]], [-Kspsp[-n:, :m], Kspsp[-n:, -n:]]])
        P = matrix(P)

        q = matrix(np.hstack((m * alpha, [-1.] * m, [-1] * n)))

        G = np.zeros((1 + 2 * (m + n), 1 + m + n))
        G[:(1 + m + n)] = -np.eye(1 + m + n)
        G[-(m + n):(-n), 0] = -wcst[:, 0]
        G[-(m + n):, 1:] = np.eye(m + n)
        G = matrix(G)

        h = matrix(np.hstack(([0.], [0.] * m, [0.] * n, [0.] * m, C * wobj[:, 0])))

        A = matrix(np.repeat([0., 1, -1], [1, m, n])).T
        b = matrix([1.])

        sol = solvers.qp(P, q, G, h, A, b, options={"show_progress": False})
        if sol["status"] != "optimal": print(sol["status"])

        return sol["x"][1:]

    def rhoprog(self, f0, m, n, C, alpha, wobj, wcst):
        """
        Linear Programming to compute the offset rho.

        Parameters:
        :param f0: the value of the difference between first two terms in f_k
        :param m: sample size of (labeled) normal observations
        :param n: sample size of unlabeled data
        :param C: hyperparameter in problem (5)
        :param alpha: non-coverage rate
        :param wobj: just set it as an array of ones
        :param wcst: same as above
        :return: the offset terms rho
        """
        c = matrix(np.hstack((-1., C * n / m * wcst[:, 0], C * wobj[:, 0])))

        G = np.zeros((1 + 2 * (m + n), 1 + m + n))
        G[:(m + n), 0] = [1.] * m + [-1] * n
        G[:(m + n), 1:] = -np.eye(m + n)
        G[m + n, 1:(1 + m)] = wcst[:, 0]
        G[-(m + n):, 1:] = -np.eye(m + n)
        G = matrix(G)

        h = matrix(np.hstack((f0[:m, 0] - 1, -f0[-n:, 0] - 1, m * alpha, [0.] * m, [0.] * n)))

        sol = solvers.lp(c, G, h, options={"show_progress": False})
        if sol["status"] != "optimal": print(sol["status"])

        return sol["x"][0]

    def EucDist(self, matM, matN):
        """
        Compute pairwise distance between observations from two datasets.

        Parameters:
        :param matM: one dataset
        :param matN: another dataset
        :return: pairwise distance matrix
        """
        MM, NN, MN = np.sum(matM ** 2, 1, keepdims=True), np.sum(matN ** 2, 1), matM @ matN.T
        return MM + NN - 2 * MN

    def Knl(self, matM, matN, r):
        """
        Compute the Gaussian kernel matrix.

        Parameters:
        :param matM: one matrix for finding pairwise distance matrix
        :param matN: another matrix for finding pairwise distance matrix
        :param r: hyperparameter sigma
        :return: Gaussian kernel matrix
        """
        return np.exp(-self.EucDist(matM, matN) / r ** 2)

    def fcvx(self, x, trsp12, m, sig, ab, rh):
        """
        Score function f_k for each test point.

        Parameters:
        :param x: test point/dataset
        :param trsp12: training set
        :param m: sample size of (labeled) normal observations in the training set
        :param sig: hyperparameter sigma
        :param ab: model parameters returned by quadratic programming
        :param rh: offset rho returned by linear programming
        :return: score function
        """
        Kmat = self.Knl(x, trsp12, sig)
        return Kmat @ np.vstack((ab[:m], -ab[m:])) - rh

    def hgls(self, u, s=1):
        """
        Hinge loss.
        """
        return np.fmax(0, s - u)

    def validCVXGPS(self, Distsp12, trsp12, calibSet, caL, k, m, n, candC, candSig, alpha, wobj, wcst):
        """
        Use the calibration dataset to select tuning parameters.

        Parameters:
        :param Distsp12: pairwise distance matrix
        :param trsp12: training set
        :param calibSet: feature array (X) of calibration set consisting of label and unlabeled sets
        :param caL: label array (Y) of calibration set where 0 refers to unlabeled observations
        :param k: the label of current normal observations
        :param m: sample size of the normal observations
        :param n: sample size of unlabeled data
        :param candC: a grid of hyperparameter C
        :param candSig: a grid of hyperparameter sigma
        :param alpha: non-coverage rate
        :param wobj: just set it as an array of ones
        :param wcst: same as above
        :return: return the optimal GPS with selected hyperparameters
        """
        nC, nSig = len(candC), len(candSig)
        cvErr = np.zeros((nC, nSig))

        minErr = np.inf
        optRes = {"solab": 0, "solr": 0, "thre": 0, "optC": 0, "optSig": 0}
        for i2 in range(nSig):
            Kmatsig = np.exp(-Distsp12 / candSig[i2] ** 2)

            for i1 in range(nC):
                ab = self.cvxoptfun(Kmatsig, m, n, candC[i1], alpha, wobj, wcst)
                f0 = Kmatsig[:, :m] @ ab[:m] - Kmatsig[:, m:] @ ab[m:]
                rho = self.rhoprog(f0, m, n, candC[i1], alpha, wobj, wcst)

                calibsco = self.fcvx(calibSet, trsp12, m, candSig[i2], ab, rho)
                threk = np.sort(calibsco[caL == k, 0])[max(0, int(np.sum(caL == k) * alpha) - 1)]
                cvErr[i1, i2] = np.mean(self.hgls(threk - calibsco[caL == 0]))

                if cvErr[i1, i2] < minErr:
                    minErr = cvErr[i1, i2]
                    optRes["solab"], optRes["solr"], optRes["thre"] = ab, rho, threk
                    optRes["optC"], optRes["optSig"] = candC[i1], candSig[i2]

        # print(f"validation err is {cvErr}")
        return optRes
