from pygam import LinearGAM
import numpy as np
import scipy as sp
from copy import deepcopy
EPS = np.finfo(np.float64).eps

from collections import defaultdict
from pygam.utils import check_y
from pygam.utils import check_X
from pygam.utils import check_X_y
from pygam.utils import check_array
from pygam.utils import check_lengths

class my_LinearGAM(LinearGAM):
    def __init__(self):
        super(my_LinearGAM, self).__init__()

    def my_pirls(self, modelmat, Y, weights):# build a basis matrix for the GLM
        n, m = modelmat.shape
        if (
                not self._is_fitted
                or len(self.coef_) != self.terms.n_coefs
                or not np.isfinite(self.coef_).all()
        ):
            # initialize the model
            self.coef_ = self._initial_estimate(Y, modelmat)

        assert np.isfinite(
            self.coef_
        ).all(), "coefficients should be well-behaved, but found: {}".format(self.coef_)

        P = self._P()
        S = sp.sparse.diags(np.ones(m) * np.sqrt(EPS))  # improve condition
        # S += self._H # add any user-chosen minumum penalty to the diagonal

        # if we dont have any constraints, then do cholesky now
        if not self.terms.hasconstraint:
            E = self._cholesky(S + P, sparse=False, verbose=self.verbose)

        min_n_m = np.min([m, n])
        Dinv = np.zeros((min_n_m + m, m)).T

        for _ in range(self.max_iter):
            # recompute cholesky if needed
            if self.terms.hasconstraint:
                P = self._P()
                C = self._C()
                E = self._cholesky(S + P + C, sparse=False, verbose=self.verbose)

            # forward pass
            y = deepcopy(Y)  # for simplicity
            lp = self._linear_predictor(modelmat=modelmat)
            mu = self.link.mu(lp, self.distribution)
            W = self._W(mu, weights, y)  # create pirls weight matrix

            # check for weghts == 0, nan, and update
            mask = self._mask(W.diagonal())
            y = y[mask]  # update
            lp = lp[mask]  # update
            mu = mu[mask]  # update
            W = sp.sparse.diags(W.diagonal()[mask])  # update

            # PIRLS Wood pg 183
            pseudo_data = W.dot(self._pseudo_data(y, lp, mu))

            # log on-loop-start stats
            self._on_loop_start(vars())

            WB = W.dot(modelmat[mask, :])  # common matrix product
            Q, R = np.linalg.qr(WB.A)

            if not np.isfinite(Q).all() or not np.isfinite(R).all():
                raise ValueError(
                    'QR decomposition produced NaN or Inf. ' 'Check X data.'
                )

            # need to recompute the number of singular values
            min_n_m = np.min([m, n, mask.sum()])
            Dinv = np.zeros((m, min_n_m))

            # SVD
            U, d, Vt = np.linalg.svd(np.vstack([R, E]))

            # mask out small singular values
            # svd_mask = d <= (d.max() * np.sqrt(EPS))

            np.fill_diagonal(Dinv, d** -1)  # invert the singular values
            U1 = U[:min_n_m, :min_n_m]  # keep only top corner of U

            # update coefficients
            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
            coef_new = B.dot(pseudo_data).flatten()
            diff = np.linalg.norm(self.coef_ - coef_new) / np.linalg.norm(coef_new)
            self.coef_ = coef_new  # update

            # log on-loop-end stats
            self._on_loop_end(vars())

            # check convergence
            if diff < self.tol:
                break

        # estimate statistics even if not converged
        self._estimate_model_statistics(
            Y, modelmat, inner=None, BW=WB.T, B=B, weights=weights, U1=U1
        )
        if diff < self.tol:
            return

        print('did not converge')
        return

    def my_fit(self, X, modelmat, y, weights=None):
        """Fit the generalized additive model.

        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values,
            ie integers in classification, real numbers in
            regression)
        weights : array-like shape (n_samples,) or None, optional
            Sample weights.
            if None, defaults to array of ones

        Returns
        -------
        self : object
            Returns fitted GAM object
        """

        # validate parameters
        self._validate_params()

        # validate data
        y = check_y(y, self.link, self.distribution, verbose=self.verbose)
        X = check_X(X, verbose=self.verbose)
        check_X_y(X, y)

        if weights is not None:
            weights = np.array(weights).astype('f').ravel()
            weights = check_array(
                weights, name='sample weights', ndim=1, verbose=self.verbose
            )
            check_lengths(y, weights)
        else:
            weights = np.ones_like(y).astype('float64')

        # validate data-dependent parameters
        self._validate_data_dep_params(X)

        # set up logging
        if not hasattr(self, 'logs_'):
            self.logs_ = defaultdict(list)

        # begin capturing statistics
        self.statistics_ = {}
        self.statistics_['n_samples'] = len(y)
        self.statistics_['m_features'] = X.shape[1]

        # optimize
        self.my_pirls(modelmat, y, weights)
        # if self._opt == 0:
        #     self._pirls(X, y, weights)
        # if self._opt == 1:
        #     self._pirls_naive(X, y)
        return self