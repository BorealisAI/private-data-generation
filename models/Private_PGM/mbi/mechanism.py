# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
from models.Private_PGM.mbi import FactoredInference
from models.Private_PGM.mbi.callbacks import Logger
from scipy.stats import norm, laplace
import numpy as np
from scipy import sparse

def run(dataset, measurements, eps=1.0, delta=0.0, bounded=True, engine='MD', 
        options={}, iters=10000, seed=None, metric='L2', elim_order=None, frequency=1,workload=None):
    """
    Run a mechanism that measures the given measurements and runs inference.
    This is a convenience method for running end-to-end experiments.
    """

    domain = dataset.domain
    total = None

    state = np.random.RandomState(seed)

    if len(measurements) >= 1 and type(measurements[0][0]) is str:
        matrix = lambda proj: sparse.eye(domain.project(proj).size())
        measurements = [(proj, matrix(proj)) for proj in measurements]

    l1 = 0
    l2 = 0
    for _, Q in measurements:
        l1 += np.abs(Q).sum(axis=0).max()
        try: l2 += Q.power(2).sum(axis=0).max()  # for spares matrices
        except: l2 += np.square(Q).sum(axis=0).max() # for dense matrices

    if bounded:
        total = dataset.df.shape[0]
        l1 *= 2
        l2 *= 2

    if delta > 0:
        noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2/delta))/eps)
    else:
        noise = laplace(loc=0, scale=l1/eps)

    if workload is None:
        workload = measurements
   
    truth = [] 
    for proj, W, in workload:
        x = dataset.project(proj).datavector()
        y = W.dot(x)
        truth.append( (W, y, proj) )

    answers = []
    for proj, Q in measurements:
        x = dataset.project(proj).datavector()
        z = noise.rvs(size=Q.shape[0], random_state=state)
        y = Q.dot(x)
        answers.append( (Q, y+z, 1.0, proj) )

    estimator = FactoredInference(domain, metric=metric, iters=iters, warm_start=False, elim_order=elim_order)
    logger = Logger(estimator, true_answers=truth, frequency=frequency)
    model = estimator.estimate(answers, total, engine=engine, callback=logger, options=options)
        
    return model, logger, answers
