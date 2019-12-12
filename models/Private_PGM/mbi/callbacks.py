# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import time
import pandas as pd
import numpy as np

class CallBack:
    """ A CallBack is a function called after every iteration of an iterative optimization procedure
    It is useful for tracking loss and other metrics over time.
    """
    def __init__(self, engine, frequency = 50):
        """ Initialize the callback objet

        :param engine: the FactoredInference object that is performing the optimization
        :param frequency: the number of iterations to perform before computing the callback function
        """
        self.engine = engine
        self.frequency = frequency
        self.calls = 0

    def run(self, marginals):
        pass

    def __call__(self, marginals):
        if self.calls == 0:
            self.start = time.time()
        if self.calls % self.frequency == 0:
            self.run(marginals)
        self.calls += 1

class Logger(CallBack):
    """ Logger is the default callback function.  It tracks the time, L1 loss, L2 loss, and
        optionally the total variation distance to the true query answers (when available).
        The last is for debugging purposes only - in practice the true answers can not  be observed.
    """
    def __init__(self, engine, true_answers = None, frequency = 50):
        """ Initialize the callback objet

        :param engine: the FactoredInference object that is performing the optimization
        :param true_answers: a dictionary containing true answers to the measurement queries.
        :param frequency: the number of iterations to perform before computing the callback function
        """
        CallBack.__init__(self, engine, frequency)
        self.true_answers = true_answers
        self.idx = 0

    def setup(self):
        model = self.engine.model
        total = sum(model.domain.size(cl) for cl in model.cliques)
        print('Total clique size:', total)
        cl = max(model.cliques, key=lambda cl: model.domain.size(cl))
        print('Maximal clique', cl, model.domain.size(cl))
        cols = ['iteration', 'time', 'l1_loss', 'l2_loss']
        if self.true_answers is not None:
            cols.append('variation')
        self.results = pd.DataFrame(columns=cols)
        print('\t\t'.join(cols))

    def variational_distances(self, marginals):
        errors = []
        model = self.engine.model
        for Q, y, proj in self.true_answers:
            for cl in marginals:
                if set(proj) <= set(cl):
                    mu = marginals[cl].project(proj)
                    x = mu.values.flatten()
                    diff = Q.dot(x) - y
                    err = 0.5 * np.abs(diff).sum() / y.sum()
                    errors.append(err)
                    break
        return errors

    def run(self, marginals):
        if self.idx == 0:
            self.setup()

        t = time.time() - self.start
        l1_loss = self.engine._marginal_loss(marginals, metric='L1')[0]
        l2_loss = self.engine._marginal_loss(marginals, metric='L2')[0]
        row = [self.calls, t, l1_loss, l2_loss]
        if self.true_answers is not None:
            variational = np.mean(self.variational_distances(marginals))
            row.append(variational)
        self.results.loc[self.idx] = row
        self.idx += 1
        
        print('\t\t'.join(['%.2f' % v for v in row]))
