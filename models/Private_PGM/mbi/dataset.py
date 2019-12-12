# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
import numpy as np
import pandas as pd
import os
import json
from models.Private_PGM.mbi import Domain

class Dataset:
    def __init__(self, df, domain):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[:,domain.attrs]

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns = domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins)[0]
        return ans.flatten() if flatten else ans
    
