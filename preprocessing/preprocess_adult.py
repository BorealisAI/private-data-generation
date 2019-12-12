# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# preprocess_adult.py can be used to preprocess the adult dataset before training a generative model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/adult.csv")


salary_map = {'<=50K': 1, '>50K': 0}
df['income'] = df['income'].map(salary_map).astype(int)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).astype(int)
df['native-country'] = df['native-country'].replace('?', np.nan)
df['workclass'] = df['workclass'].replace('?', np.nan)
df['occupation'] = df['occupation'].replace('?', np.nan)
df.dropna(how='any', inplace=True)

df.loc[df['native-country'] != 'United-States', 'native-country'] = 'Non-US'
df.loc[df['native-country'] == 'United-States', 'native-country'] = 'US'
df['native-country'] = df['native-country'].map({'US': 1, 'Non-US': 0}).astype(int)

df['marital-status'] = df['marital-status'].replace(['Divorced', 'Married-spouse-absent', 'Never-married', 'Separated',
                                                     'Widowed'], 'Single')
df['marital-status'] = df['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse'], 'Couple')
df['marital-status'] = df['marital-status'].map({'Couple': 0, 'Single': 1})
rel_map = {'Unmarried': 0, 'Wife': 1, 'Husband': 2, 'Not-in-family': 3, 'Own-child': 4, 'Other-relative': 5}
df['relationship'] = df['relationship'].map(rel_map)

df['race'] = df['race'].map({'White': 0, 'Amer-Indian-Eskimo': 1, 'Asian-Pac-Islander': 2, 'Black': 3, 'Other': 4})


def f(x):
    if x['workclass'] == 'Federal-gov' or x['workclass'] == 'Local-gov' or x['workclass'] == 'State-gov':
        return 'govt'
    elif x['workclass'] == 'Private':
        return 'private'
    elif x['workclass'] == 'Self-emp-inc' or x['workclass'] == 'Self-emp-not-inc':
        return 'self_employed'
    else:
        return 'without_pay'


df['employment_type'] = df.apply(f, axis=1)
employment_map = {'govt': 0, 'private': 1, 'self_employed': 2, 'without_pay': 3}
df['employment_type'] = df['employment_type'].map(employment_map)
df.drop(labels=['workclass', 'education', 'occupation'], axis=1, inplace=True)

df.loc[(df['capital-gain'] > 0), 'capital-gain'] = 1
df.loc[(df['capital-gain'] == 0, 'capital-gain')] = 0
df.loc[(df['capital-loss'] > 0), 'capital-loss'] = 1
df.loc[(df['capital-loss'] == 0, 'capital-loss')] = 0

df.drop(['fnlwgt'], axis=1, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
train_df.to_csv('./data/adult_processed_train.csv', index=False)
test_df.to_csv('./data/adult_processed_test.csv', index=False)
