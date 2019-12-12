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
#
# preprocess_nhanes.py can be used to preprocess the NHANES dataset before training a generative model
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv('./data/diabetes_data_train.csv')
df_test = pd.read_csv('./data/diabetes_data_test.csv')

X_train = df_train.drop(['status'], axis=1)
y_train = df_train.status

X_test = df_test.drop(['status'], axis=1)
y_test = df_test.status

enc = OneHotEncoder(handle_unknown='ignore')
continuous_cols = [
    "ALQ120Q",
    "BMXBMI",
    "BMXHT",
    "BMXWAIST",
    "BMXWT",
    "RIDAGEYR",
    "SMD030"
]

categorical_cols = [
    "DMDEDUC2",
    "INDHHINC",
    "PAQ180",
    "BPQ020",
    "MCQ250A",
    "RIAGENDR",
    "RIDRETH1"
]

X_train_categorical = pd.concat(map(lambda col: pd.get_dummies(data=df_train[col],
                          prefix=col,
                          drop_first=True,
                          dummy_na=True).astype('int'), categorical_cols),
                                axis=1)

X_train = pd.concat([X_train_categorical, df_train[continuous_cols]], axis=1)
train_cols = X_train.columns
X_test_categorical = pd.concat(map(lambda col: pd.get_dummies(data=df_test[col],
                                                              prefix=col,
                                                              drop_first=True,
                                                              dummy_na=True).astype('int'), categorical_cols),
                               axis=1)
X_test = pd.concat([X_test_categorical, df_test[continuous_cols]], axis=1)

# for features in training set but not in test set,
# add feature and set equal to 0
for col in train_cols:
    if col not in X_test:
        X_test.loc[:, col] = 0

# reorder columns with training set
X_test = X_test[train_cols]

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('./data/processed_train.csv', index=False)
test_df.to_csv('./data/processed_test.csv', index=False)
