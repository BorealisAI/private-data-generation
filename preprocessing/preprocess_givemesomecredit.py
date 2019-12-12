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

# preprocess_givemesomecredit.py can be used to preprocess the GiveMeSomeCredit dataset before training a generative model
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv('./data/cs-training.csv')
train = train.dropna()
train.drop(['Unnamed: 0'], axis=1, inplace=True)
data_train, data_test = train_test_split(train, test_size=0.25, random_state=42)
data_train.to_csv('./data/processed_train.csv', index=False)
data_test.to_csv('./data/processed_test.csv', index=False)