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
# This pre-processing code is derived from the below kaggle notebook :
# https://www.kaggle.com/shep312/deep-learning-in-tf-with-upsampling-lb-758
#
# preprocess_home_credit.py can be used to preprocess the home-credit dataset before training a generative model

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def agg_and_merge(left_df, right_df, agg_method, right_suffix):


    agg_df = right_df.groupby('SK_ID_CURR').agg(agg_method)
    merged_df = left_df.merge(agg_df, left_on='SK_ID_CURR', right_index=True, how='left',
                              suffixes=['', '_' + right_suffix + agg_method.upper()])
    return merged_df


def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):

    # # Add new features

    # Amount loaned relative to salary
    app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']

    # Number of overall payments
    app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']

    # Social features
    app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']

    # A lot of the continuous days variables have integers as missing value indicators.
    prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

    # # Aggregate and merge supplementary datasets

    # Previous applications
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

    # Average the rest of the previous app data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, prev_app_df, agg_method, 'PRV')
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR') \
        .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1 / credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CC_WAVG'])
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, credit_card_avgs, agg_method, 'CC')
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist() + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left', suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    # Credit bureau data - numerical features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, bureau_df, agg_method, 'B')
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                                how='left', suffixes=['', '_B_B'])
    print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1 / pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF': wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])

    # Unweighted aggregations of numeric features
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, pos_cash_df, agg_method, 'PC')

    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist() + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR',
                                right_on='SK_ID_CURR',
                                how='left', suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    # Installments data
    for agg_method in ['mean', 'max', 'min']:
        merged_df = agg_and_merge(merged_df, install_df, agg_method, 'I')
    print('Shape after merging with installments data = {}'.format(merged_df.shape))

    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    print('Shape after merging with counts data = {}'.format(merged_df.shape))

    return merged_df


def process_dataframe(input_df, encoder_dict=None):

    # Label encode categoricals
    print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict


input_dir = "./data/"
print('Loading data sets...')

sample_size = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size)
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size)
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size)
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size)
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size)

# Merge the datasets into a single one for training
len_train = len(app_train_df)
app_both = pd.concat([app_train_df, app_test_df])
merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,
                                pos_cash_df, prev_app_df, install_df)

meta_cols = ['SK_ID_CURR']
meta_df = merged_df[meta_cols]
merged_df.drop(labels=meta_cols, axis=1, inplace=True)

merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)
null_columns = ['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'AMT_DRAWINGS_ATM_CURRENT',
                'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
                'AMT_PAYMENT_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
                'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'MONTHS_BALANCE_B_B']

merged_df.drop(labels=null_columns, axis=1, inplace=True)
labeled_df = merged_df[:len_train]
train_df, test_df = train_test_split(labeled_df, test_size=0.25, random_state=42)
train_df.to_csv('./data/processed_train.csv', index=False)
test_df.to_csv('./data/processed_test.csv', index=False)
