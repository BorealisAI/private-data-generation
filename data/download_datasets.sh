#!/bin/bash

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
# download_datasets.sh can be used to download the datasets used in the experiments

if [ "$1" = "adult" ]; then
    kaggle datasets download wenruliu/adult-income-dataset
    unzip adult-income-dataset.zip
    rm adult-income-dataset.zip
elif [ "$1" = "nhanes" ]; then
    wget https://raw.githubusercontent.com/semerj/NHANES-diabetes/master/data/diabetes_data_train.csv
    wget https://raw.githubusercontent.com/semerj/NHANES-diabetes/master/data/diabetes_data_test.csv
elif [ "$1" = "givemesomecredit" ]; then
    kaggle competitions download -f cs-training.csv GiveMeSomeCredit
elif [ "$1" = "home-credit" ]; then
    kaggle competitions download -c home-credit-default-risk
    unzip home-credit-default-risk.zip
    rm home-credit-default-risk.zip
elif [ "$1" = "adult-categorical" ]; then
    wget https://raw.githubusercontent.com/ryan112358/private-pgm/master/data/adult.csv
else
    echo "Unknown dataset"
fi