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
# evaluate.py is used to create the synthetic data generation and evaluation pipeline.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn import preprocessing
from scipy.special import expit
from models import dp_wgan, pate_gan, ron_gauss
from models.Private_PGM import private_pgm
import argparse
import numpy as np
import pandas as pd
import collections
import os
try:
    from models.IMLE import imle
except ImportError as error:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--categorical', action='store_true', help='All attributes of the data are categorical with small domains')
parser.add_argument('--target-variable', help='Required if data has a target class')
parser.add_argument('--train-data-path', required=True)
parser.add_argument('--test-data-path', required=True)
parser.add_argument('--normalize-data', action='store_true', help='Apply sigmoid function to each value in the data')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--downstream-task', default="classification", help='classification | regression')

privacy_parser = argparse.ArgumentParser(add_help=False)

privacy_parser.add_argument('--enable-privacy', action='store_true', help='Enable private data generation')
privacy_parser.add_argument('--target-epsilon', type=float, default=8, help='Epsilon differential privacy parameter')
privacy_parser.add_argument('--target-delta', type=float, default=1e-5, help='Delta differential privacy parameter')
privacy_parser.add_argument('--save-synthetic', action='store_true', help='Save the synthetic data into csv')
privacy_parser.add_argument('--output-data-path', help='Required if synthetic data needs to be saved')



noisy_sgd_parser = argparse.ArgumentParser(add_help=False)

noisy_sgd_parser.add_argument('--sigma', type=float,
                              default=2, help='Gaussian noise variance multiplier. A larger sigma will make the model '
                                              'train for longer epochs for the same privacy budget')
noisy_sgd_parser.add_argument('--clip-coeff', type=float,
                              default=0.1, help='The coefficient to clip the gradients before adding noise for private '
                                                'SGD training')
noisy_sgd_parser.add_argument('--micro-batch-size',
                              type=int, default=8,
                              help='Parameter to tradeoff speed vs efficiency. Gradients are averaged for a microbatch '
                                   'and then clipped before adding noise')

noisy_sgd_parser.add_argument('--num-epochs', type=int, default=500)
noisy_sgd_parser.add_argument('--batch-size', type=int, default=64)

subparsers = parser.add_subparsers(help="generative model type", dest="model")

parser_pate_gan = subparsers.add_parser('pate-gan', parents=[privacy_parser])
parser_pate_gan.add_argument('--lap-scale', type=float,
                             default=0.0001, help='Inverse laplace noise scale multiplier. A larger lap_scale will '
                                                  'reduce the noise that is added per iteration of training.')
parser_pate_gan.add_argument('--batch-size', type=int, default=64)
parser_pate_gan.add_argument('--num-teachers', type=int, default=10, help="Number of teacher disciminators in the pate-gan model")
parser_pate_gan.add_argument('--teacher-iters', type=int, default=5, help="Teacher iterations during training per generator iteration")
parser_pate_gan.add_argument('--student-iters', type=int, default=5, help="Student iterations during training per generator iteration")
parser_pate_gan.add_argument('--num-moments', type=int, default=100, help="Number of higher moments to use for epsilon calculation for pate-gan")

parser_ron_gauss = subparsers.add_parser('ron-gauss', parents=[privacy_parser])

parser_pgm = subparsers.add_parser('private-pgm', parents=[privacy_parser])

parser_real_data = subparsers.add_parser('real-data')

parser_imle = subparsers.add_parser('imle', parents=[privacy_parser, noisy_sgd_parser])
parser_imle.add_argument('--decay-step', type=int, default=25)
parser_imle.add_argument('--decay-rate', type=float, default=1.0)
parser_imle.add_argument('--staleness', type=int, default=5, help="Number of iterations after which new synthetic samples are generated")
parser_imle.add_argument('--num-samples-factor', type=int, default=10, help="Number of synthetic samples generated per real data point")

parser_dp_wgan = subparsers.add_parser('dp-wgan', parents=[privacy_parser, noisy_sgd_parser])
parser_dp_wgan.add_argument('--clamp-lower', type=float, default=-0.01, help="Clamp parameter for wasserstein GAN")
parser_dp_wgan.add_argument('--clamp-upper', type=float, default=0.01, help="Clamp parameter for wasserstein GAN")

opt = parser.parse_args()

# Loading the data
train = pd.read_csv(opt.train_data_path)
test = pd.read_csv(opt.test_data_path)

data_columns = [col for col in train.columns if col != opt.target_variable]
if opt.categorical:
    combined = train.append(test)
    config = {}
    for col in combined.columns:
        col_count = len(combined[col].unique())
        config[col] = col_count

class_ratios = None

if opt.downstream_task == "classification":
    class_ratios = train[opt.target_variable].sort_values().groupby(train[opt.target_variable]).size().values/train.shape[0]


X_train = np.nan_to_num(train.drop([opt.target_variable], axis=1).values)
y_train = np.nan_to_num(train[opt.target_variable].values)
X_test = np.nan_to_num(test.drop([opt.target_variable], axis=1).values)
y_test = np.nan_to_num(test[opt.target_variable].values)

if opt.normalize_data:
    X_train = expit(X_train)
    X_test = expit(X_test)

input_dim = X_train.shape[1]
z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

conditional = (opt.downstream_task == "classification")

# Training the generative model
if opt.model == 'pate-gan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

    model = pate_gan.PATE_GAN(input_dim, z_dim, opt.num_teachers, opt.target_epsilon, opt.target_delta, conditional)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, num_teacher_iters=opt.teacher_iters,
                                              num_student_iters=opt.student_iters, num_moments=opt.num_moments,
                                              lap_scale=opt.lap_scale, class_ratios=class_ratios, lr=1e-4))

elif opt.model == 'dp-wgan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

    model = dp_wgan.DP_WGAN(input_dim, z_dim, opt.target_epsilon, opt.target_delta, conditional)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, micro_batch_size=opt.micro_batch_size,
                                              clamp_lower=opt.clamp_lower, clamp_upper=opt.clamp_upper,
                                              clip_coeff=opt.clip_coeff, sigma=opt.sigma, class_ratios=class_ratios, lr=
                                              5e-5, num_epochs=opt.num_epochs), private=opt.enable_privacy)

elif opt.model == 'ron-gauss':
    model = ron_gauss.RONGauss(z_dim, opt.target_epsilon, opt.target_delta, conditional)

elif opt.model == 'imle':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'lr batch_size micro_batch_size sigma num_epochs class_ratios clip_coeff decay_step decay_rate staleness num_samples_factor')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

    model = imle.IMLE(input_dim, z_dim, opt.target_epsilon, opt.target_delta, conditional)
    model.train(X_train, y_train, Hyperparams(lr=1e-3, batch_size=opt.batch_size, micro_batch_size=opt.micro_batch_size,
                                              sigma=opt.sigma, num_epochs=opt.num_epochs, class_ratios=class_ratios,
                                              clip_coeff=opt.clip_coeff, decay_step=opt.decay_step,
                                              decay_rate=opt.decay_rate, staleness=opt.staleness,
                                              num_samples_factor=opt.num_samples_factor), private=opt.enable_privacy)

elif opt.model == 'private-pgm':
    if not conditional:
        raise Exception('Private PGM cannot be used to generate data for regression')
    model = private_pgm.Private_PGM(opt.target_variable, opt.target_epsilon, opt.target_delta)
    model.train(train, config)

# Generating synthetic data from the trained model
if opt.model == 'real-data':
    X_syn = X_train
    y_syn = y_train

elif opt.model == 'ron-gauss':

    if conditional:
        X_syn, y_syn, dp_mean_dict = model.generate(X_train, y=y_train)
        for label in np.unique(y_test):
            idx = np.where(y_test == label)
            x_class = X_test[idx]
            x_norm = preprocessing.normalize(x_class)
            x_bar = x_norm - dp_mean_dict[label]
            x_bar = preprocessing.normalize(x_bar)
            X_test[idx] = x_bar
    else:
        X_syn, y_syn, mu_dp = model.generate(X_train, y_train,
                                             max_y=np.max(np.concatenate([y_train,y_test], axis=0)))
        X_norm = preprocessing.normalize((X_test))
        X_bar = X_norm - mu_dp
        X_test = preprocessing.normalize(X_bar)

elif opt.model == 'imle' or opt.model == 'dp-wgan' or opt.model == 'pate-gan':
    syn_data = model.generate(X_train.shape[0], class_ratios)
    X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

elif opt.model == 'private-pgm':
    syn_data = model.generate()
    X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

# Testing the quality of synthetic data by training and testing the downstream learners

# Creating downstream learners
learners = []

if opt.downstream_task == "classification":
    names = ['LR', 'Random Forest', 'Neural Network', 'GaussianNB', 'GradientBoostingClassifier']

    learners.append((LogisticRegression()))
    learners.append((RandomForestClassifier()))
    learners.append((MLPClassifier(early_stopping=True)))
    learners.append((GaussianNB()))
    learners.append((GradientBoostingClassifier()))

    print("AUC scores of downstream classifiers on test data : ")
    for i in range(0, len(learners)):
        score = learners[i].fit(X_syn, y_syn)
        pred_probs = learners[i].predict_proba(X_test)
        auc_score = roc_auc_score(y_test, pred_probs[:, 1])
        print('-' * 40)
        print('{0}: {1}'.format(names[i], auc_score))

else:
    names = ['Ridge', 'Lasso', 'ElasticNet', 'Bagging', 'MLP']

    learners.append((Ridge()))
    learners.append((Lasso()))
    learners.append((ElasticNet()))
    learners.append((BaggingRegressor()))
    learners.append((MLPRegressor()))

    print("RMSE scores of downstream regressors on test data : ")
    for i in range(0, len(learners)):
        score = learners[i].fit(X_syn, y_syn)
        pred_vals = learners[i].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_vals))
        print('-' * 40)
        print('{0}: {1}'.format(names[i], rmse))

if opt.model != 'real-data':
    if opt.save_synthetic:

        if not os.path.isdir(opt.output_data_path):
            raise Exception('Output directory does not exist')

        X_syn_df = pd.DataFrame(data=X_syn, columns=data_columns)
        y_syn_df = pd.DataFrame(data=y_syn, columns=[opt.target_variable])

        syn_df = pd.concat([X_syn_df, y_syn_df], axis=1)
        syn_df.to_csv(opt.output_data_path + "/synthetic_data.csv")
        print("Saved synthetic data at : ", opt.output_data_path)



