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
# helper.py contains some helper utility functions
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def pate(data, netTD, lap_scale):
    results = torch.Tensor(len(netTD), data.size()[0]).type(torch.int64)
    for i in range(len(netTD)):
        output = netTD[i].forward(data)
        pred = (output > 0.5).type(torch.Tensor).squeeze()
        results[i] = pred

    clean_votes = torch.sum(results, dim=0).unsqueeze(1).type(torch.cuda.DoubleTensor)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1/lap_scale, size=clean_votes.size())).cuda()
    noisy_results = clean_votes + noise
    noisy_labels = (noisy_results > len(netTD)/2).type(torch.cuda.DoubleTensor)

    return noisy_labels, clean_votes


def moments_acc(num_teachers, clean_votes, lap_scale, l_list):
    q = (2 + lap_scale * torch.abs(2*clean_votes - num_teachers)
         )/(4 * torch.exp(lap_scale * torch.abs(2*clean_votes - num_teachers)))

    update = []
    for l in l_list:
        a = 2*lap_scale*lap_scale*l*(l + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2*lap_scale) * q), l)
        t_two = q * torch.exp(2*lap_scale * l)
        t = t_one + t_two
        update.append(torch.clamp(t, max=a).sum())

    return torch.cuda.DoubleTensor(update)


def mutual_information(labels_x: pd.Series, labels_y: pd.DataFrame):

    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.get_values()), axis=1)

    return mutual_info_score(labels_x, labels_y)


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)
