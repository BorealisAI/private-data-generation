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
# architectures.py is used to create the generator and discriminator used in the generative models
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, output_size, conditional=True):
        super().__init__()
        z = latent_size
        d = output_size
        if conditional:
            z = z + 1
        else:
            d = d + 1
        self.main = nn.Sequential(
            nn.Linear(z, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, d))

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, wasserstein=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size + 1, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), 1))

        if not wasserstein:
            self.main.add_module(str(3), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)