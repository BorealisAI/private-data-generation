# BSD 3-Clause License
#
# Copyright (c) 2017, Martin Arjovsky (NYU), Soumith Chintala (Facebook), Leon Bottou (Facebook)
# All rights reserved.
#
# This code has been modified from the original version at https://github.com/martinarjovsky/WassersteinGAN
# Modifications copyright (C) 2019-present, Royal Bank of Canada.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# dp_wgan.py implements the DP_WGAN generative model to generate private synthetic data
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from utils.architectures import Generator, Discriminator
from utils.helper import weights_init


class DP_WGAN:
    def __init__(self, input_dim, z_dim, target_epsilon, target_delta, conditional=True):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.generator = Generator(z_dim, input_dim, conditional).cuda().double()
        self.discriminator = Discriminator(input_dim, wasserstein=True).cuda().double()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional

    def train(self, x_train, y_train, hyperparams, private=False):
        batch_size = hyperparams.batch_size
        micro_batch_size = hyperparams.micro_batch_size
        lr = hyperparams.lr
        clamp_upper = hyperparams.clamp_upper
        clamp_lower = hyperparams.clamp_lower
        clip_coeff = hyperparams.clip_coeff
        sigma = hyperparams.sigma
        class_ratios = None

        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.cuda.DoubleTensor(x_train), torch.cuda.DoubleTensor(y_train)),
                                            batch_size=batch_size, shuffle=True)

        optimizer_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=lr)

        one = torch.cuda.DoubleTensor([1])
        mone = one * -1
        epsilon = 0
        gen_iters = 0
        steps = 0
        epoch = 0

        while epsilon < self.target_epsilon:

            data_iter = iter(data_loader)
            i = 0

            while i < len(data_loader):

                # Update Critic

                for p in self.discriminator.parameters():
                    p.requires_grad = True

                if gen_iters < 25 or gen_iters % 500 == 0:
                    disc_iters = 100

                else:
                    disc_iters = 5

                j = 0
                while j < disc_iters and i < len(data_loader):
                    j += 1

                    # clamp parameters to a cube

                    for p in self.discriminator.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)

                    data = data_iter.next()
                    i += 1

                    # train with real
                    optimizer_d.zero_grad()
                    inputs, categories = data
                    err_d_real = self.discriminator(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))

                    if private:
                        # For privacy, clip the avg gradient of each micro-batch

                        clipped_grads = {
                            name: torch.zeros_like(param) for name, param in self.discriminator.named_parameters()}

                        for k in range(int(err_d_real.size(0) / micro_batch_size)):
                            err_micro = err_d_real[k * micro_batch_size: (k + 1) * micro_batch_size].mean(0).view(1)
                            err_micro.backward(one, retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_coeff)
                            for name, param in self.discriminator.named_parameters():
                                clipped_grads[name] += param.grad
                            self.discriminator.zero_grad()

                        for name, param in self.discriminator.named_parameters():
                            # add noise here
                            param.grad = (clipped_grads[name] + torch.DoubleTensor(
                                clipped_grads[name].size()).normal_(0, sigma * clip_coeff).cuda()) / (
                                                     err_d_real.size(0) / micro_batch_size)

                        steps += 1

                    else:
                        err_d_real.mean(0).view(1).backward(one)

                    # train with fake
                    noise = torch.randn(batch_size, self.z_dim).cuda()
                    if self.conditional:
                        category = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).cuda().double()
                        fake = self.generator(torch.cat([noise.double(), category], dim=1))
                        err_d_fake = self.discriminator(torch.cat([fake.detach(), category], dim=1)).mean(0).view(1)

                    else:
                        fake = self.generator(noise.double())
                        err_d_fake = self.discriminator(fake.detach()).mean(0).view(1)
                    err_d_fake.backward(mone)
                    optimizer_d.step()

                # Update Generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                optimizer_g.zero_grad()
                noise = torch.randn(batch_size, self.z_dim).cuda()
                if self.conditional:
                    category = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).cuda().double()
                    fake = self.generator(torch.cat([noise.double(), category], dim=1))
                    err_g = self.discriminator(torch.cat([fake, category.double()], dim=1)).mean(0).view(1)
                else:
                    fake = self.generator(noise.double())
                    err_g = self.discriminator(fake).mean(0).view(1)
                err_g.backward(one)
                optimizer_g.step()
                gen_iters += 1

            epoch += 1
            if private:
                # Calculate the current privacy cost using the accountant
                max_lmbd = 4095
                lmbds = range(2, max_lmbd + 1)
                rdp = compute_rdp(batch_size / x_train.shape[0], sigma, steps, lmbds)
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            else:
                if epoch > hyperparams.num_epochs:
                    epsilon = np.inf

            print("Epoch :", epoch, "Loss D real : ", err_d_real.mean(0).view(1).item(),
                  "Loss D fake : ", err_d_fake.item(), "Loss G : ", err_g.item(), "Epsilon spent : ", epsilon)

    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps*batch_size < num_rows:
            noise = torch.randn(num_rows - steps*batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = torch.multinomial(class_ratios, num_rows - steps*batch_size, replacement=True).unsqueeze(1).cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)
