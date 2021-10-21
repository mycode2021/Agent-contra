from torch.nn import Conv2d, ReLU, Linear, init
import torch.nn as nn
import numpy as np


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.num_input, self.channels, self.kernel, self.stride, self.padding = num_inputs, 32, 3, 2, 1
        self.fc = 32 * 6 * 6
        self.conv0 = Conv2d(out_channels=self.channels, 
                            kernel_size=self.kernel, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=[1, 1], 
                            groups=1, 
                            in_channels=num_inputs)
        self.relu0 = ReLU()
        self.conv1 = Conv2d(out_channels=self.channels, 
                            kernel_size=self.kernel, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=[1, 1], 
                            groups=1, 
                            in_channels=self.channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(out_channels=self.channels, 
                            kernel_size=self.kernel, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=[1, 1], 
                            groups=1, 
                            in_channels=self.channels)
        self.relu2 = ReLU()
        self.conv3 = Conv2d(out_channels=self.channels, 
                            kernel_size=self.kernel, 
                            stride=self.stride, 
                            padding=self.padding, 
                            dilation=[1, 1], 
                            groups=1, 
                            in_channels=self.channels)
        self.relu3 = ReLU()
        self.linear0 = Linear(in_features=int(self.fc), out_features=512)
        self.linear1 = Linear(in_features=512, out_features=num_actions)
        self.linear2 = Linear(in_features=512, out_features=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.relu0(self.conv0(x))
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.linear0(x.reshape([x.shape[0], -1]))
        return self.linear1(x), self.linear2(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape([input.shape[0], -1])

class RND(nn.Module):
    def __init__(self, input_size, output_size):
        super(RND, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=self.input_size,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.target = nn.Sequential(
            nn.Conv2d(in_channels=self.input_size,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )
  
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature

    def reward(self, next_obs):
        predict_next_feature, target_next_feature = self.forward(next_obs)
        intrinsic_reward = (target_next_feature-predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_reward.data.cpu().numpy()
