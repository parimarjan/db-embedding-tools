import torch
from torch import nn
import torch.nn.functional as F

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, n_input, n_hidden, n_output):
        super(SimpleRegression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=True),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        return output

class SiameseNetwork(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size,
            embedding_layer_size):
        super(SiameseNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size, bias=True),
            nn.ReLU()
        )

        # I guess we don't need to do RELU here?
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_layer_size, embedding_layer_size, bias=True),
        )

    def forward_once(self, x):
        output = self.layer1(x)
        # output = output.view(output.size()[0], -1)
        output = self.layer2(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # FIXME: need to check if this distance calculation is right
        F = nn.PairwiseDistance(p=2)
        # we want to add an empty last dimension
        output1 = output1.unsqueeze(output1.dim())
        output2 = output2.unsqueeze(output2.dim())
        euclidean_distance = F(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
