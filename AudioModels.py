import torch.nn as nn
import torch.nn.functional as F

# TODO
class ConvAutoEncoder(nn.Module):
  def __init__(self, input_dim=40, embedding_dim=128):
    super(ConvAutoEncoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.enc = nn.Sequential(
      nn.BatchNorm2d(1),
      nn.Conv2d(1, 128, kernel_size=(input_dim, 5), stride=(1,1), padding=(0,2)),
      # nn.LeakyReLU(negative_slope=0.2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
      # nn.LeakyReLU(negative_slope=0.2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, 128, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
      # nn.LeakyReLU(negative_slope=0.2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.Conv2d(128, embedding_dim, kernel_size=(1,5), stride=(1,2), padding=(0,2)), 
    )
    self.dec = nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(embedding_dim),
      nn.ConvTranspose2d(embedding_dim, 128, kernel_size=(1,6), stride=(1,2), padding=(0,2)),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.ConvTranspose2d(128, 128, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.ConvTranspose2d(128, 128, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.ConvTranspose2d(128, input_dim, kernel_size=(1,5), stride=(1,1), padding=(0,2)) 
    )

  def forward(self, x, save_features=False):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    z = self.enc(x)
    x_r = self.dec(z).squeeze(2)
    if save_features:
      return z.squeeze(-2), x_r
    else:
      return x_r

class BLSTM2(nn.Module):
  def __init__(self, n_class, embedding_dim=100, n_layers=1):
    super(BLSTM2, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    # self.i2h = nn.Linear(40 + embedding_dim, embedding_dim)
    # self.i2o = nn.Linear(40 + embedding_dim, n_class) 
    self.rnn = nn.LSTM(input_size=40, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x.unsqueeze(0)

    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    embed, _ = self.rnn(x, (h0, c0))
    outputs = []
    for b in range(B):
      # out = self.softmax(self.fc(embed[b]))
      out = self.fc(embed[b])
      outputs.append(out)

    if save_features:
      return embed, torch.stack(outputs, dim=1)
    else:
      return torch.stack(outputs, dim=1)

class BLSTM3(nn.Module):
  def __init__(self, n_class, embedding_dim=100, n_layers=2, layer1_pretrain_file=None):
    super(BLSTM3, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn1 = BLSTM2(n_class, embedding_dim)
    if layer1_pretrain_file:
      self.rnn1.load_state_dict(torch.load(layer1_pretrain_file))
    self.rnn2 = nn.LSTM(input_size=2*embedding_dim, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)

  def forward(self, x, save_features=False):
    x, _ = self.rnn1(x, save_features=True) 
    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
    embed, _ = self.rnn2(x)
    outputs = [self.fc(embed[b]) for b in range(B)]

    if save_features:
      return embed, torch.stack(outputs, dim=1)
    else:
      return torch.stack(outputs, dim=1)
