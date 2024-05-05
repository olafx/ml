'''
An implementation and showcase of the Adam optimizer on the MNIST handwritten
digit dataset.
https://arxiv.org/abs/1412.6980
http://yann.lecun.com/exdb/mnist/
'''

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

PREPROCESS_NORMALIZE = True
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 512
N_EPOCHS = 4

MODEL_LSTM_HIDDEN_SIZE = 32
MODEL_DENSE_SIZE = 32
MODEL_DROPOUT_1_PROB = .25
MODEL_DROPOUT_2_PROB = .5

transformer = [torchvision.transforms.ToTensor()]
if PREPROCESS_NORMALIZE:
  transformer += [torchvision.transforms.Normalize((.1307,), (.3081,))]
transformer = torchvision.transforms.Compose(transformer)
data_train = torchvision.datasets.MNIST('MNIST_image', train=True, download=True, transform=transformer)
data_test = torchvision.datasets.MNIST('MNIST_image', train=False, download=True, transform=transformer)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE_TRAINING, shuffle=True)
loader_test = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE_TESTING, shuffle=True)

'''
A RNN using 2 orthogonal LSTMs.
'''
class Model(torch.nn.Module):

  def __init__(self, lstm_hidden_size, dense_size, dropout_1_prob, dropout_2_prob):
    super(Model, self).__init__()
    self.LSTM_hor = nn.LSTM(input_size=28, hidden_size=lstm_hidden_size, batch_first=True)
    self.LSTM_ver = nn.LSTM(input_size=28, hidden_size=lstm_hidden_size, batch_first=True)
    self.batchnorm_1 = nn.BatchNorm1d(2*lstm_hidden_size)
    self.dropout_1 = nn.Dropout(dropout_1_prob)
    self.dropout_2 = nn.Dropout(dropout_2_prob)
    self.fc_1 = nn.Linear(2*lstm_hidden_size, dense_size)
    self.fc_2 = nn.Linear(dense_size, 10)

  def forward(self, x):
    # Making an orthgonal copy.
    # (...,28,28) -> (...,28,28), (...,28,28)
    x_hor = x
    x_ver = torch.einsum('...ij->...ji', x)
    # (...,28,28) -> (...,28,lstm_hidden_size)
    # (...,28,28) -> (...,28,lstm_hidden_size)
    x_hor, h_hor = self.LSTM_hor(x_hor)
    x_ver, h_ver = self.LSTM_ver(x_ver)
    # (...,28,lstm_hidden_size) -> (...,lstm_hidden_size)
    # (...,28,lstm_hidden_size) -> (...,lstm_hidden_size)
    x_hor = x_hor[:,-1,:]
    x_ver = x_ver[:,-1,:]
    # (...,lstm_hidden_size), (...,lstm_hidden_size) -> (...,2*lstm_hidden_size)
    x = torch.cat((x_hor, x_ver), dim=1)
    x = self.batchnorm_1(x)
    x = self.dropout_1(x)
    # (...,2*lstm_hidden_size) -> (...,dense_size)
    x = self.fc_1(x)
    x = F.gelu(x)
    x = self.dropout_2(x)
    # (...,dense_size) -> (...,10)
    x = self.fc_2(x)
    x = F.log_softmax(x, dim=1)
    return x

model = Model(MODEL_LSTM_HIDDEN_SIZE, MODEL_DENSE_SIZE, MODEL_DROPOUT_1_PROB, MODEL_DROPOUT_2_PROB)

'''
The Adam optimizer, implemented as a user-defined torch optimizer, closely
following torch's implementation.
'''
class Adam(torch.optim.Optimizer):

  def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8):
    super().__init__(params, defaults={'lr':lr, 'betas':betas, 'eps':eps})

  def step(self):

    for group in self.param_groups:
      for p in group['params']:
        # The trainable tensors are the ones with gradients that are not `None`
        # after the backward pass.
        if p.grad is not None:
          state = self.state[p]

          # Initialization of Adam's variables for this tensor.
          if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          state['step'] += 1

          lr, (beta_1, beta_2), eps = group['lr'], group['betas'], group['eps']
          step, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']

          # Now the actual optimization. The tensor operations here are inplace,
          # making it not very readable, so equivalent readable versions are
          # added as comments.
          bias_corr_1 = 1-beta_1**step
          bias_corr_2 = 1-beta_2**step
          # exp_avg.lerp_(p.grad.data, 1-beta_1)
          # exp_avg = exp_avg+(1-beta_1)*(p.grad.data-exp_avg)
          exp_avg = exp_avg*beta_1+(1-beta_1)*p.grad.data
          # exp_avg_sq.mul_(beta_2).addcmul_(p.grad.data, p.grad.data, value=1-beta_2)
          exp_avg_sq = exp_avg_sq*beta_2+(1-beta_2)*p.grad.data**2
          step_size_neg = -lr/bias_corr_1
          denom = (exp_avg_sq.sqrt()/(bias_corr_2**.5*step_size_neg)).add_(eps/step_size_neg)
          # p.data.addcdiv_(exp_avg, denom)
          p.data += exp_avg/denom

'''
Training.
'''
loss_f = nn.CrossEntropyLoss()
optim = Adam(model.parameters())
for i_epoch in range(N_EPOCHS):
  for i_batch, (images, labels) in enumerate(loader_train):
    # (...,1,28,28) -> (...,28,28)
    images = torch.einsum('...ijk->...jk', images)
    out = model(images)
    loss = loss_f(out, labels)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (i_batch+1) % 100 == 0 or i_batch == len(loader_train)-1:
      print(f'epoch {i_epoch+1}/{N_EPOCHS} batch {i_batch+1}/{len(loader_train)} batch size {len(images)} cross entropy loss {loss.item():.2e}')
  optim.param_groups[0]['lr'] /= 10

'''
Testing.
'''
model.eval()
with torch.no_grad():
  correct = 0
  total = len(loader_test.dataset)
  for images, labels in loader_test:
    # (...,1,28,28) -> (...,28,28)
    images = torch.einsum('...ijk->...jk', images)
    out = model(images)
    pred = out.argmax(dim=1, keepdim=True)
    correct += pred.eq(labels.view_as(pred)).sum().item()
  accuracy = correct/total
  print(f'maximum likelihood accuracy {accuracy:.4f}')
model.train()
