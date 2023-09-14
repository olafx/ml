'''
An implementation and showcase of the Adam optimizer on the MNIST handwritten
digit dataset.
https://arxiv.org/abs/1412.6980
http://yann.lecun.com/exdb/mnist/
'''

import torch
import torchvision

PREPROCESS_NORMALIZE = True
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_TESTING = 512

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

LSTM_HIDDEN_SIZE = 32
DENSE_SIZE = 32
DROPOUT_1_PROB = .25
DROPOUT_2_PROB = .5

class Model(torch.nn.Module):

  def __init__(self, lstm_hidden_size, dense_size, dropout_1_prob, dropout_2_prob):
    super(Model, self).__init__()
    self.LSTM_hor = torch.nn.LSTM(input_size=28, hidden_size=lstm_hidden_size, batch_first=True)
    self.LSTM_ver = torch.nn.LSTM(input_size=28, hidden_size=lstm_hidden_size, batch_first=True)
    self.batchnorm_1 = torch.nn.BatchNorm1d(2*lstm_hidden_size)
    self.dropout_1 = torch.nn.Dropout(dropout_1_prob)
    self.dropout_2 = torch.nn.Dropout(dropout_2_prob)
    self.fc_1 = torch.nn.Linear(2*lstm_hidden_size, dense_size)
    self.fc_2 = torch.nn.Linear(dense_size, 10)

  def forward(self, x):
    # Making an orthgonal copy.
    # (...,28,28) -> (...,28,28), (...,28,28)
    x_hor = x
    x_ver = torch.transpose(x_hor, dim0=1, dim1=2)
    x_ver = x_hor
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
    x = torch.nn.functional.gelu(x)
    x = self.dropout_2(x)
    # (...,dense_size) -> (...,10)
    x = self.fc_2(x)
    x = torch.nn.functional.log_softmax(x, dim=1)
    return x

model = Model(LSTM_HIDDEN_SIZE, DENSE_SIZE, DROPOUT_1_PROB, DROPOUT_2_PROB)

'''
The Adam optimizer, implemented as a user-defined torch optimizer.
'''

class Adam(torch.optim.Optimizer):

  def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8):
    self.exp_avgs = []
    self.exp_avg_sqs = []
    self.t = 0
    defaults = dict(lr=lr, betas=betas, eps=eps)
    super().__init__(params, defaults)

  def step(self):
    self.t += 1
    params_with_grad = []
    params = self.param_groups[0]['params'] # From the superclass.
    # The trainable parameter tensors are the ones that are not `None` after the
    # backwards pass. Getting these every step, just like torch.
    for param in params:
      if param.grad is not None:
        params_with_grad.append(param)
    # Initialize the moving averages.
    if self.t == 1:
      for param in params_with_grad:
        self.exp_avgs.append(torch.zeros_like(param, memory_format=torch.preserve_format))
        self.exp_avg_sqs.append(torch.zeros_like(param, memory_format=torch.preserve_format))
    # Optimize, following torch's numerical ordering.
    for i, param in enumerate(params_with_grad):
      lr = self.param_groups[0]['lr']
      beta_1, beta_2 = self.param_groups[0]['betas']
      eps = self.param_groups[0]['eps']
      grad = param.grad
      exp_avg = self.exp_avgs[i]
      exp_avg_sq = self.exp_avg_sqs[i]
      exp_avg *= beta_1
      exp_avg += (1-beta_1)*grad
      exp_avg_sq *= beta_2
      exp_avg_sq += (1-beta_2)*grad**2
      bias_corr_1 = 1-beta_1**self.t
      bias_corr_2 = 1-beta_2**self.t
      denom = exp_avg_sq.sqrt()/(-bias_corr_2**.5*lr/bias_corr_1)-eps*lr/bias_corr_1
      param.data += exp_avg/denom

adam_ = Adam(model.parameters())

'''
Training.
'''

N_EPOCHS = 2

loss_f = torch.nn.CrossEntropyLoss()
optim = Adam(model.parameters())
for i_epoch in range(N_EPOCHS):
  for i_batch, (images, labels) in enumerate(loader_train):
    # (...,1,28,28) -> (...,28,28)
    images = images.reshape(-1, 28, 28)
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
    images = images.reshape(-1, 28, 28)
    out = model(images)
    pred = out.argmax(dim=1, keepdim=True)
    correct += pred.eq(labels.view_as(pred)).sum().item()
  accuracy = correct/total
  print(f'maximum likelihood accuracy {accuracy:.4f}')
model.train()
