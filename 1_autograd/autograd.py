'''
An autograd engine. Based on https://github.com/karpathy/micrograd.
'''

import numpy as np
from uuid import uuid4
import unittest

class Node:

  def __init__(self, value, label=None, children=set(), operation=None):
    self.value = value
    self.children = children
    self.label = label if label else str(uuid4())
    self.operation = operation
    # Before a `Node` is connected to the loss `Node`, the derivative of the
    # loss with respect to such a `Node` is 0. It is only when the loss `Node`
    # is identified, by assigning its gradient to be 1 with respect to itself,
    # that some `Node`s obtain a nonzero gradient.
    self.grad = 0
    # The `backward_children` function acts to set the gradients of the children
    # via the chain rule. The gradient of the loss is as of yet unknown, but
    # this unknown quantity is the product of the gradient of the `Node` (0 for
    # now) and the derivative of the `Node` with respect to the child, which is
    # known, and depends on the nature of how this `Node` was constructed from
    # the children.
    self.backward_children = lambda: None

  def __repr__(self):
    ret = self.label
    if self.operation:
      ret += f' [{self.operation}]'
    ret += f' v={self.value:.2e} g={self.grad:.2e}'
    if self.children:
      ret += f' {self.children}'
    return ret

  def __add__(self, other):
    other = other if isinstance(other, Node) else Node(other, str(other))
    ret = Node(self.value+other.value, f'({self.label})+({other.label})', {self, other}, '+')
    def backward_children(): # Use additivity.
      # The gradients are always accumulated instead of assigned because if a
      # `Node` is used multiple times, i.e. has multiple parents, there will by
      # the chain rule be multiple contributions to the gradient.
      self.grad += 1*ret.grad
      other.grad += 1*ret.grad
    ret.backward_children = backward_children
    return ret

  def __radd__(self, other):
    return self+other

  def __mul__(self, other):
    other = other if isinstance(other, Node) else Node(other, str(other))
    ret = Node(self.value*other.value, f'({self.label})*({other.label})', {self, other}, '*')
    def backward_children(): # Use Leibniz rule.
      self.grad += other.value*ret.grad
      other.grad += self.value*ret.grad
    ret.backward_children = backward_children
    return ret

  def __rmul__(self, other):
    return self*other

  def __neg__(self):
    return -1*self

  def __sub__(self, other):
    return self+-other

  def __rsub__(self, other):
    return other+-self

  def __pow__(self, power:float):
    # `power` must be a constant, not a `Node`.
    ret = Node(self.value**power, f'({self.label})^{power}', {self}, f'^{power}')
    def backward_children(): # Use chain rule.
      self.grad += power*self.value**(power-1)*ret.grad
    ret.backward_children = backward_children
    return ret

  def __truediv__(self, other):
    return self*other**-1

  def __rtruediv__(self, other):
    return other*self**-1

  def exp(self):
    ret = Node(np.exp(self.value), f'exp({self.label})', {self}, 'exp')
    def backward_children(): # Use knowledge.
      self.grad += np.exp(self.value)*ret.grad
    ret.backward_children = backward_children
    return ret

  def tanh(self):
    ret = Node(np.tanh(self.value), f'tanh({self.label})', {self}, 'tanh')
    def backward_children(): # Use knowledge.
      self.grad += (1-np.tanh(self.value)**2)*ret.grad
    ret.backward_children = backward_children
    return ret

  def relu(self):
    ret = Node(0 if self.value < 0 else self.value, f'ReLU({self.label})', {self}, 'ReLU')
    def backward_children(): # Use knowledge.
      self.grad += 0 if self.value < 0 else 1*ret.grad
    ret.backward_children = backward_children
    return ret

  def backward(self):
    # Start the backward pass by identifying this `Node` as the loss.
    self.grad = 1
    # Perform a topological sort of the graph, so that if we iterate over the
    # `Node`s, each will be ready to have its gradient assigned due to its
    # parent already having a gradient assigned. (The algorithm used here is the
    # one based on depth-first search, adding children before `node` itself.)
    sorted, visited = [], set()
    def sort(node):
      if node not in visited:
        visited.add(node)
        for child in node.children:
          sort(child)
        sorted.append(node)
    sort(self)
    # Finally, set the gradients.
    for node in reversed(sorted):
      node.backward_children()

class NodeTest(unittest.TestCase):

  def setUp(self):
    self.a = Node(3, 'a')
    self.b = Node(2, 'b')

  def zero_grad(self, L:Node):
    L.grad = 0
    for child in L.children:
      self.zero_grad(child)

  def test_loss_gradient(self):
    self.a.backward()
    self.assertAlmostEqual(self.a.grad, 1)
    self.zero_grad(self.a)

  def test_addition(self):
    L = self.a+self.b
    L.backward()
    self.assertAlmostEqual(L.value, 2+3)
    self.assertAlmostEqual(self.a.grad, 1)
    self.assertAlmostEqual(self.b.grad, 1)
    self.zero_grad(L)

  def test_reverse_addition(self):
    L = 1+self.a
    L.backward()
    self.assertAlmostEqual(L.value, 4)
    self.assertAlmostEqual(self.a.grad, 1)
    self.zero_grad(L)

  def test_multiplication(self):
    L = self.a*self.b
    L.backward()
    self.assertAlmostEqual(L.value, 2*3)
    self.assertAlmostEqual(self.a.grad, 2)
    self.assertAlmostEqual(self.b.grad, 3)
    self.zero_grad(L)

  def test_multiplication_duplicate(self):
    L = self.a*self.a
    L.backward()
    self.assertAlmostEqual(L.value, 3*3)
    self.assertAlmostEqual(self.a.grad, 2*3)
    self.zero_grad(L)

  def test_reverse_multiplication(self):
    L = 2*self.a
    L.backward()
    self.assertAlmostEqual(L.value, 2*3)
    self.assertAlmostEqual(self.a.grad, 2)
    self.zero_grad(L)

  def test_power(self):
    L = self.a**2
    L.backward()
    self.assertAlmostEqual(L.value, 3**2)
    self.assertAlmostEqual(self.a.grad, 2*3)
    self.zero_grad(L)

  def test_division(self):
    L = self.a/2
    L.backward()
    self.assertAlmostEqual(L.value, 3/2)
    self.assertAlmostEqual(self.a.grad, 1/2)
    self.zero_grad(L)

  def test_reverse_division(self):
    L = 2/self.a
    L.backward()
    self.assertAlmostEqual(L.value, 2/3)
    self.assertAlmostEqual(self.a.grad, -2/3**2)
    self.zero_grad(L)

  def test_combined(self):
    L = 1+2*self.a
    L.backward()
    self.assertAlmostEqual(L.value, 7)
    self.assertAlmostEqual(self.a.grad, 2)
    self.zero_grad(L)

  def test_tanh_native(self):
    L = self.a.tanh()
    L.backward()
    self.assertAlmostEqual(L.value, np.tanh(3))
    self.assertAlmostEqual(self.a.grad, 1-np.tanh(3)**2)
    self.zero_grad(L)

  def test_tanh_nonnative(self):
    L = ((2*self.a).exp()-1)/((2*self.a).exp()+1)
    L.backward()
    self.assertAlmostEqual(L.value, np.tanh(3))
    self.assertAlmostEqual(self.a.grad, 1-np.tanh(3)**2)
    self.zero_grad(L)

  def test_negate(self):
    L = -self.a
    L.backward()
    self.assertAlmostEqual(L.value, -3)
    self.assertAlmostEqual(self.a.grad, -1)
    self.zero_grad(L)

if __name__ == '__main__':
  unittest.main()
