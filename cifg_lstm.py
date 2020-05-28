import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def clip_grad(param):
    return np.clip(param, -5, 5, out=param)


class Param:
    def __init__(self, input_dim, num_hidden_units):
        concat_len = input_dim + num_hidden_units
        weight_sd = 0.1  # Standard deviation of weights for initialization

        self.W_C = np.random.randn(num_hidden_units, concat_len) * weight_sd
        self.b_C = np.zeros((num_hidden_units, 1))
        self.dW_C = np.zeros_like(self.W_C)
        self.db_C = np.zeros_like(self.b_C)
        self.mW_C = np.zeros_like(self.W_C)
        self.mb_C = np.zeros_like(self.b_C)

        self.W_i = np.random.randn(num_hidden_units, concat_len) * weight_sd
        self.b_i = np.zeros((num_hidden_units, 1))
        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)
        self.mW_i = np.zeros_like(self.W_i)
        self.mb_i = np.zeros_like(self.b_i)

        self.W_o = np.random.randn(num_hidden_units, concat_len) * weight_sd
        self.b_o = np.zeros((num_hidden_units, 1))
        self.dW_o = np.zeros_like(self.W_o)
        self.db_o = np.zeros_like(self.b_o)
        self.mW_o = np.zeros_like(self.W_o)
        self.mb_o = np.zeros_like(self.b_o)

    def reinit(self):
        self.dW_C = np.zeros_like(self.W_C)
        self.db_C = np.zeros_like(self.b_C)

        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)

        self.dW_o = np.zeros_like(self.W_o)
        self.db_o = np.zeros_like(self.b_o)

    def step(self, learning_rate):
        for param in [self.dW_C, self.dW_i, self.dW_o, self.db_C, self.db_i, self.db_o]:
            clip_grad(param)

        self.mW_C += self.dW_C*self.dW_C
        self.mW_i += self.dW_i*self.dW_i
        self.mW_o += self.dW_o*self.dW_o

        self.mb_C += self.db_C*self.db_C
        self.mb_i += self.db_i*self.db_i
        self.mb_o += self.db_o*self.db_o

        self.W_C -= learning_rate*self.dW_C/(np.sqrt(self.mW_C + 1e-8))
        self.W_i -= learning_rate*self.dW_i/(np.sqrt(self.mW_i + 1e-8))
        self.W_o -= learning_rate*self.dW_o/(np.sqrt(self.mW_o + 1e-8))

        self.b_C -= learning_rate*self.db_C/(np.sqrt(self.mb_C + 1e-8))
        self.b_i -= learning_rate*self.db_i/(np.sqrt(self.mb_i + 1e-8))
        self.b_o -= learning_rate*self.db_o/(np.sqrt(self.mb_o + 1e-8))


class States:
    def __init__(self, num_hidden_units):
        self.xc = dict()
        self.Cbar = dict()
        self.h = dict()
        self.C = dict()
        self.ig = dict()
        self.og = dict()

        self.cprev = np.zeros((num_hidden_units, 1))
        self.hprev = np.zeros((num_hidden_units, 1))

        self.dCnext = np.zeros((num_hidden_units, 1))
        self.dhnext = np.zeros((num_hidden_units, 1))

    def reinit(self, num_hidden_units):
        self.xc = dict()
        self.Cbar = dict()
        self.h = dict()
        self.C = dict()
        self.ig = dict()
        self.og = dict()

        self.dCnext = np.zeros((num_hidden_units, 1))
        self.dhnext = np.zeros((num_hidden_units, 1))


class LSTM:
    def __init__(self, input_dim, num_hidden_units):
        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        self.params = Param(input_dim, num_hidden_units)
        self.state_vars = States(num_hidden_units)

    def forward(self, x, id, hprev, cprev):
        if id == 0:
            # First layer
            self.state_vars.reinit(self.num_hidden_units)
            self.params.reinit()
            self.state_vars.h[id - 1] = np.copy(hprev)
            self.state_vars.C[id - 1] = np.copy(cprev)

        states = self.state_vars
        states.xc[id] = np.resize(np.hstack((states.h[id-1].flatten(), x.flatten())),
                                  (self.input_dim + self.num_hidden_units, 1))
        states.Cbar[id] = np.tanh(np.dot(self.params.W_C, states.xc[id]) + self.params.b_C)
        states.ig[id] = sigmoid(np.dot(self.params.W_i, states.xc[id]) + self.params.b_i)
        states.og[id] = sigmoid(np.dot(self.params.W_o, states.xc[id]) + self.params.b_o)
        states.C[id] = states.ig[id]*states.Cbar[id] + (1 - states.ig[id])*states.C[id - 1]  # Coupled Input-Forget Gate
        states.h[id] = states.og[id] * np.tanh(states.C[id])

        return states.h[id], states.C[id]

    def backward(self, top_hdiff, i):
        states = self.state_vars
        dh = top_hdiff + states.dhnext
        C_p = (1 - (np.tanh(states.C[i])) ** 2)

        dC = dh * states.og[i] * C_p + states.dCnext

        dog = dh * np.tanh(states.C[i])
        dig = dC * states.C[i]
        dg = dC * states.ig[i]

        dog_in = dog * (states.og[i] * (1. - states.og[i]))
        dig_in = dig * (states.ig[i] * (1. - states.ig[i]))
        dg_in = dg * (1. - states.C[i] ** 2)

        self.params.dW_i += np.dot(dig_in, states.xc[i].T)
        self.params.dW_o += np.dot(dog_in, states.xc[i].T)
        self.params.dW_C += np.dot(dg_in, states.xc[i].T)

        self.params.db_o += dog_in
        self.params.db_i += dig_in
        self.params.db_C += dg_in

        dXc = np.dot(self.params.W_C.T, dg_in)
        dXc += np.dot(self.params.W_i.T, dig_in)
        dXc += np.dot(self.params.W_o.T, dog_in)

        states.dCnext = dC * (1 - states.ig[i])
        states.dhnext = dXc[:self.num_hidden_units]
        dx = dXc[self.num_hidden_units:]

        return dx




