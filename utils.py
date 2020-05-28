import numpy as np


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def forward_backward(inputs, outputs, W_y, b_y, lstm_cells, seq_len, vocab_size, hprev1, cprev1):
    probs = {}
    loss = 0
    dW_y = np.zeros_like(W_y)
    db_y = np.zeros_like(b_y)
    hprev = {}
    cprev = {}
    num_lstm_layers = len(lstm_cells)

    for j in range(seq_len):
        x = np.zeros((vocab_size, 1))
        x[inputs[j]] = 1
        target = outputs[j]

        for k in range(num_lstm_layers):
            h1, c1 = lstm_cells[k].forward(x, j, hprev1[k], cprev1[k])
            x = np.copy(h1)
            hprev[k] = np.copy(h1)
            cprev[k] = np.copy(c1)

        # Take the last layer h1 and calculate y.
        y = np.dot(W_y, h1) + b_y

        probs[j] = softmax(y)

        loss += -np.log(probs[j][target])

    for j in reversed(range(seq_len)):
        dy = np.copy(probs[j])
        dy[outputs[j]] -= 1

        dW_y += np.dot(dy, lstm_cells[num_lstm_layers - 1].state_vars.h[j].T)
        db_y += dy

        dh = np.dot(W_y.T, dy)

        for k in reversed(range(num_lstm_layers)):
            dh = lstm_cells[k].backward(dh, j)

    return dW_y, db_y, loss, hprev, cprev


def sample(idx, W_y, b_y, lstm_cells, vocab_size, index_char, seq_len, hprev, cprev):
    seq = list()
    seq.append(index_char[idx])

    for i in range(seq_len):
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        for j in range(len(lstm_cells)):
            h, c = lstm_cells[j].forward(x, i, hprev[j], cprev[j])
            x = np.copy(h)

        y = np.dot(W_y, h) + b_y
        probs = softmax(y)
        idx = np.random.choice(range(vocab_size), p=probs.ravel())
        seq.append(index_char[idx])

    return seq

