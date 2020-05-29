import requests
from cifg_lstm import *
from utils import *


def main(**kwargs):
    """
    This script reads the data from a weblink (Shakespeare data) and trains a
    Unidirectional Multi-Layer CIFG-LSTM model.

    """
    num_lstm_layers = kwargs['nlayers']
    num_hidden_units = kwargs['hidden_units']
    seq_len = kwargs['seq_length']
    learning_rate = kwargs['lr']

    f = requests.get(kwargs['link'])
    data = f.text
    chars = list(set(data))
    vocab_size = len(chars)
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}

    lstm_cells = dict()
    input_dim = vocab_size

    for i in range(num_lstm_layers):
        lstm_cells[i] = LSTM(input_dim, num_hidden_units)
        input_dim = num_hidden_units

    Wy = np.random.randn(vocab_size, num_hidden_units) * 0.1
    by = np.zeros((vocab_size, 1)) * 0.1

    mWy = np.zeros_like(Wy)
    mby = np.zeros_like(by)

    # Exponential average of loss
    # Initialize to a error of a random model
    smooth_loss = -np.log(1.0 / vocab_size) * seq_len

    count = 0

    hprev = dict()
    cprev = dict()

    for i in range(num_lstm_layers):
        hprev[i] = np.zeros((num_hidden_units, 1))
        cprev[i] = np.zeros((num_hidden_units, 1))

    while True:

        for i in range(num_lstm_layers):
            hprev[i] = np.zeros((num_hidden_units, 1))
            cprev[i] = np.zeros((num_hidden_units, 1))

        for i in range(len(data) // seq_len):
            inputs = [char_to_index[ch] for ch in data[i * seq_len:(i + 1) * seq_len]]
            outputs = [char_to_index[ch] for ch in data[i * seq_len + 1:(i + 1) * seq_len + 1]]

            dWy, dby, loss, hprev, cprev = forward_backward(inputs, outputs, Wy, by, lstm_cells, seq_len, vocab_size,
                                                            hprev, cprev)

            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            clip_grad(dWy)
            clip_grad(dby)

            mWy += dWy * dWy
            mby += dby * dby

            Wy -= learning_rate * dWy / (np.sqrt(mWy + 1e-8))
            by -= learning_rate * dby / (np.sqrt(mby + 1e-8))

            for k in range(num_lstm_layers):
                lstm_cells[k].params.step(learning_rate)

            if count % 100 == 0:
                seq = sample(char_to_index[data[(i + 1) * seq_len + 1]], Wy, by, lstm_cells, vocab_size, index_to_char,
                             100, hprev, cprev)
                txt = ''.join(ix for ix in seq)
                print("----\n %s \n----" % (txt,))
                print("iter %d, loss %f" % (count, smooth_loss))

            count += 1


if __name__ == '__main__':
    main(link='https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt',
         nlayers=2,
         hidden_units=256,
         seq_length=25,
         lr=1e-1
         )
