import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from evaluation import compute_classifier_metrics


CHAIN_MAX_LEN = 150
EMBEDDING_DIM = 512

KERNEL_SIZE = 7
DILATION = 5
STRIDE = 1

with open("precomputed/embeddings.p", "rb") as f:
    embedding_dict = pickle.load(f)


def encode_seq(sequence):
    encoded = torch.zeros((CHAIN_MAX_LEN, EMBEDDING_DIM))
    encoded[:len(sequence)] = embedding_dict[sequence].squeeze(0)
    return encoded


def encode_batch(batch_of_sequences):
    encoded_seqs = [encode_seq(seq) for seq in batch_of_sequences]
    seq_lens = [len(seq) for seq in batch_of_sequences]
    return torch.stack(encoded_seqs), torch.as_tensor(seq_lens)


class Masked1dConvolution(nn.Module):
    def __init__(self, input_dim, in_channels, output_dim=None, out_channels=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.kernel_size = KERNEL_SIZE
        self.dilation = DILATION
        self.stride = STRIDE

        padding = (((self.output_dim - 1) * self.stride) + 1 - self.input_dim + (
                    self.dilation * (self.kernel_size - 1))) // 2

        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, padding, self.dilation)

    def forward(self, x, mask):
        return self.conv(x) * mask


def generate_mask(input_tensor, sequence_lengths):
    mask = torch.ones_like(input_tensor, dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        mask[i][length:, :] = False
    return mask


class Parabert(nn.Module):
    def __init__(self,
                 input_dim=EMBEDDING_DIM,
                 n_channels=CHAIN_MAX_LEN,
                 n_hidden_cells=1024):

        super().__init__()

        self.mconv = Masked1dConvolution(input_dim, n_channels)
        self.elu = nn.ELU()
        self.lstm = nn.LSTM(input_size=input_dim,
                            num_layers=1,
                            # dropout=0.4,
                            hidden_size=n_hidden_cells,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(n_hidden_cells * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, mask, lengths):
        # residual connection following ELU
        o = input_tensor + self.elu(self.mconv(input_tensor, mask))
        # Packing sequences to remove padding
        packed_seq = pack_padded_sequence(o, lengths, batch_first=True, enforce_sorted=False)
        o_packed, c = self.lstm(packed_seq)
        # Re-pad sequences before prediction of probabilities
        o, lengths = pad_packed_sequence(o_packed, batch_first=True, total_length=CHAIN_MAX_LEN)
        # Predict probabilities
        return self.sigmoid(self.fc(o))


def train(model, train_dl, val_dl, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in tqdm(range(epochs)):
        model.train()

        train_loss = 0.0

        for batch in train_dl:
            chains, labels = batch

            optimizer.zero_grad()

            sequences, lengths = encode_batch(chains)
            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64)

            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()

        train_loss = train_loss / len(train_dl)
        val_loss = evaluate(model, val_dl)

        print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))
        print("-----------------------------------------------")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return history


def evaluate(model, loader, eval_phase=True, cv=None):
    loss_fn = nn.BCELoss()
    model.eval()

    val_loss = 0.0
    all_outs = []
    all_lengths = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            chains, labels = batch

            sequences, lengths = encode_batch(chains)
            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64)

            loss = loss_fn(out, labels)
            val_loss += loss.data.item()

            all_outs.append(out)
            all_lengths.extend(lengths)
            all_labels.append(labels)

    if eval_phase:
        return val_loss / len(loader)
    else:
        return compute_classifier_metrics(torch.cat(all_outs), torch.cat(all_labels), all_lengths, cv)


def predict(chains):
    sequences, lengths = encode_batch(chains)
    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    model = torch.load("precomputed/chains.pth")

    model.eval()
    with torch.no_grad():
        probabilities = model(sequences, m, lengths)

    return probabilities.squeeze(2).type(torch.float64)
