import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from cnn import Masked1dConvolution, generate_mask
import time

from main import MAX_PARAPRED_LEN
from preprocessing import encode_batch

# accepts CDRs up to length 28 + 2 residues either side
PARAPRED_MAX_LEN = 40

# 21 amino acids + 7 meiler features
PARAPRED_N_FEATURES = 28

# kernel size as per Parapred
PARAPRED_KERNEL_SIZE = 3


class Parapred(nn.Module):
    def __init__(self,
                 input_dim=PARAPRED_MAX_LEN,
                 output_dim=PARAPRED_MAX_LEN,
                 n_channels=PARAPRED_N_FEATURES,
                 kernel_size=PARAPRED_KERNEL_SIZE,
                 n_hidden_cells=256):

        super().__init__()
        self.mconv = Masked1dConvolution(input_dim, in_channels=n_channels, output_dim=output_dim, out_channels=n_channels, kernel_size=kernel_size)
        self.elu = nn.ELU()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=n_hidden_cells, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(n_hidden_cells*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor, mask: torch.BoolTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of Parapred given the input, mask, and sequence lengths

        :param input_tensor: an input tensor of (bsz x features x seqlen)
        :param mask: a boolean tensor of (bsz x 1 x seqlen)
        :param lengths: a LongTensor of (seqlen); must be equal to bsz
        :return:
        """
        # residual connection following ELU
        o = input_tensor + self.elu(self.mconv(input_tensor, mask))

        # Packing sequences to remove padding
        packed_seq = pack_padded_sequence(o.permute(0, 2, 1), lengths, batch_first=True, enforce_sorted=True)
        o_packed, (h, c) = self.lstm(packed_seq)

        # Re-pad sequences before prediction of probabilities
        o, lengths = pad_packed_sequence(o_packed, batch_first=True, total_length=PARAPRED_MAX_LEN)

        # Predict probabilities
        o = self.sigmoid(self.fc(o))

        return o


def clean_output(output_tensor: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    Clean the output tensor of probabilities to remove the predictions for padded positions

    :param output_tensor: output from the Parapred model; shape: (max_length x 1)
    :param sequence_length: length of sequence

    :return:
    """
    return output_tensor[:sequence_length].view(-1)


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20):
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    start_time_sec = time.time()

    for epoch in range(epochs):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0

        for batch in train_dl:
            cdrs, lbls = batch

            optimizer.zero_grad()
            sequences, lengths = encode_batch(cdrs, max_length=MAX_PARAPRED_LEN)

            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)

            probabilities = model(sequences, m, lengths)

            # Linearise probabilities for viewing
            out = {}
            clean = clean_output(probabilities, lengths[0]).tolist()
            for cdr in cdrs:
                i_prob = [round(_, 5) for i, _ in enumerate(clean)]
                seq_to_prob = list(zip(cdr, i_prob))
                out[cdr] = seq_to_prob

            loss = loss_fn(out, lbls)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * cdrs.size(0)
            num_train_correct += (torch.max(out, 1)[1] == lbls).sum().item()
            num_train_examples += cdrs.shape[0]

        # train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0

        for batch in val_dl:
            x = batch[0].squeeze(1).float()
            y = batch[2].squeeze(1).float()
            yhat = model(x)
            yhat = yhat.reshape(64, 32, 1)
            loss = loss_fn(yhat, y)

            val_loss += loss.data.item() * x.size(0)
            # num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            # num_val_examples += y.shape[0]

        # val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
              (epoch + 1, epochs, train_loss, 0, val_loss, 0))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        # history['acc'].append(train_acc)
        # history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP

    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % total_time_sec)
    print('Time per epoch: %5.2f sec' % time_per_epoch_sec)

    return history
