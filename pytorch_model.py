import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from cnn import Masked1dConvolution, generate_mask
from preprocessing import encode_batch

# accepts CDRs up to length 28 + 2 residues either side
PARAPRED_MAX_LEN = 32

# 21 amino acids + 7 meiler features
PARAPRED_N_FEATURES = 28

# kernel size as per Parapred
PARAPRED_KERNEL_SIZE = 3

# threshold
Threshold = .73


class Parapred(nn.Module):
    def __init__(self, input_dim=PARAPRED_N_FEATURES, n_channels=PARAPRED_MAX_LEN, kernel_size=PARAPRED_KERNEL_SIZE,
                 n_hidden_cells=256):
        super().__init__()
        self.mconv = Masked1dConvolution(input_dim, in_channels=n_channels, kernel_size=kernel_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.15)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_hidden_cells, batch_first=True, bidirectional=True)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.fc = nn.Linear(n_hidden_cells * 2, 1)
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
        o = self.dropout(o)

        # Packing sequences to remove padding
        packed_seq = pack_padded_sequence(o, lengths, batch_first=True, enforce_sorted=False)

        o_packed, (h, c) = self.lstm(packed_seq)

        # Re-pad sequences before prediction of probabilities
        o, lengths = pad_packed_sequence(o_packed, batch_first=True, total_length=PARAPRED_MAX_LEN)

        o_droped = self.dropout_2(o)

        # Predict probabilities
        return self.sigmoid(self.fc(o_droped))


def clean_output(output_tensor: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    Clean the output tensor of probabilities to remove the predictions for padded positions

    :param output_tensor: output from the Parapred model; shape: (max_length x 1)
    :param sequence_length: length of sequence

    :return:
    """
    return output_tensor[:sequence_length].view(-1)


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=16):
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0
        train_paratopes = 0
        train_correct_paratopes = 0
        train_non_paratopes = 0
        train_correct_non_paratopes = 0

        for batch in train_dl:
            cdrs, lbls = batch

            optimizer.zero_grad()
            sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)
            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64)

            loss = loss_fn(out, lbls)
            loss.detach()

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            out = out.detach().apply_(lambda x: 0. if x < Threshold else 1.)
            for o, l in zip(out, lbls):
                num_train_correct += (o == l).sum().item()
                for i, l_i in enumerate(l):
                    if l_i == 1:
                        train_paratopes += 1
                        if l_i == o[i]:
                            train_correct_paratopes += 1
                    else:
                        train_non_paratopes += 1
                        if l_i == o[i]:
                            train_correct_non_paratopes += 1

            num_train_examples += len(cdrs) * PARAPRED_MAX_LEN

        paratope_acc = train_correct_paratopes / train_paratopes
        non_paratopes_acc = train_correct_non_paratopes / train_non_paratopes
        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dl)

        model.eval()

        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0
        val_paratopes = 0
        val_correct_paratopes = 0
        val_non_paratopes = 0
        val_correct_non_paratopes = 0

        for batch in val_dl:
            cdrs, lbls = batch

            sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)
            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64)

            loss = loss_fn(out, lbls)
            loss.detach()

            val_loss += loss.data.item()
            out = out.detach().apply_(lambda x: 0. if x < Threshold else 1.)
            for o, l in zip(out, lbls):
                num_val_correct += (o == l).sum().item()
                for i, l_i in enumerate(l):
                    if l_i == 1:
                        val_paratopes += 1
                        if l_i == o[i]:
                            val_correct_paratopes += 1
                    else:
                        val_non_paratopes += 1
                        if l_i == o[i]:
                            val_correct_non_paratopes += 1

            num_val_examples += len(cdrs) * PARAPRED_MAX_LEN

        val_paratope_acc = val_correct_paratopes / val_paratopes
        val_non_paratopes_acc = val_correct_non_paratopes / val_non_paratopes
        val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl)

        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
              (epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc))
        print(train_correct_paratopes, train_paratopes, train_correct_non_paratopes, train_non_paratopes)
        print(val_correct_paratopes, val_paratopes, val_correct_non_paratopes, val_non_paratopes)
        print('paratope %5.2f, non-paratopr %5.2f' % (paratope_acc, non_paratopes_acc))
        print('val-paratope %5.2f, val-non-paratopr %5.2f' % (val_paratope_acc, val_non_paratopes_acc))
        print("-----------------------------------------------------------------------------------")

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    return history


def evaluate(model, loader):
    model.eval()
    num_val_correct = 0
    num_val_examples = 0

    for batch in loader:
        cdrs, lbls = batch

        sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)
        # Generate a mask for the input
        m = generate_mask(sequences, sequence_lengths=lengths)
        probabilities = model(sequences, m, lengths)
        out = probabilities.squeeze(2).type(torch.float64)

        compute_classifier_metrics(out, lbls)

        num_val_correct += (out.detach().apply_(lambda x: 0. if x < Threshold else 1.) == lbls).sum().item()
        num_val_examples += len(cdrs)

    val_acc = num_val_correct / num_val_examples
    print('val acc: %5.2f' % val_acc)
