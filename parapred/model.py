import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from evaluation import compute_classifier_metrics
from parapred.structure_processor import encode_batch

# accepts CDRs up to length 28 + 2 residues either side
PARAPRED_MAX_LEN = 32
# PARAPRED_MAX_LEN = 150

# 21 amino acids + 7 meiler features
PARAPRED_N_FEATURES = 28

# kernel size as per Parapred
PARAPRED_KERNEL_SIZE = 3


class Masked1dConvolution(nn.Module):
    def __init__(self,
                 input_dim: int,
                 in_channels: int,
                 output_dim=None,
                 out_channels=None,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1,
                 ):
        """
        A "masked" 1d convolutional neural network.

        For an input tensor T (bsz x seq_len x features), apply a boolean mask M (bsz x seq_len x features) following
        convolution. This essentially "zeros out" some of the values following convolution.
        """

        super().__init__()

        # Assert same shape
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        # Determine the padding required for keeping the same sequence length
        assert dilation >= 1 and stride >= 1, "Dilation and stride must be >= 1."
        self.dilation, self.stride = dilation, stride
        self.kernel_size = kernel_size

        padding = self.determine_padding(self.input_dim, self.output_dim)

        self.conv = nn.Conv1d(
            in_channels,
            self.out_channels,
            self.kernel_size,
            padding=padding
        )

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: an input tensor of (bsz x in_channels x seqlen)
        :param mask: a mask tensor (boolean) of (bsz x out_channels x seqlen)
        :return:
        """
        assert x.shape == mask.shape, \
            f"Shape of input tensor ({x.size()[0]}, {x.size()[1]}, {x.size()[2]}) " \
            f"does not match mask shape ({mask.size()[0]}, {mask.size()[1]}, {mask.size()[2]})."

        # Run through a regular convolution
        o = self.conv(x)

        # Apply the mask to "zero out" positions beyond sequence length
        return o * mask

    def determine_padding(self, input_shape: int, output_shape: int) -> int:
        """
        Determine the padding required to keep the same length (i.e. padding='same' from Keras)
        https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html

        L_out = ((L_in + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1)

        :return: An integer defining the amount of padding required to keep the "same" padding effect
        """
        padding = (((output_shape - 1) * self.stride) + 1 - input_shape + (self.dilation * (self.kernel_size - 1)))

        # integer division
        padding = padding // 2
        assert output_shape == l_out(
            input_shape, padding, self.dilation, self.kernel_size, self.stride
        ) and padding >= 0, f"Input and output of {input_shape} and {output_shape} with " \
                            f"kernel {self.kernel_size}, dilation {self.dilation}, stride {self.stride} " \
                            f"are incompatible for a Conv1D network."
        return padding


def generate_mask(input_tensor: torch.Tensor, sequence_lengths: torch.LongTensor) -> torch.Tensor:
    """
    Generate a mask for masked 1d convolution.

    :param input_tensor: an input tensor for convolution (bsz x features x seqlen)
    :param sequence_lengths: length of sequences (bsz,)
    :return:
    """
    assert input_tensor.size()[0] == sequence_lengths.size()[0], \
        f"Batch size {input_tensor.size()[0]} != number of provided lengths {sequence_lengths.size()[0]}."

    mask = torch.ones_like(input_tensor, dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        mask[i][length:, :] = False

    return mask


def l_out(l_in: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    """
    Determine the L_out of a 1d-CNN model given parameters for the 1D CNN

    :param l_in: length of input
    :param padding: number of units to pad
    :param dilation: dilation for CNN
    :param kernel: kernel size for CNN
    :param stride: stride size for CNN
    :return:
    """
    return (l_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class Parapred(nn.Module):
    def __init__(self,
                 input_dim=PARAPRED_N_FEATURES,
                 n_channels=PARAPRED_MAX_LEN,
                 kernel_size=PARAPRED_KERNEL_SIZE,
                 n_hidden_cells=256):
        super().__init__()
        self.mconv = Masked1dConvolution(input_dim, in_channels=n_channels, kernel_size=kernel_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.15)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_hidden_cells, batch_first=True, bidirectional=True)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.fc = nn.Linear(n_hidden_cells * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, mask, lengths):
        """
        Forward pass of Parapred given the input, mask, and sequence lengths

        :param input_tensor: an input tensor of (bsz x seq_len x features)
        :param mask: a boolean tensor of (bsz x seq_len x 1)
        :param lengths: a LongTensor of (seq_len); must be equal to bsz
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


def train(model, train_dl, val_dl, epochs=16):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for batch in train_dl:
            cdrs, lbls = batch

            optimizer.zero_grad()
            sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)
            # Generate a mask for the input
            m = generate_mask(sequences, sequence_lengths=lengths)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64)

            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()

        train_loss = train_loss / len(train_dl)
        val_loss = evaluate(model, val_dl)

        print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))
        print("-----------------------------------------------")

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return history


def evaluate(model, loader, eval=True):
    loss_fn = nn.BCELoss()
    model.eval()

    val_loss = 0.0
    all_outs = []
    all_labels = []

    for batch in loader:
        cdrs, lbls = batch

        sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)
        # Generate a mask for the input
        m = generate_mask(sequences, sequence_lengths=lengths)
        probabilities = model(sequences, m, lengths)
        out = probabilities.squeeze(2).type(torch.float64)

        loss = loss_fn(out, lbls)
        val_loss += loss.data.item()

        all_outs.append(out)
        all_labels.append(lbls)

    if eval:
        return val_loss / len(loader)
    else:
        compute_classifier_metrics(torch.cat(all_outs), torch.cat(all_labels))


def predict(cdrs):
    sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    p = Parapred()
    p.load_state_dict(torch.load("precomputed/sabdab_new.pth"))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    return probabilities.squeeze(2).type(torch.float64)
