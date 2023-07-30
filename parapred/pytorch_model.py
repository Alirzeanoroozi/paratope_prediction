import torch
import torch.nn as nn
import time
from torch.autograd import Variable

NUM_FEATURES = 28


def false_neg(y_true, y_pred):
    return torch.squeeze(torch.clamp(y_true - torch.round(y_pred), 0.0, 1.0), dim=-1)


def false_pos(y_true, y_pred):
    return torch.squeeze(torch.clamp(torch.round(y_pred) - y_true, 0.0, 1.0), dim=-1)


class ab_seq_model(nn.Module):
    def __init__(self):
        super().__init__()
        # exd_mask =
        # return x * torch.unsqueeze(self.mask_func(x, mask), dim=-1).float()
        # self.masking_layer = MaskingByLambda(mask_by_input)
        # self.convolution_layer = MaskedConvolution1D(NUM_FEATURES, 28, kernel_size=3, padding=1)
        self.input_dim = 28
        self.hidden_dim = 256
        self.n_layers = 1
        self.batch_size = 1
        self.seq_len = 1

        inp = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        hidden_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim).double()
        cell_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim).double()

        hidden = (hidden_state, cell_state)
        self.bi_lstm = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.n_layers,
                               dropout=0.15,
                               bidirectional=True,
                               batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, res_fts):

        # seq = self.masking_layer(input_ab, label_mask)
        # loc_fts = self.convolution_layer(seq, label_mask)
        # res_fts = seq + loc_fts
        # print(res_fts.shape)
        glb_fts, _ = self.bi_lstm(res_fts)
        fts = self.dropout(glb_fts)
        probs = self.sigmoid(self.dense(fts))
        return probs.squeeze(2)


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cpu'):
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    start_time_sec = time.time()

    for epoch in range(epochs):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0

        for batch in train_dl:
            optimizer.zero_grad()

            x = batch[0].squeeze(1).type(torch.DoubleTensor)
            y = batch[2].squeeze(1)
            print(x.dtype)

            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * x.size(0)
            num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0

        for batch in val_dl:
            x = batch[0]
            y = batch[2]
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss += loss.data.item() * x.size(0)
            num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
              (epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP

    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % total_time_sec)
    print('Time per epoch: %5.2f sec' % time_per_epoch_sec)

    return history
