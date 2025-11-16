import time
from torch import nn
import torch
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


# from modules.AutoEncoder import AE, train

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Encoder, self).__init__()
        layer_dims = [input_dim] + hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]), nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]), nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]), nn.ReLU())

    def forward(self, input_data):
        hidden = self.layers(input_data)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        layer_dims = hidden_dims + [output_dim]
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]), nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]), nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]), nn.Tanh()
        )

    def forward(self, input_data):
        output = self.layers(input_data)
        return output


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AE, self).__init__()
        self.hidden_dims = hidden_dims
        self.criterion = nn.MSELoss(reduction="none")
        self.input_dim = input_dim
        # self.embedder = Embedder(vocab_size, embedding_dim)
        # self.bertembedder = BERTEmbedder()
        #
        # self.rnn = nn.LSTM(
        #     input_size=embedding_dim,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     # bidirectional=(self.num_directions == 2),
        # )
        self.encoder = Encoder(input_dim, hidden_dims)
        # self.clustering_layer = nn.Linear(hidden_dims[-1], num_clusters)
        # Use BERTEmbedder here
        self.decoder = Decoder(input_dim, list(reversed(hidden_dims)))

    def forward(self, input_data):

        #
        representation = input_data
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)

        if representation.dim() == 3:
            pred = self.criterion(representation, decoded).mean(dim=(-1, -2))
        elif representation.dim() == 2:
            pred = self.criterion(representation, decoded).mean(dim=-1)

        # pred = self.criterion(representation, decoded).mean(dim=(-1,-2))
        # pred = self.criterion(representation, representation).mean(dim=-1)
        # pred should be (n_sample),loss,should be (1)

        loss = pred.mean()
        # loss = pred.mean()
        return_dict = {"loss": loss,
                       "y_pred": pred,
                       "encoded": encoded,
                       "rep": encoded}
        return return_dict


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_preds = []
    epoch_time_start = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        res_dict = model(inputs)
        loss = torch.mean(res_dict["loss"])
        y_pred = res_dict["y_pred"]
        loss.backward()
        optimizer.step()

        y_preds.extend(y_pred.tolist())
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    # loss_history.append_loss(epoch_loss)
    # epoch_time_elapsed = time.time() - epoch_time_start

    return epoch_loss, y_preds


def infer_AE(model, data_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            res_dict = model(inputs)
            loss = res_dict["y_pred"]
            losses.extend(loss.tolist())
    return losses


if __name__ == '__main__':

    input_dim = 784
    hidden_dims = [256, 128, 64]
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = 10000
    x_data = torch.rand(n_samples, input_dim)
    y_data = torch.zeros(n_samples)
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AE(input_dim, hidden_dims).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss, y_preds = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    infer(model, train_loader, device)
    torch.save(model.state_dict(), "autoencoder_model.pth")

