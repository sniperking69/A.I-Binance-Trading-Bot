import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from binance.client import Client
from keys import api_key, api_secret
import math
from time import sleep
import os
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import multiprocessing

# Suppress FutureWarning and UserWarning for pin_memory
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*", category=UserWarning)

# Detect total CPU cores
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cores)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} with {num_cores} CPU threads")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

class CausalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(CausalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding)
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, 1, 1, 0)

        self.layer_norm = nn.LayerNorm([num_hidden, width, width]) if layer_norm else nn.Identity()

    def forward(self, x, h, c, m):
        if h is None: h = torch.zeros_like(x[:, :self.num_hidden, ...])
        if c is None: c = torch.zeros_like(h)
        if m is None: m = torch.zeros_like(h)

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        i_x, f_x, g_x, i_m, f_m, g_m, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_mh, f_mh, g_mh = torch.split(m_concat, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self._forget_bias)
        g = torch.tanh(g_x + g_h)
        delta_c = i * g
        c_new = f * c + delta_c

        i_m_comb = torch.sigmoid(i_m + i_mh)
        f_m_comb = torch.sigmoid(f_m + f_mh + self._forget_bias)
        g_m_comb = torch.tanh(g_m + g_mh)
        delta_m = i_m_comb * g_m_comb
        m_new = f_m_comb * m + delta_m

        mem_concat = torch.cat([c_new, m_new], dim=1)
        o = torch.sigmoid(o_x + o_h + self.conv_o(mem_concat))
        h_new = o * torch.tanh(self.layer_norm(c_new + m_new))

        return h_new, c_new, m_new, delta_c, delta_m

class PredRNNpp(nn.Module):
    def __init__(self, num_layers, num_hidden, in_channel, out_channel, width, filter_size=5):
        super(PredRNNpp, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.cell_list = nn.ModuleList([
            CausalLSTMCell(in_channel if i == 0 else num_hidden, num_hidden, width, filter_size, 1, True)
            for i in range(num_layers)
        ])
        self.conv_last = nn.Conv2d(num_hidden, out_channel, 1, 1, 0)

    def forward(self, frames):
        batch, time_len, _, height, width = frames.shape
        h_t = [torch.zeros(batch, self.num_hidden, height, width, device=frames.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch, self.num_hidden, height, width, device=frames.device) for _ in range(self.num_layers)]
        m_t = torch.zeros(batch, self.num_hidden, height, width, device=frames.device)

        next_frames, decouple_losses = [], []

        for t in range(time_len - 1):
            x = frames[:, t]
            h_t[0], c_t[0], m_t, delta_c, delta_m = self.cell_list[0](x, h_t[0], c_t[0], m_t)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], m_t, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)

            # Decoupling loss
            dc = delta_c.flatten(start_dim=1)
            dm = delta_m.flatten(start_dim=1)
            cos_sim = torch.abs((dc * dm).sum(dim=1) / (torch.norm(dc, dim=1) * torch.norm(dm, dim=1) + 1e-8))
            decouple_losses.append(cos_sim.mean())

        return torch.stack(next_frames, dim=1), torch.stack(decouple_losses).mean()


def marketDaysImage():
    """
    Creates 8-hour market matrices from token price changes and saves them as images.
    Each matrix represents one 8-hour period, where each pixel is a token's fractional price change.
    Tokens are arranged in alphabetical order in a square matrix.

    Returns:
        tuple: (period_matrices, dates, tokens)
        - period_matrices: dict with dates as keys and numpy arrays (matrices) as values
        - dates: list of dates in chronological order
        - tokens: list of token symbols in order
    """
    # Initialize Binance client
    client = Client(api_key, api_secret)
    exInfo = client.futures_exchange_info()
    tokens = sorted([symbol["symbol"] for symbol in exInfo["symbols"]
                    if symbol["contractType"] == "PERPETUAL" and "USDT" in symbol["symbol"]
                    and symbol["status"] == "TRADING"])

    # Create square grid dimensions
    num_tokens = len(tokens)
    grid_size = math.ceil(math.sqrt(num_tokens))
    total_cells = grid_size * grid_size

    # Initialize storage
    token_data = {}
    all_dates = set()

    # Fetch data for each token
    for symbol in tokens:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_8HOUR, limit=1500) # 30 periods = 240 hours = 10 days orginal was 1500
            df = pd.DataFrame(candles, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["date"] = pd.to_datetime(df["open_time"].astype(int), unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["pct_change"] = ((df["close"] - df["open"]) / df["open"]) * 100
            df = df.set_index("date")
            token_data[symbol] = df[["pct_change"]]
            all_dates.update(df.index)
            sleep(0.2)  # Avoid rate limits
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    # Create global date index and sort
    all_dates = sorted(list(set(all_dates)))

    # Reindex all token data to global date index
    for symbol in tokens:
        if symbol in token_data:
            token_data[symbol] = token_data[symbol].reindex(all_dates, fill_value=0)

    period_matrices = {}
    for date in all_dates:
        period_pct = [token_data[symbol].loc[date, "pct_change"] for symbol in tokens]
        if len(period_pct) < total_cells:
            period_pct += [0] * (total_cells - len(period_pct))
        matrix = np.array(period_pct).reshape((grid_size, grid_size))
        period_matrices[date] = matrix

    return period_matrices, all_dates, tokens

def train_pred_rnn(period_matrices, all_dates, num_epochs=1000, batch_size=32, model_path='pred_rnn_model.pth'):
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()  # Clear GPU cache

    # Convert matrices to tensor
    matrices = [period_matrices[date] for date in all_dates]
    matrices = np.array(matrices)
    matrices = matrices[:, np.newaxis, :, :]

    # Prepare sequences with batching
    seq_length = 10  # 10 periods = 80 hours = 3.33 days
    sequences = []
    for i in range(len(matrices) - seq_length):
        sequences.append(matrices[i:i+seq_length+1])  # input: seq_length, target: next frame

    sequences = torch.FloatTensor(np.array(sequences))

    # Set pin_memory only if using CUDA
    pin_memory = torch.cuda.is_available()
    # Set number of workers to 0 (original setting)
    num_workers = 0

    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(sequences)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    # Model parameters
    num_layers = 4
    num_hidden = 128
    in_channel = 1
    out_channel = 1
    width = matrices.shape[-1]

    model = PredRNNpp(num_layers, num_hidden, in_channel, out_channel, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start_epoch = 0

    # Try to load checkpoint if exists and shape matches
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}, continuing training.")
        except Exception as e:
            print(f"Could not load checkpoint (possible shape mismatch): {e}")
            print("Starting training from scratch.")

    # Training loop with batches
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch_seq = batch[0].to(device)  # shape: (batch, seq, 1, width, width)
            inputs = batch_seq[:, :-1]       # input frames
            targets = batch_seq[:, -1]       # target frame

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            pred = outputs[:, -1, 0]  # shape: [batch, width, width]
            targets = targets.squeeze(1)  # shape: [batch, width, width]

            # Handle pixel mismatch: crop or pad prediction to match target
            pred_shape = pred.shape
            target_shape = targets.shape[-2:]
            if pred_shape[-2:] != target_shape:
                min_h = min(pred_shape[-2], target_shape[0])
                min_w = min(pred_shape[-1], target_shape[1])
                pred = pred[..., :min_h, :min_w]
                targets = targets[..., :min_h, :min_w]

            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Print and save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    return model

def disp_predict(period_matrices, all_dates, model_path, num_layers=4, num_hidden=128):
    """
    Loads the model, predicts the upcoming period (T+1, next 8 hours) using the latest periods,
    saves the last 100 actual periods as images, and saves the prediction as well.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from datetime import datetime, timedelta
    import torch
    import numpy as np

    print("Current working directory:", os.getcwd())
    # Ensure the diagram directory exists (use absolute path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diagram_dir = os.path.join(script_dir, "diagram")
    os.makedirs(diagram_dir, exist_ok=True)

    custom_cmap = LinearSegmentedColormap.from_list(
        "smooth_red_black_green",
        [
            (0.0, "red"),
            (0.48, "#330000"),
            (0.5, "black"),
            (0.52, "#003300"),
            (1.0, "green"),
        ]
    )

    # --- Save last 100 actual periods as images ---
    for date in all_dates[-100:]:
        matrix = period_matrices[date]
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap=custom_cmap, interpolation='nearest', vmin=-1, vmax=1)
        plt.title(f"Actual {date}")
        plt.axis('off')
        plt.tight_layout()
        safe_date = date.replace(':', '-')
        try:
            plt.savefig(os.path.join(diagram_dir, f"actual_{safe_date}.png"))
        except Exception as e:
            print(f"Error saving image for {date}: {e}")
        plt.close()  # <-- Add this line

    # Load model
    width = list(period_matrices.values())[0].shape[0]
    in_channel = 1
    out_channel = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PredRNNpp(num_layers, num_hidden, in_channel, out_channel, width).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare matrices for prediction (use last 10 periods)
    last_dates = all_dates[-10:]
    matrices = np.array([period_matrices[d] for d in last_dates])[:, np.newaxis, :, :]
    matrices = torch.FloatTensor(matrices).to(device)

    with torch.no_grad():
        # Predict the next period (T+1, next 8 hours) using the last 10 periods
        pred_next, _ = model(matrices.unsqueeze(0))  # input: periods 0-9, predict period 10
        pred_next = pred_next[:, -1, 0].cpu().numpy().squeeze()

    # Save predicted upcoming period
    latest_date = all_dates[-1]
    next_date = (datetime.strptime(latest_date, "%Y-%m-%d %H:%M") + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_next, cmap=custom_cmap, interpolation='nearest', vmin=-1, vmax=1)
    plt.title(f"Predicted {next_date}")
    plt.axis('off')
    plt.tight_layout()
    safe_next_date = next_date.replace(':', '-')
    plt.savefig(os.path.join(diagram_dir, f"predicted_{safe_next_date}.png"))
    plt.close()
    print(f"Saved last 100 actual periods and prediction for {next_date} to 'diagram/' folder.")

# -- Script entry point --
if __name__ == "__main__":
    period_matrices, all_dates, tokens = marketDaysImage()
    # --- Set model parameters here to match your checkpoint ----
    # ---- parameters for server : batch size 4 and epochs 10 and limit 100---
    num_layers = 4
    num_hidden = 128
    model = train_pred_rnn(period_matrices, all_dates, num_epochs=200, batch_size=32, model_path='pred_rnn_model.pth')

    # Save model
    torch.save(model.state_dict(), 'pred_rnn_model.pth')
    print("Model saved as pred_rnn_model.pth")

    # Save visualizations and print accuracy
    #disp_predict(period_matrices, all_dates, 'pred_rnn_model.pth', num_layers=num_layers, num_hidden=num_hidden)