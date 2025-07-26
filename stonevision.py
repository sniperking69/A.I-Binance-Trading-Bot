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
import warnings
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*", category=UserWarning)

# CPU/GPU setup
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} with {num_cores} CPU threads")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Fixed grid size
GRID_SIZE = 28  # 28x28 grid = 784 slots

def marketDaysImage():
    client = Client(api_key, api_secret)
    exInfo = client.futures_exchange_info()
    tokens = sorted([
        symbol["symbol"] for symbol in exInfo["symbols"]
        if symbol["contractType"] == "PERPETUAL" and "USDT" in symbol["symbol"]
        and symbol["status"] == "TRADING"
    ])
    total_cells = GRID_SIZE * GRID_SIZE
    token_data = {}
    all_dates = set()

    for symbol in tokens:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=1500)
            df = pd.DataFrame(candles, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
            df["date"] = pd.to_datetime(df["open_time"].astype(int), unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["z_open"] = (df["open"] - df["open"].mean()) / (df["open"].std() + 1e-6)
            df["z_close"] = (df["close"] - df["close"].mean()) / (df["close"].std() + 1e-6)
            df["z_open"] = df["z_open"].replace([np.inf, -np.inf], 0).fillna(0)
            df["z_close"] = df["z_close"].replace([np.inf, -np.inf], 0).fillna(0)
            df = df.set_index("date")
            token_data[symbol] = df[["z_open", "z_close"]]
            all_dates.update(df.index)
            sleep(0.2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    all_dates = sorted(all_dates)
    for symbol in token_data:
        token_data[symbol] = token_data[symbol].reindex(all_dates, fill_value=0)

    period_matrices = {}
    for date in all_dates:
        z_open_values = [token_data[sym].loc[date, "z_open"] for sym in tokens]
        z_close_values = [token_data[sym].loc[date, "z_close"] for sym in tokens]
        if len(z_open_values) < total_cells:
            z_open_values += [0] * (total_cells - len(z_open_values))
            z_close_values += [0] * (total_cells - len(z_close_values))
        matrix_open = np.array(z_open_values).reshape((GRID_SIZE, GRID_SIZE))
        matrix_close = np.array(z_close_values).reshape((GRID_SIZE, GRID_SIZE))
        matrix = np.stack([matrix_open, matrix_close], axis=0)
        period_matrices[date] = matrix

    return period_matrices, all_dates, tokens

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
        next_frames = []
        for t in range(time_len - 1):
            x = frames[:, t]
            h_t[0], c_t[0], m_t, _, _ = self.cell_list[0](x, h_t[0], c_t[0], m_t)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], m_t, _, _ = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], m_t)
            x_gen = self.conv_last(h_t[-1])
            next_frames.append(x_gen)
        return torch.stack(next_frames, dim=1)

def train_pred_rnn(period_matrices, all_dates, num_grand_epochs=10, chunk_size=100, batch_size=8, model_path='pred_rnn_model.pth'):
    torch.cuda.empty_cache()
    matrices = np.array([period_matrices[date] for date in all_dates])
    model = PredRNNpp(4, 128, in_channel=2, out_channel=1, width=GRID_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"✅ Loaded model from {model_path}, continuing training.")
        except:
            print("⚠️ Starting training from scratch (checkpoint mismatch).")

    for grand_epoch in range(num_grand_epochs):
        print(f"\n🌐 Grand Epoch [{grand_epoch+1}/{num_grand_epochs}]")
        grand_total_loss = 0
        total_batches = 0
        for start in range(0, len(matrices)-chunk_size+1, chunk_size):
            end = start + chunk_size
            sequences = [matrices[i:i+11] for i in range(start, end-10)]
            if not sequences:
                continue
            sequences = torch.FloatTensor(np.array(sequences))
            dataset = torch.utils.data.TensorDataset(sequences)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
            for batch in dataloader:
                inputs = batch[0][:, :-1].to(device)
                targets = batch[0][:, -1, 1, :, :].unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[:, -1], targets)
                loss.backward()
                optimizer.step()
                grand_total_loss += loss.item()
                total_batches += 1
        avg_grand_loss = grand_total_loss / total_batches
        print(f"✅ Grand Epoch [{grand_epoch+1}] Avg Loss: {avg_grand_loss:.5f}")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model checkpoint saved.")

    return model

if __name__ == "__main__":
    matrices, dates, tokens = marketDaysImage()
    model = train_pred_rnn(matrices, dates, num_grand_epochs=1, chunk_size=12, batch_size=6, model_path='pred_rnn_model.pth')
    print("🏁 Training complete and model saved.")
