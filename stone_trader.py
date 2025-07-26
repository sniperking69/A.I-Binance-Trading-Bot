import torch
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta, timezone
import os
import json
from time import sleep
from keys import api_key, api_secret
from stonevision import PredRNNpp

# --- Config ---
model_path = "pred_rnn_model.pth"
num_layers = 4
num_hidden = 128
in_channel = 2   # z_open + z_close input
out_channel = 1  # predict z_close
grid_size = 28
max_positions = 8
mode = 'S'  # 'S'=Simulation, 'R'=Real Trading
TRADED_BUFFER_FILE = "traded_buffer.json"

# --- Device & API ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = Client(api_key, api_secret)

# --- Helpers ---
def load_traded_buffer():
    if os.path.exists(TRADED_BUFFER_FILE):
        with open(TRADED_BUFFER_FILE, "r") as f:
            data = json.load(f)
            return {k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S%z") for k, v in data.items()}
    return {}

def save_traded_buffer(buffer):
    data = {k: v.strftime("%Y-%m-%d %H:%M:%S%z") for k, v in buffer.items()}
    with open(TRADED_BUFFER_FILE, "w") as f:
        json.dump(data, f)

def get_tokens():
    exInfo = client.futures_exchange_info()
    return sorted([
        s["symbol"] for s in exInfo["symbols"]
        if s["contractType"] == "PERPETUAL" and "USDT" in s["symbol"] and s["status"] == "TRADING"
    ])

def get_token_matrix(tokens, periods=40):
    token_data, all_dates = {}, set()
    for symbol in tokens:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=periods)
            df = pd.DataFrame(candles, columns=["open_time", "open", "high", "low", "close", "volume",
                                                "close_time", "quote_asset_volume", "number_of_trades",
                                                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
            df["date"] = pd.to_datetime(df["open_time"].astype(int), unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["z_open"] = (df["open"] - df["open"].mean()) / (df["open"].std() + 1e-6)
            df["z_close"] = (df["close"] - df["close"].mean()) / (df["close"].std() + 1e-6)
            df["pct_change"] = df["close"].pct_change() * 100  # Percent change
            df = df.set_index("date")
            token_data[symbol] = df[["z_open", "z_close", "pct_change"]]
            all_dates.update(df.index)
            sleep(0.2)
        except:
            continue
    all_dates = sorted(list(all_dates))[-periods:]
    print("Last 2 complete candle timestamps:", all_dates[-2:])

    total_cells = grid_size ** 2
    matrices = []
    for date in all_dates:
        z_open = [token_data.get(sym, pd.DataFrame()).get("z_open", pd.Series()).get(date, 0) for sym in tokens]
        z_close = [token_data.get(sym, pd.DataFrame()).get("z_close", pd.Series()).get(date, 0) for sym in tokens]
        z_open += [0] * (total_cells - len(z_open))
        z_close += [0] * (total_cells - len(z_close))
        matrix = np.stack([np.array(z_open).reshape(grid_size, grid_size),
                           np.array(z_close).reshape(grid_size, grid_size)], axis=0)
        matrices.append(matrix)
    return np.array(matrices), tokens, token_data

def compute_volatility(token_data, tokens):
    vol_dict = {}
    for token in tokens:
        try:
            last_changes = token_data[token]["pct_change"].dropna().iloc[-4:]
            volatility = np.std(last_changes)
            vol_dict[token] = volatility
        except:
            vol_dict[token] = 0
    return vol_dict

def place_order(client, symbol, side, quantity, precision, mode, reduce_only=False):
    if mode == 'R':
        try:
            order = client.futures_create_order(
                symbol=symbol, type=ORDER_TYPE_MARKET, side=side,
                quantity=quantity, reduceOnly=reduce_only)
            print(f"Order placed: {symbol}, {side}, Qty: {quantity}, Order ID: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            print(f"Order failed for {symbol}: {e}")
            return None
    else:
        print(f"Simulated order: {symbol}, {side}, Qty: {quantity}, ReduceOnly: {reduce_only}")
        return {'symbol': symbol, 'side': side, 'executedQty': str(quantity)}

def truncate(number, precision):
    return int(number * (10 ** precision)) / (10 ** precision)

def get_active_positions():
    return {p['symbol']: float(p['positionAmt']) for p in client.futures_position_information() if float(p['positionAmt']) != 0}

def get_token_info():
    exInfo = client.futures_exchange_info()
    return {
        s["symbol"]: {
            "quantityPrecision": s["quantityPrecision"],
            "minNotional": float(next(f["notional"] for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"))
        } for s in exInfo["symbols"] if s["contractType"] == "PERPETUAL" and "USDT" in s["symbol"] and s["status"] == "TRADING"
    }

def get_futures_balance():
    for asset in client.futures_account_balance():
        if asset['asset'] == 'USDT':
            return float(asset['balance'])
    return 0.0

def close_position(symbol):
    try:
        amt = float(client.futures_position_information(symbol=symbol)[0]['positionAmt'])
        if amt == 0: return
        side = SIDE_SELL if amt > 0 else SIDE_BUY
        qty = abs(amt)
        precision = get_token_info()[symbol]["quantityPrecision"]
        place_order(client, symbol, side, truncate(qty, precision), precision, mode, reduce_only=True)
    except: pass

def Lsafe(symbol, mrgType="ISOLATED", lvrg=2):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lvrg)
        client.futures_change_margin_type(symbol=symbol, marginType=mrgType)
    except Exception as e:
        if "-4046" not in str(e): print(f"Lsafe error: {e}")

# --- Main ---
if __name__ == "__main__":
    tokens = get_tokens()
    tinfo = get_token_info()
    traded_buffer = load_traded_buffer()
    now = datetime.now(timezone.utc)

    eligible_tokens = [t for t in tokens if t not in traded_buffer or (now - traded_buffer[t]) > timedelta(hours=12)]
    if not eligible_tokens:
        print("No eligible tokens to trade (all in 4h buffer).")
        exit(0)

    matrices, tokens, token_data = get_token_matrix(eligible_tokens, periods=40)
    if matrices.shape[0] < 10:
        print("Not enough data.")
        exit(0)

    volatility_map = compute_volatility(token_data, tokens)

    model = PredRNNpp(num_layers, num_hidden, in_channel, out_channel, grid_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    for symbol in get_active_positions():
        close_position(symbol)
    print("All positions closed.")

    balance = get_futures_balance()
    allocation = balance * 0.6

    input_seq = matrices[-10:]
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        output_seq = model(input_tensor)
        pred_next = output_seq.squeeze(0).cpu().numpy()

    avg_pred_next = np.mean(pred_next[-4:], axis=0)
    avg_pred_flat = avg_pred_next.flatten()

    signals = []
    for i, token in enumerate(tokens):
        avg_z = avg_pred_flat[i]
        volatility = volatility_map.get(token, 0)
        if abs(avg_z) < 0.5:  # Ignore weak signals
            continue
        score = abs(avg_z) * volatility  # Combined strength
        direction = "LONG" if avg_z > 0 else "SHORT"
        signals.append((token, direction, avg_z, volatility, score))

    signals_sorted = sorted(signals, key=lambda x: x[4], reverse=True)[:max_positions]

    if not signals_sorted:
        print("No strong signals to open positions.")
        exit(0)

    allocation_per = allocation / len(signals_sorted)
    for token, direction, zscore, vol, score in signals_sorted:
        try:
            opprice = float(client.futures_klines(symbol=token, interval=Client.KLINE_INTERVAL_1HOUR, limit=1)[-1][4])
            precision = tinfo[token]["quantityPrecision"]
            min_notional = tinfo[token]["minNotional"]
            qty = truncate(allocation_per / opprice, precision)
            if qty * opprice < min_notional:
                qty = truncate(min_notional / opprice, precision)
            side = SIDE_BUY if direction == "LONG" else SIDE_SELL
            Lsafe(token)
            print(f"[{direction}] {token}: Avg Z-Score={zscore:.2f}, Volatility={vol:.2f}, Score={score:.2f}")
            print(f"Placing {direction} for {token} Qty={qty} Notional={qty * opprice:.2f}")
            place_order(client, token, side, qty, precision, mode)
            traded_buffer[token] = datetime.now(timezone.utc)
            sleep(0.2)
        except Exception as e:
            print(f"Error placing order for {token}: {str(e)}")

    for token in list(traded_buffer.keys()):
        if (now - traded_buffer[token]) > timedelta(hours=12):
            del traded_buffer[token]
    save_traded_buffer(traded_buffer)
