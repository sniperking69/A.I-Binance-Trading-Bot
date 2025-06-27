import torch
import numpy as np
import pandas as pd
import math
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from datetime import datetime
from time import sleep
from keys import api_key, api_secret
from stonevision import PredRNNpp

# --- Config ---
model_path = "pred_rnn_model.pth"
num_layers = 4
num_hidden = 128
in_channel = 1
out_channel = 1
signal_mode = "both"  # Options: "reversal", "trend", "both"
max_positions = 5
mode = 'D'  # 'S' for simulation, 'R' for real trading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = Client(api_key, api_secret)

def get_tokens():
    exInfo = client.futures_exchange_info()
    return sorted([
        s["symbol"] for s in exInfo["symbols"]
        if s["contractType"] == "PERPETUAL" and "USDT" in s["symbol"] and s["status"] == "TRADING"
    ])

def get_token_matrix(tokens, periods=20):
    token_data, all_dates = {}, set()
    for symbol in tokens:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_8HOUR, limit=periods)
            df = pd.DataFrame(candles, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                                "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                                "taker_buy_quote_asset_volume", "ignore"])
            df["date"] = pd.to_datetime(df["open_time"].astype(int), unit='ms').dt.strftime('%Y-%m-%d %H:%M')
            df["pct_change"] = ((df["close"].astype(float) - df["open"].astype(float)) / df["open"].astype(float)) * 100
            df = df.set_index("date")
            token_data[symbol] = df["pct_change"]
            all_dates.update(df.index)
        except:
            continue
    all_dates = sorted(list(all_dates))[-periods:]
    print("Last 2 complete candle timestamps:", all_dates[-2:])
    grid_size = math.ceil(math.sqrt(len(tokens)))
    matrices = []
    for date in all_dates:
        row = [token_data.get(sym, pd.Series()).get(date, 0) for sym in tokens]
        row += [0] * (grid_size ** 2 - len(row))
        matrices.append(np.array(row).reshape(grid_size, grid_size))
    return np.array(matrices), tokens, grid_size

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

if __name__ == "__main__":
    tokens = get_tokens()
    tinfo = get_token_info()
    matrices, tokens, width = get_token_matrix(tokens, periods=20)
    if matrices.shape[0] < 10:
        print("Not enough data.")
        exit(0)

    model = PredRNNpp(num_layers, num_hidden, in_channel, out_channel, width).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    print(f"Most recent actual candle (completed): {tokens[0]} @ {datetime.utcnow()} UTC")

    for symbol in get_active_positions():
        close_position(symbol)
    print("All positions closed.")

    balance = get_futures_balance()
    allocation = balance * 0.6

    input_seq = matrices[-10:]  # Use last 10 completed candles
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        output_seq, _ = model(input_tensor)
        output_np = output_seq.squeeze(0).squeeze(1).cpu().numpy()

    pred_next = output_np[-1]            # T+1 predicted candle (future)
    actual_current = input_seq[-1]       # T actual last complete candle

    pred_next_flat = pred_next.flatten()
    actual_current_flat = actual_current.flatten()

    reversal_signals, trend_signals = [], []
    for i, token in enumerate(tokens):
        p, n = actual_current_flat[i], pred_next_flat[i]
        if p < 0 and n > 0: reversal_signals.append((token, "LONG", p, n))
        elif p > 0 and n < 0: reversal_signals.append((token, "SHORT", p, n))
        if p > 0 and n > 0: trend_signals.append((token, "LONG", p, n))
        elif p < 0 and n < 0: trend_signals.append((token, "SHORT", p, n))

    signals = []
    if signal_mode == "reversal":
        signals = sorted(reversal_signals, key=lambda x: abs(x[3] - x[2]), reverse=True)
    elif signal_mode == "trend":
        signals = sorted(trend_signals, key=lambda x: abs(x[3] - x[2]), reverse=True)
    elif signal_mode == "both":
        reversal_sorted = sorted(reversal_signals, key=lambda x: abs(x[3] - x[2]), reverse=True)
        trend_sorted = sorted(trend_signals, key=lambda x: abs(x[3] - x[2]), reverse=True)
        signals = (reversal_sorted[:3] + trend_sorted[:2])[:max_positions]

    if not signals:
        print("No signals to open positions.")
        exit(0)

    allocation_per = allocation / len(signals)
    for token, direction, prev_val, next_val in signals:
        try:
            opprice = float(client.futures_klines(symbol=token, interval=Client.KLINE_INTERVAL_8HOUR, limit=1)[-1][4])
            precision = tinfo[token]["quantityPrecision"]
            min_notional = tinfo[token]["minNotional"]
            qty = truncate(allocation_per / opprice, precision)
            if qty * opprice < min_notional:
                qty = truncate(min_notional / opprice, precision)
            side = SIDE_BUY if direction == "LONG" else SIDE_SELL
            Lsafe(token)
            print(f"Placing {direction} for {token} Qty={qty} Notional={qty * opprice:.2f} | Change: {prev_val:.2f} → {next_val:.2f}")
            place_order(client, token, side, qty, precision, mode)
        except Exception as e:
            print(f"Error placing order for {token}: {str(e)}")
