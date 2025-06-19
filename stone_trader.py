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
num_hidden = 128  # Updated to match stonevision.py
in_channel = 1
out_channel = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_positions = 5  # <-- Set this manually
mode = 'R'  # 'S' for simulation, 'R' for real trading

client = Client(api_key, api_secret)

def get_tokens():
    exInfo = client.futures_exchange_info()
    tokens = sorted([
        symbol["symbol"] for symbol in exInfo["symbols"]
        if symbol["contractType"] == "PERPETUAL"
        and "USDT" in symbol["symbol"]
        and symbol["status"] == "TRADING"
    ])
    return tokens

def get_token_matrix(tokens, days=20):
    token_data = {}
    all_dates = set()
    for symbol in tokens:
        try:
            candles = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=days)
            df = pd.DataFrame(candles, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["date"] = pd.to_datetime(df["open_time"].astype(int), unit='ms').dt.strftime('%Y-%m-%d')
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["pct_change"] = ((df["close"] - df["open"]) / df["open"]) * 100  # Match stonevision.py
            df = df.set_index("date")
            token_data[symbol] = df["pct_change"]
            all_dates.update(df.index)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    all_dates = sorted(list(all_dates))[-days:]
    num_tokens = len(tokens)
    grid_size = math.ceil(math.sqrt(num_tokens))
    total_cells = grid_size * grid_size
    matrices = []
    for date in all_dates:
        day_pct = [token_data.get(symbol, pd.Series()).get(date, 0) for symbol in tokens]
        if len(day_pct) < total_cells:
            day_pct += [0] * (total_cells - len(day_pct))
        matrix = np.array(day_pct).reshape(grid_size, grid_size)
        matrices.append(matrix)
    return np.array(matrices), tokens, grid_size

def place_order(client, symbol, side, quantity, precision, mode, reduce_only=False):
    if mode == 'R':
        try:
            order = client.futures_create_order(
                symbol=symbol,
                type=ORDER_TYPE_MARKET,
                side=side,
                quantity=quantity,
                reduceOnly=reduce_only
            )
            print(f"Order placed: {symbol}, {side}, Qty: {quantity}, Order ID: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            print(f"Order failed for {symbol}: {e}")
            return None
    else:
        print(f"Simulated order: {symbol}, {side}, Qty: {quantity}, ReduceOnly: {reduce_only}")
        return {'symbol': symbol, 'side': side, 'executedQty': str(quantity)}

def truncate(number, precision):
    factor = 10.0 ** precision
    return int(number * factor) / factor

def get_active_positions(client):
    try:
        positions = client.futures_position_information()
        return {pos['symbol']: float(pos['positionAmt']) for pos in positions if float(pos['positionAmt']) != 0}
    except Exception:
        print("Error fetching positions.")
        return {}

def get_token_info():
    exInfo = client.futures_exchange_info()
    return {
        symbol["symbol"]: {
            "quantityPrecision": symbol["quantityPrecision"],
            "minNotional": float(next(f["notional"] for f in symbol["filters"] if f["filterType"] == "MIN_NOTIONAL"))
        }
        for symbol in exInfo["symbols"]
        if symbol["contractType"] == "PERPETUAL" and "USDT" in symbol["symbol"] and symbol["status"] == "TRADING"
    }

def get_futures_balance(client):
    try:
        balances = client.futures_account_balance()
        for asset in balances:
            if asset['asset'] == 'USDT':
                return float(asset['balance'])
    except Exception as e:
        print(f"Error fetching futures balance: {e}")
    return 0.0

def close_position(client, symbol, mode):
    try:
        pos_info = client.futures_position_information(symbol=symbol)
        amt = float(pos_info[0]['positionAmt'])
        if amt == 0:
            return
        side = SIDE_SELL if amt > 0 else SIDE_BUY
        qty = abs(amt)
        precision = 2  # Default, will try to get from token info
        tinfo = get_token_info()
        if symbol in tinfo:
            precision = tinfo[symbol]['quantityPrecision']
        qty = truncate(qty, precision)
        if qty > 0:
            print(f"Closing position for {symbol}: {side}, Qty: {qty}")
            place_order(client, symbol, side, qty, precision, mode, reduce_only=True)
    except Exception as e:
        print(f"Error closing position for {symbol}: {e}")

def Lsafe(client, symbol, mrgType="ISOLATED", lvrg=2):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lvrg)
        client.futures_change_margin_type(symbol=symbol, marginType=mrgType)
    except Exception as e:
        print(f"Lsafe error for {symbol}: {str(e)}")

if __name__ == "__main__":
    tokens = get_tokens()
    tinfo = get_token_info()
    matrices, tokens, width = get_token_matrix(tokens, days=20)  # Get width here

    model = PredRNNpp(num_layers, num_hidden, in_channel, out_channel, width)  # Use actual width
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    while True:
        # --- Close all open positions before opening new ones ---
        active_positions = get_active_positions(client)
        for symbol in active_positions:
            close_position(client, symbol, mode)
        print("All positions closed. Waiting 10 seconds before opening new ones...")
        sleep(10)

        # --- Get available balance and calculate allocation ---
        total_balance = get_futures_balance(client)
        allocation_balance = total_balance * 0.6  # Use 60% of balance
        matrices, tokens, width = get_token_matrix(tokens, days=20)
        if matrices.shape[0] < 20:
            print("Not enough historical data for prediction. Got:", matrices.shape[0])
            sleep(600)
            continue

        model.width = width  # set width if needed
        input_seq = matrices[:19]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            output_seq, _ = model(input_tensor)
            output_np = output_seq.squeeze(0).squeeze(1).cpu().numpy()
        if output_np.shape[0] < 2:
            print("Model did not return enough predictions.")
            sleep(600)
            continue

        pred_next = output_np[-1]  # T+1 prediction
        actual_current = matrices[9]   # T (latest actual)

        pred_next_flat = pred_next.flatten()
        actual_current_flat = actual_current.flatten()

        signals = []
        for idx, token in enumerate(tokens):
            prev_val = actual_current_flat[idx]
            next_val = pred_next_flat[idx]
            if prev_val < 0 and next_val > 0:
                signals.append((token, "LONG", prev_val, next_val))
            elif prev_val > 0 and next_val < 0:
                signals.append((token, "SHORT", prev_val, next_val))

        # Sort by absolute difference descending
        signals = sorted(signals, key=lambda x: abs(x[3] - x[2]), reverse=True)

        # --- Live trading logic ---
        num_positions = min(max_positions, len(signals))
        if num_positions > 0:
            allocation_per_position = allocation_balance / num_positions
            opened = 0
            for token, direction, prev_val, next_val in signals[:num_positions]:
                try:
                    # Get price and precision
                    raw_data = client.futures_klines(symbol=token, interval=Client.KLINE_INTERVAL_1DAY, limit=1)
                    opprice = float(raw_data[-1][4])
                    precision = tinfo.get(token, {}).get("quantityPrecision", 2)
                    min_notional = tinfo.get(token, {}).get("minNotional", 5.0)
                    qty = truncate(allocation_per_position / opprice, precision)
                    notional = qty * opprice
                    if notional < min_notional:
                        qty = truncate(min_notional / opprice, precision)
                        notional = qty * opprice
                    side = SIDE_BUY if direction == "LONG" else SIDE_SELL

                    # Set margin and leverage before placing order
                    Lsafe(client, token, mrgType="ISOLATED", lvrg=2)

                    print(f"Placing {direction} order for {token}: Qty={qty}, Notional={notional:.2f}")
                    place_order(client, token, side, qty, precision, mode)
                    opened += 1
                except Exception as e:
                    print(f"Order error for {token}: {str(e)}")
            print(f"Opened {opened} positions.")
        else:
            print("No signals to open positions.")

        print("Sleeping for 12 hours...")
        sleep(43200)