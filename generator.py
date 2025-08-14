import json
import time
import math
import random
import os
import subprocess
from datetime import datetime, timezone

# --- CONFIGURATION ---
STATE_FILE = "engine_state.json"
HISTORY_FILE = "stock_history.json"
SYMBOLS = ["AAPL", "GOOG", "AMZN", "MSFT", "NVDA"]

# --- MODEL PARAMETERS ---
# GARCH(1,1) parameters for volatility clustering
GARCH_OMEGA = 1e-7  # Baseline variance
GARCH_ALPHA = 0.06  # Reaction to previous price shock
GARCH_BETA = 0.88  # Persistence of volatility
GARCH_GAMMA = 0.06  # Asymmetric impact (leverage effect)

# Mean Reversion Strength (defines the width of the price bands)
# These correspond to the "Full" ranges in the Kotlin code.
REVERSION_RANGES = {
    "s1": (0.001, 0.02),
    "m1": (0.002, 0.05),
    "m5": (0.005, 0.07),
    "h1": (0.01, 0.10),
    "d1": (0.10, 0.10)  # Daily is a fixed, wide band
}


class SyntheticStream:
    """Holds the entire state for a single stock's simulation."""

    def __init__(self, initial_price=100.0):
        self.p = float(initial_price)

        # Anchors for mean reversion
        self.d1_open = self.p
        self.h1_open = self.p
        self.m5_open = self.p
        self.m1_open = self.p

        # Reversion strengths (half-life of the bands)
        self.reversion_strength = {
            "s1": random.uniform(*REVERSION_RANGES["s1"]) / 2.0,
            "m1": random.uniform(*REVERSION_RANGES["m1"]) / 2.0,
            "m5": random.uniform(*REVERSION_RANGES["m5"]) / 2.0,
            "h1": random.uniform(*REVERSION_RANGES["h1"]) / 2.0,
            "d1": random.uniform(*REVERSION_RANGES["d1"]) / 2.0,
        }

        # State for the GARCH volatility model
        self.garch_variance = 1e-5
        self.prev_return = 0.0

        # State for boundary enforcement
        self.boundary_trend = 0  # -1 for down pressure, +1 for up pressure

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        instance.__dict__.update(data)
        return instance


def _u_seasonality(minute_of_day):
    """Models the U-shaped volatility pattern of a trading day."""
    x = 2.0 * math.pi * (minute_of_day / 1440.0)
    return 0.75 + 0.5 * (math.cos(x) ** 2)


def _get_garch_volatility(prev_return, prev_variance):
    """Calculates the next volatility value based on the GARCH model."""
    # The GJR-GARCH model includes leverage effect (bad news increases vol more)
    leverage_effect = GARCH_GAMMA * (prev_return ** 2 if prev_return < 0 else 0.0)

    new_variance = (
            GARCH_OMEGA +
            GARCH_ALPHA * prev_return ** 2 +
            leverage_effect +
            GARCH_BETA * prev_variance
    )
    return max(1e-9, math.sqrt(new_variance)), new_variance


def _generate_one_second(st, now_dt):
    """Generates a single bar of data using all modern principles."""
    minute_of_day = now_dt.hour * 60 + now_dt.minute

    # 1. Update Time-Based Anchors
    if now_dt.second == 0:
        st.m1_open = st.p
        if now_dt.minute % 5 == 0: st.m5_open = st.p
        if now_dt.minute == 0: st.h1_open = st.p
        if now_dt.hour == 0 and now_dt.minute == 0: st.d1_open = st.p

    # 2. Calculate Volatility for this Second (GARCH + Seasonality)
    sigma, new_variance = _get_garch_volatility(st.prev_return, st.garch_variance)
    st.garch_variance = new_variance
    seasonal_vol = sigma * _u_seasonality(minute_of_day)

    # 3. Define the Price Boundaries (Multi-Scale Mean Reversion)
    # The "safe" price is bounded by the tightest of all timeframe bands.
    s1_open = st.p  # The 1-second anchor is always the last price

    bands = {
        "s1": (s1_open * (1 - st.reversion_strength["s1"]), s1_open * (1 + st.reversion_strength["s1"])),
        "m1": (st.m1_open * (1 - st.reversion_strength["m1"]), st.m1_open * (1 + st.reversion_strength["m1"])),
        "m5": (st.m5_open * (1 - st.reversion_strength["m5"]), st.m5_open * (1 + st.reversion_strength["m5"])),
        "h1": (st.h1_open * (1 - st.reversion_strength["h1"]), st.h1_open * (1 + st.reversion_strength["h1"])),
        "d1": (st.d1_open * (1 - st.reversion_strength["d1"]), st.d1_open * (1 + st.reversion_strength["d1"])),
    }

    # The final boundary is the most restrictive combination of all bands
    lower_bound = max(b[0] for b in bands.values())
    upper_bound = min(b[1] for b in bands.values())

    # 4. Generate Random Shock and Apply Boundary Force
    base_return = random.gauss(0, 1) * seasonal_vol

    # Apply a drift based on the boundary trend memory
    boundary_drift = st.boundary_trend * seasonal_vol * 0.1

    total_return = base_return + boundary_drift

    o = st.p
    c = o * math.exp(total_return)

    # 5. Enforce Strict Boundaries and Update Trend
    if c >= upper_bound:
        c = upper_bound
        st.boundary_trend = -1  # Force price down
    elif c <= lower_bound:
        c = lower_bound
        st.boundary_trend = 1  # Force price up
    # If price returns to the middle of the channel, ease the trend pressure
    elif st.boundary_trend != 0:
        channel_mid = (upper_bound + lower_bound) / 2
        if (st.boundary_trend == -1 and c < channel_mid) or \
                (st.boundary_trend == 1 and c > channel_mid):
            st.boundary_trend = 0

    st.p = c
    st.prev_return = math.log(c / o) if o != 0 else 0

    return {
        "t": int(now_dt.timestamp() * 1000),
        "o": round(o, 2),
        "h": round(max(o, c), 2),
        "l": round(min(o, c), 2),
        "c": round(c, 2)
    }


def main_loop():
    print("Starting 24/7 high-fidelity price generation server...")

    # Load previous state or initialize new state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            states = {s: SyntheticStream.from_dict(d) for s, d in json.load(f).items()}
        print(f"Loaded existing state for {len(states)} symbols.")
    else:
        states = {s: SyntheticStream(random.uniform(50, 200)) for s in SYMBOLS}
        print(f"Initialized new state for {len(states)} symbols.")

    # Load history or initialize
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        print(f"Loaded existing history.")
    else:
        history = {s: [] for s in SYMBOLS}
        print("Initialized new history file.")

    last_save_minute = -1
    while True:
        now = datetime.now(timezone.utc)

        # Generate one second of data for each stock
        for symbol, stream in states.items():
            new_bar = _generate_one_second(stream, now)
            if symbol not in history: history[symbol] = []
            history[symbol].append(new_bar)

            # Trim history to prevent file from growing forever (e.g., keep last 3 days)
            if len(history[symbol]) > 3 * 24 * 60 * 60:
                history[symbol].pop(0)

        # Save state and push to git periodically (e.g., every minute)
        if now.minute != last_save_minute:
            last_save_minute = now.minute
            print(f"[{now.isoformat()}] Saving state and history. Pushing to remote. AAPL: ${states['AAPL'].p:.2f}")

            with open(STATE_FILE, 'w') as f:
                json.dump({s: v.to_dict() for s, v in states.items()}, f)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f)

            try:
                subprocess.run(["git", "add", HISTORY_FILE, STATE_FILE], check=True, capture_output=True)
                commit_msg = f"Data update {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
                subprocess.run(["git", "push"], check=True, capture_output=True)
                print("Successfully pushed to GitHub.")
            except subprocess.CalledProcessError as e:
                print(f"Error pushing to git: {e}\n{e.stderr.decode()}")
            except Exception as e:
                print(f"An unexpected error occurred during git push: {e}")

        # Sleep until the next whole second to maintain a steady tick rate
        time.sleep(1.0 - (datetime.now(timezone.utc).microsecond / 1_000_000.0))


if __name__ == "__main__":
    main_loop()