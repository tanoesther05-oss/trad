import numpy as np
import pandas as pd


def load_mt5_csv(file) -> pd.DataFrame:
    """Lit un export MT5 (tabulation ou virgule, colonnes <DATE> <TIME> <OPEN>...)."""
    df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError("Colonne de date/heure introuvable.")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")
    if "spread" not in df.columns:
        df["spread"] = 0
    return df[["time", "open", "high", "low", "close", "spread"]].sort_values("time").reset_index(drop=True)


def demo_data(n=20000, seed=7) -> pd.DataFrame:
    """Données M15 synthétiques (marche aléatoire avec dérive lente) pour tester l'interface."""
    rng = np.random.default_rng(seed)
    drift = np.sin(np.linspace(0, 12, n)) * 0.05
    close = 2000 + np.cumsum(rng.normal(drift, 1.2))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.random(n) * 0.8
    low = np.minimum(open_, close) - rng.random(n) * 0.8
    t = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low, "close": close, "spread": 0})


def add_indicators(df, atr_n=14):
    d = df.copy()
    for n in (20, 50, 200):
        d[f"ema{n}"] = d["close"].ewm(span=n, adjust=False).mean()
    pc = d["close"].shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / atr_n, adjust=False).mean()
    return d


def run_backtest(df, p):
    """p: dict de paramètres. Renvoie un DataFrame de trades."""
    d = add_indicators(df, p["atr_n"])
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    e20, e50, e200, atr = (d[k].to_numpy() for k in ("ema20", "ema50", "ema200", "atr"))
    hours = d["time"].dt.hour.to_numpy()
    sp = np.where(d["spread"].to_numpy() > 0, d["spread"].to_numpy() * p["point"], p["default_spread"])
    n = len(d)
    equity, trades, i = p["capital"], [], 200

    while i < n - 1:
        up = e50[i] > e200[i] and c[i] > e50[i]
        dn = e50[i] < e200[i] and c[i] < e50[i]
        direction = 0
        if up and c[i - 1] <= e20[i - 1] and c[i] > e20[i]:
            direction = 1
        elif dn and c[i - 1] >= e20[i - 1] and c[i] < e20[i]:
            direction = -1
        j = i + 1
        if direction == 0 or not (p["h_start"] <= hours[j] < p["h_end"]):
            i += 1
            continue

        dist = p["sl_atr"] * atr[i]
        s = sp[j]
        entry = o[j] + s if direction == 1 else o[j]
        sl = entry - direction * dist
        tp = entry + direction * dist * p["rr"]

        exit_px, reason, k = c[-1], "fin", n - 1
        for k in range(j, n):
            if direction == 1:
                hit_sl, hit_tp = l[k] <= sl, h[k] >= tp
            else:
                hit_sl, hit_tp = h[k] + sp[k] >= sl, l[k] + sp[k] <= tp
            if hit_sl:  # SL prioritaire (hypothèse prudente)
                exit_px, reason = sl, "SL"
                break
            if hit_tp:
                exit_px, reason = tp, "TP"
                break

        risk_money = equity * p["risk"]
        lots = np.floor(risk_money / (dist * p["contract"]) / 0.01) * 0.01
        lots = max(lots, p["min_lot"])
        pnl = direction * (exit_px - entry) * lots * p["contract"]
        equity += pnl
        trades.append({
            "entrée": d["time"][j], "sortie": d["time"][k], "sens": "Achat" if direction == 1 else "Vente",
            "prix entrée": entry, "SL": sl, "TP": tp, "prix sortie": exit_px, "résultat": reason,
            "lots": round(lots, 2), "risque réel %": lots * dist * p["contract"] / (equity - pnl) * 100,
            "R": direction * (exit_px - entry) / dist, "P&L $": pnl, "capital": equity,
        })
        i = k + 1 if k > i else i + 1
    return pd.DataFrame(trades)


def metrics(t, capital):
    if t.empty:
        return {"Trades": 0}
    wins, losses = t[t["P&L $"] > 0], t[t["P&L $"] <= 0]
    eq = np.r_[capital, t["capital"].to_numpy()]
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq)
    gl = abs(losses["P&L $"].sum())
    return {
        "Trades": len(t),
        "Win rate %": len(wins) / len(t) * 100,
        "Espérance (R)": t["R"].mean(),
        "Profit factor": wins["P&L $"].sum() / gl if gl else float("inf"),
        "Drawdown max $": dd.max(),
        "Drawdown max %": (dd / peak).max() * 100,
        "Profit net $": t["P&L $"].sum(),
    }
