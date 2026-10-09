import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backtest import demo_data, load_mt5_csv, metrics, run_backtest

st.set_page_config(page_title="Backtester MT5", layout="wide")
st.title("📈 Backtester MT5")
st.caption("Simulation d'une seule stratégie sur données historiques. Aucun ordre n'est passé.")

with st.sidebar:
    st.header("Données")
    up = st.file_uploader("CSV exporté de MT5", type=["csv", "txt"])
    use_demo = st.checkbox("Utiliser des données de démo", value=up is None)
    st.header("Paramètres")
    capital = st.number_input("Capital ($)", 100.0, 1e7, 1000.0, 100.0)
    risk = st.number_input("Risque par trade (%)", 0.1, 10.0, 1.0, 0.1) / 100
    sl_atr = st.number_input("SL (× ATR)", 0.5, 5.0, 1.5, 0.1)
    rr = st.number_input("TP (× SL)", 0.5, 5.0, 1.5, 0.1)
    atr_n = st.number_input("Période ATR", 5, 50, 14)
    h_start, h_end = st.slider("Session (heure serveur)", 0, 24, (8, 20))
    st.subheader("Instrument")
    contract = st.number_input("Taille du contrat (1 lot =)", 1.0, 1e6, 100.0)
    point = st.number_input("Valeur d'un point de spread", 0.0001, 1.0, 0.01, format="%.4f")
    default_spread = st.number_input("Spread par défaut (prix)", 0.0, 10.0, 0.30, 0.05)
    min_lot = st.number_input("Lot minimum", 0.001, 1.0, 0.01, format="%.3f")
    split = st.slider("Part entraînement (%)", 50, 90, 70) / 100

try:
    df = demo_data() if (use_demo or up is None) else load_mt5_csv(up)
except Exception as e:
    st.error(f"Lecture impossible : {e}")
    st.stop()

if use_demo or up is None:
    st.info("Données de démo synthétiques : les résultats n'ont aucune valeur réelle.")
st.write(f"**{len(df):,} bougies** — du {df.time.iloc[0]:%d/%m/%Y} au {df.time.iloc[-1]:%d/%m/%Y}")

p = dict(capital=capital, risk=risk, sl_atr=sl_atr, rr=rr, atr_n=int(atr_n), h_start=h_start, h_end=h_end,
         contract=contract, point=point, default_spread=default_spread, min_lot=min_lot)

if st.button("Lancer le backtest", type="primary"):
    with st.spinner("Simulation en cours…"):
        trades = run_backtest(df, p)
    if trades.empty:
        st.warning("Aucun trade généré avec ces paramètres.")
        st.stop()
    cut = df.time.iloc[int(len(df) * split)]
    parts = {"Total": trades, "Entraînement": trades[trades["entrée"] < cut], "Test": trades[trades["entrée"] >= cut]}

    st.info(f"Seuil d'équilibre (avant spread) : **{100 / (1 + rr):.1f} %** de réussite avec un ratio de {rr}.")
    tabs = st.tabs(list(parts) + ["Trades"])
    for tab, (name, t) in zip(tabs, parts.items()):
        with tab:
            m = metrics(t, capital)
            if m["Trades"] == 0:
                st.write("Aucun trade sur cette période.")
                continue
            cols = st.columns(4)
            vals = [("Trades", f"{m['Trades']}"), ("Win rate", f"{m['Win rate %']:.1f} %"),
                    ("Espérance", f"{m['Espérance (R)']:.2f} R"), ("Profit factor", f"{m['Profit factor']:.2f}"),
                    ("Drawdown max", f"{m['Drawdown max %']:.1f} %"), ("Drawdown $", f"{m['Drawdown max $']:.2f}"),
                    ("Profit net", f"{m['Profit net $']:.2f} $"), ("Capital final", f"{t['capital'].iloc[-1]:.2f} $")]
            for i, (k, v) in enumerate(vals):
                cols[i % 4].metric(k, v)
            if m["Trades"] < 100:
                st.warning("Moins de 100 trades : ces pourcentages sont peu fiables statistiquement.")
            fig = go.Figure(go.Scatter(x=t["sortie"], y=t["capital"], mode="lines"))
            fig.update_layout(title="Courbe de capital (trades clôturés)", height=350, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with tabs[-1]:
        st.dataframe(trades.round(4), use_container_width=True)
        st.download_button("Télécharger les trades (CSV)", trades.to_csv(index=False).encode(), "trades.csv")

    with st.expander("Limites à connaître"):
        st.markdown("""
- Une seule stratégie ; le split 70/30 ne remplace pas un walk-forward (aucun paramètre n'est optimisé ici).
- Peu de trades = résultats peu fiables.
- Arrondi des lots : le risque réel dépasse souvent 1 % (colonne « risque réel % »), alors que **R** utilise le risque théorique.
- Pas de slippage ni de gaps : sortie au SL/TP exact → résultats optimistes.
- Spread : vérifiez la valeur d'un point ; si l'export contient 0, le spread par défaut s'applique.
- L'heure du CSV est l'heure serveur du courtier, pas GMT.
- Drawdown calculé sur les trades clôturés uniquement.
- Les résultats décrivent le passé, pas une probabilité de gains futurs.
""")
