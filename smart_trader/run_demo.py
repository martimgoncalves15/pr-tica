"""
Smart Trader – Demonstração Visual Animada
==========================================
Corre este script para ver os agentes a atuar em tempo real.

Uso:
    python run_demo.py              # mostra menu para escolher agente
    python run_demo.py --all        # anima os 3 agentes um a seguir ao outro
    python run_demo.py --speed 5    # velocidade da animação (1=lento, 10=rápido)

Requisitos:
    pip install matplotlib numpy
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from collections import defaultdict

# Adicionar pasta ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import MarketEnvironment, generate_price_series
from agent import SafeAgent, BoldAgent, DayTraderAgent


# ──────────────────────────────────────────────────────────────────────────────
# Configuração visual
# ──────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "Seguro":     "#2196F3",
    "Arrojado":   "#E53935",
    "Day Trader": "#2E7D32",
}

ACTION_NAMES = {0: "HOLD ⏸", 1: "BUY  ▲", 2: "SELL ▼"}
ACTION_COLORS_MAP = {0: "#9E9E9E", 1: "#4CAF50", 2: "#F44336"}


# ──────────────────────────────────────────────────────────────────────────────
# Treino (rápido, sem verbose)
# ──────────────────────────────────────────────────────────────────────────────

def train_agents(prices, n_episodes=800):
    """Treina os 3 agentes e devolve-os prontos a avaliar."""
    window = 10
    kwargs = dict(alpha=0.15, gamma=0.95, epsilon_start=1.0,
                  epsilon_end=0.05, epsilon_decay=(1.0 - 0.05) / n_episodes)

    agents = []
    classes = [SafeAgent, BoldAgent, DayTraderAgent]
    for AgentClass in classes:
        env = MarketEnvironment(prices, window=window)
        ag = AgentClass(env, **kwargs)
        print(f"  A treinar: {ag.name}...", end="", flush=True)
        ag.train(n_episodes=n_episodes, verbose=False)
        print(" ✓")
        agents.append(ag)
    return agents


# ──────────────────────────────────────────────────────────────────────────────
# Recolha de dados de avaliação (sem animação — só dados)
# ──────────────────────────────────────────────────────────────────────────────

def collect_eval_data(agent, prices, window=10):
    """
    Corre um episódio de avaliação e recolhe, passo a passo:
      - preço atual
      - ação tomada
      - valor do portfolio
      - estado discreto atual
      - Q-values para o estado atual
    """
    saved_eps    = agent.epsilon
    agent.epsilon = 0.0

    env   = MarketEnvironment(prices, window=window)
    state, _ = env.reset()
    done  = False

    steps = []
    while not done:
        action    = agent.choose_action(state)
        q_vals    = agent.Q[state].copy()
        price     = env.current_price
        next_state, _, done, _, info = env.step(action)

        steps.append({
            "price":     price,
            "action":    action,
            "portfolio": info["portfolio"],
            "state":     state,
            "q_vals":    q_vals,
        })
        state = next_state

    agent.epsilon = saved_eps
    return steps, env.portfolio_values


# ──────────────────────────────────────────────────────────────────────────────
# Animação principal
# ──────────────────────────────────────────────────────────────────────────────

def animate_agent(agent, prices, speed=3, window=10):
    """
    Anima o agente a operar no mercado com 3 painéis:
      1. Gráfico de preços  com marcas BUY/SELL em tempo real
      2. Evolução do portfolio vs Buy&Hold
      3. Q-Values do estado atual (barras que mudam a cada passo)
    """
    color = AGENT_COLORS.get(agent.name, "#607D8B")
    print(f"\n  A preparar animação para: [{agent.name}]...")
    steps, port_vals = collect_eval_data(agent, prices, window)

    n = len(steps)
    price_series = [s["price"]    for s in steps]
    port_series  = [s["portfolio"] for s in steps]
    actions      = [s["action"]   for s in steps]

    # Buy & Hold baseline
    initial     = 1000.0
    buy_hold    = initial / price_series[0] * np.array(price_series)

    # ── Layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#1A1A2E")
    fig.suptitle(f"Smart Trader  ·  Agente: {agent.name}",
                 fontsize=16, fontweight="bold", color=color, y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.96, top=0.92, bottom=0.08)

    ax_price = fig.add_subplot(gs[0, :])   # topo: preços (largura toda)
    ax_port  = fig.add_subplot(gs[1, 0])   # baixo esq: portfolio
    ax_q     = fig.add_subplot(gs[1, 1])   # baixo dir: Q-values

    for ax in [ax_price, ax_port, ax_q]:
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── Painel 1: Preços ─────────────────────────────────────────────────
    ax_price.set_title("Preço de Mercado  |  ▲ BUY   ▼ SELL",
                        color="#CCCCCC", fontsize=11, pad=6)
    ax_price.set_xlim(0, n)
    ax_price.set_ylim(min(price_series) * 0.96, max(price_series) * 1.04)
    ax_price.set_ylabel("Preço (€)", color="#CCCCCC")

    line_price, = ax_price.plot([], [], color="#78909C", lw=1.2, alpha=0.9)
    scatter_buy  = ax_price.scatter([], [], marker="^", color="#4CAF50",
                                    s=70, zorder=6, label="BUY")
    scatter_sell = ax_price.scatter([], [], marker="v", color="#F44336",
                                    s=70, zorder=6, label="SELL")
    day_line     = ax_price.axvline(0, color=color, lw=1, alpha=0.4,
                                    linestyle="--")
    legend = ax_price.legend(facecolor="#1A1A2E", labelcolor="white",
                              fontsize=9, loc="upper right")

    # Texto de estado no canto
    state_text = ax_price.text(0.01, 0.95, "", transform=ax_price.transAxes,
                                color="#DDDDDD", fontsize=9, va="top",
                                fontfamily="monospace",
                                bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="#0F3460", alpha=0.8))

    # ── Painel 2: Portfolio ───────────────────────────────────────────────
    ax_port.set_title("Valor do Portfolio", color="#CCCCCC", fontsize=10, pad=6)
    ax_port.set_xlim(0, n)
    ymin = min(min(port_series), min(buy_hold)) * 0.97
    ymax = max(max(port_series), max(buy_hold)) * 1.03
    ax_port.set_ylim(ymin, ymax)
    ax_port.set_xlabel("Dia", color="#CCCCCC")
    ax_port.set_ylabel("€", color="#CCCCCC")
    ax_port.axhline(initial, color="#555577", lw=0.8, linestyle=":")

    line_port,   = ax_port.plot([], [], color=color, lw=2,   label=agent.name)
    line_bh,     = ax_port.plot([], [], color="#607D8B", lw=1.2,
                                 linestyle=":", alpha=0.7, label="Buy&Hold")
    port_text    = ax_port.text(0.05, 0.92, "", transform=ax_port.transAxes,
                                 color=color, fontsize=10, fontweight="bold",
                                 va="top")
    ax_port.legend(facecolor="#1A1A2E", labelcolor="white",
                   fontsize=8, loc="lower right")

    # ── Painel 3: Q-Values ────────────────────────────────────────────────
    ax_q.set_title("Q-Values  (estado atual)", color="#CCCCCC",
                   fontsize=10, pad=6)
    bar_labels = ["HOLD", "BUY", "SELL"]
    bar_colors = ["#78909C", "#4CAF50", "#F44336"]
    bars_q = ax_q.bar(bar_labels, [0, 0, 0], color=bar_colors,
                       edgecolor="#1A1A2E", linewidth=1.5, width=0.5)
    ax_q.set_ylabel("Valor Q", color="#CCCCCC")
    ax_q.axhline(0, color="#444466", lw=0.8)

    q_val_texts = [
        ax_q.text(bar.get_x() + bar.get_width() / 2, 0, "",
                  ha="center", va="bottom", color="white",
                  fontsize=9, fontweight="bold")
        for bar in bars_q
    ]
    best_marker = ax_q.text(0.5, 0.92, "", transform=ax_q.transAxes,
                             ha="center", color="#FFD700", fontsize=10,
                             fontweight="bold")

    # ── Acumuladores para BUY/SELL ────────────────────────────────────────
    buy_xs, buy_ys   = [], []
    sell_xs, sell_ys = [], []

    # ── Função de actualização ────────────────────────────────────────────
    INTERVAL = max(10, 120 // speed)   # ms entre frames

    def init():
        line_price.set_data([], [])
        scatter_buy.set_offsets(np.empty((0, 2)))
        scatter_sell.set_offsets(np.empty((0, 2)))
        line_port.set_data([], [])
        line_bh.set_data([], [])
        return []

    def update(frame):
        i = frame

        # Painel 1: preço
        xs = list(range(i + 1))
        line_price.set_data(xs, price_series[:i + 1])
        day_line.set_xdata([i, i])

        act = actions[i]
        if act == 1:   # BUY
            buy_xs.append(i); buy_ys.append(price_series[i])
        elif act == 2: # SELL
            sell_xs.append(i); sell_ys.append(price_series[i])

        if buy_xs:
            scatter_buy.set_offsets(np.c_[buy_xs, buy_ys])
        if sell_xs:
            scatter_sell.set_offsets(np.c_[sell_xs, sell_ys])

        # Estado no texto
        st = steps[i]["state"]
        pc_lbl = {-1:"↓QUEDA", 0:"→ESTÁVEL", 1:"↑SUBIDA"}[st[0]]
        ma_lbl = {-1:"↓abaixo MA", 0:"≈MA", 1:"↑acima MA"}[st[1]]
        ps_lbl = "TEM AÇÃO" if st[2] else "SEM AÇÃO"
        pnl_lbl = {-1:"❌EM PERDA", 0:"➖NEUTRO", 1:"✅EM LUCRO"}[st[3]]
        state_text.set_text(
            f"Dia {i:>3d}  |  {pc_lbl}  |  {ma_lbl}  |  {ps_lbl}  |  {pnl_lbl}\n"
            f"Ação: {ACTION_NAMES[act]}"
        )
        state_text.set_color(ACTION_COLORS_MAP[act])

        # Painel 2: portfolio
        line_port.set_data(xs, port_series[:i + 1])
        line_bh.set_data(xs, buy_hold[:i + 1])
        ret_pct = (port_series[i] - initial) / initial * 100
        port_text.set_text(f"{port_series[i]:.2f} € ({ret_pct:+.1f}%)")

        # Painel 3: Q-values
        qv = steps[i]["q_vals"]
        qmin, qmax = qv.min() - 1, qv.max() + 1
        ax_q.set_ylim(qmin, qmax)
        best_a = int(np.argmax(qv))
        for j, (bar, txt) in enumerate(zip(bars_q, q_val_texts)):
            bar.set_height(qv[j])
            bar.set_alpha(1.0 if j == best_a else 0.5)
            y_pos = qv[j] + (qmax - qmin) * 0.02
            txt.set_position((bar.get_x() + bar.get_width() / 2, y_pos))
            txt.set_text(f"{qv[j]:+.2f}")

        best_marker.set_text(f"▶ Melhor: {bar_labels[best_a]}")

        return [line_price, scatter_buy, scatter_sell, line_port,
                line_bh, day_line, state_text, port_text, best_marker,
                *bars_q, *q_val_texts]

    ani = animation.FuncAnimation(
        fig, update, frames=n,
        init_func=init, interval=INTERVAL,
        blit=False, repeat=False
    )

    plt.show()
    return ani


# ──────────────────────────────────────────────────────────────────────────────
# Menu interativo
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart Trader – Demo Visual")
    parser.add_argument("--all",   action="store_true",
                        help="Anima os 3 agentes um a seguir ao outro")
    parser.add_argument("--speed", type=int, default=3,
                        help="Velocidade da animação (1=lento, 10=rápido)")
    parser.add_argument("--episodes", type=int, default=800,
                        help="Nº de episódios de treino")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  🤖  Smart Trader – Demonstração Visual")
    print("═" * 55)

    # Gerar mercado
    prices = generate_price_series(n_steps=500, seed=42)
    print(f"\n  Mercado gerado: {len(prices)} dias  "
          f"| Preço: {prices.min():.0f}€ – {prices.max():.0f}€\n")

    # Treino
    print("  A treinar agentes (aguarda um momento)...")
    agents = train_agents(prices, n_episodes=args.episodes)
    print("\n  ✅ Treino concluído!\n")

    # Escolha do agente
    if args.all:
        for ag in agents:
            animate_agent(ag, prices, speed=args.speed)
    else:
        print("  Escolhe o agente a visualizar:\n")
        for i, ag in enumerate(agents):
            print(f"    [{i + 1}]  {ag.name}")
        print("    [4]  Todos (sequencial)\n")

        try:
            choice = int(input("  → Opção: ").strip())
        except (ValueError, KeyboardInterrupt):
            print("  Cancelado.")
            return

        if choice == 4:
            for ag in agents:
                animate_agent(ag, prices, speed=args.speed)
        elif 1 <= choice <= 3:
            animate_agent(agents[choice - 1], prices, speed=args.speed)
        else:
            print("  Opção inválida.")


if __name__ == "__main__":
    main()
