"""
Smart Trader - Visualizações
=============================
Funções de plotagem para análise e apresentação dos resultados do projeto.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Paleta de cores por personalidade
# ---------------------------------------------------------------------------

COLORS = {
    "Seguro":     "#2196F3",   # azul
    "Arrojado":   "#F44336",   # vermelho
    "Day Trader": "#4CAF50",   # verde
}

ACTION_COLORS = {0: "#BDBDBD", 1: "#4CAF50", 2: "#F44336"}   # cinza, verde, vermelho
ACTION_LABELS = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ---------------------------------------------------------------------------
# 1. Série de Preços
# ---------------------------------------------------------------------------

def plot_prices(prices: np.ndarray, window: int = 10) -> None:
    """Plota a série temporal de preços com a média móvel."""
    ma = np.array([np.mean(prices[max(0, i - window):i])
                   for i in range(1, len(prices) + 1)])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prices, color="#546E7A", lw=1.2, label="Preço diário", alpha=0.8)
    ax.plot(ma,     color="#FF9800", lw=2,   label=f"MA-{window}", linestyle="--")
    ax.fill_between(range(len(prices)), prices, ma,
                    where=(prices > ma), alpha=0.12, color="#4CAF50", label="Preço > MA")
    ax.fill_between(range(len(prices)), prices, ma,
                    where=(prices < ma), alpha=0.12, color="#F44336", label="Preço < MA")
    ax.set_title("Série de Preços do Mercado Simulado", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Preço (€)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output_prices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Gráfico guardado: output_prices.png")


# ---------------------------------------------------------------------------
# 2. Curva de Aprendizagem
# ---------------------------------------------------------------------------

def plot_learning_curves(agents: list) -> None:
    """Plota a evolução da recompensa média ao longo dos episódios para cada agente."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for agent in agents:
        rewards = np.array(agent.episode_rewards)
        # Média móvel suavizada de 50 episódios
        smoothed = np.convolve(rewards, np.ones(50) / 50, mode="valid")
        x = np.arange(len(smoothed)) + 50
        color = COLORS.get(agent.name, "gray")
        ax.plot(x, smoothed, color=color, lw=2, label=agent.name)
        ax.fill_between(x, smoothed - np.std(rewards[:len(smoothed)]),
                           smoothed + np.std(rewards[:len(smoothed)]),
                        color=color, alpha=0.08)

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title("Curva de Aprendizagem – Recompensa Acumulada por Episódio",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa Total (média móvel 50 ep.)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output_learning.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Gráfico guardado: output_learning.png")


# ---------------------------------------------------------------------------
# 3. Comparação de Portfolio na Avaliação Final
# ---------------------------------------------------------------------------

def plot_portfolio_comparison(prices: np.ndarray,
                               results: dict,
                               window: int = 10) -> None:
    """
    Compara a evolução do portfolio de cada agente durante a avaliação,
    incluindo uma estratégia de referência Buy & Hold.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Painel superior: preços ---
    ax1 = axes[0]
    ax1.plot(prices[window:], color="#546E7A", lw=1, alpha=0.7, label="Preço")
    ax1.set_ylabel("Preço (€)")
    ax1.set_title("Comparação de Performance dos Agentes", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # --- Painel inferior: valores de carteira ---
    ax2 = axes[1]

    # Baseline: Buy & Hold (compra no dia 1, vende no dia final)
    initial = 1000.0
    n = len(prices[window:])
    buy_hold = initial / prices[window] * prices[window:]
    ax2.plot(buy_hold, color="black", lw=1.5, linestyle=":", label="Buy & Hold", alpha=0.6)

    for name, result in results.items():
        vals  = result["portfolio_values"]
        color = COLORS.get(name, "gray")
        ax2.plot(vals, color=color, lw=2, label=f"{name} ({result['return_pct']:+.1f}%)")

    ax2.axhline(initial, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Dias")
    ax2.set_ylabel("Valor do Portfolio (€)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output_portfolio.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  → Gráfico guardado: output_portfolio.png")


# ---------------------------------------------------------------------------
# 4. Decisões do Agente sobre o Preço
# ---------------------------------------------------------------------------

def plot_agent_decisions(prices: np.ndarray,
                          result: dict,
                          agent_name: str,
                          window: int = 10) -> None:
    """
    Plota o preço com marcas de BUY (▲ verde) e SELL (▼ vermelho)
    para um agente específico.
    """
    actions  = result["actions_log"]
    price_slice = prices[window: window + len(actions)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(price_slice, color="#546E7A", lw=1.2, alpha=0.7, label="Preço")

    buys  = [i for i, a in enumerate(actions) if a == 1]
    sells = [i for i, a in enumerate(actions) if a == 2]

    ax.scatter(buys,  price_slice[buys],  marker="^", color="#4CAF50",
               s=80, zorder=5, label="BUY")
    ax.scatter(sells, price_slice[sells], marker="v", color="#F44336",
               s=80, zorder=5, label="SELL")

    color = COLORS.get(agent_name, "gray")
    ax.set_title(f"Decisões do Agente [{agent_name}]  –  "
                 f"Retorno: {result['return_pct']:+.1f}%  |  "
                 f"Operações: {result['n_trades']}",
                 fontsize=13, fontweight="bold", color=color)
    ax.set_xlabel("Dias")
    ax.set_ylabel("Preço (€)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"output_decisions_{agent_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  → Gráfico guardado: {fname}")


# ---------------------------------------------------------------------------
# 5. Tabela Resumo de Métricas
# ---------------------------------------------------------------------------

def print_summary_table(results: dict) -> None:
    """Imprime uma tabela comparativa com as métricas dos 3 agentes."""
    print("\n" + "═" * 62)
    print("  TABELA RESUMO – AVALIAÇÃO FINAL DOS AGENTES")
    print("═" * 62)
    print(f"  {'Agente':<14}  {'Retorno €':>10}  {'Retorno %':>10}  {'N.º Trades':>11}")
    print("─" * 62)
    for name, r in results.items():
        print(f"  {name:<14}  {r['total_return']:>+10.2f}  "
              f"{r['return_pct']:>+9.2f}%  {r['n_trades']:>11d}")
    print("═" * 62 + "\n")
