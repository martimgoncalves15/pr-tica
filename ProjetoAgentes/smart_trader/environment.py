"""
Smart Trader - Ambiente de Mercado Simulado
============================================
Módulo que define o ambiente (Environment) onde o agente opera.
O ambiente simula um mercado com preços gerados sinteticamente,
encapsulando a lógica de estado, recompensa e transição.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gerador de Preços Sintético
# ---------------------------------------------------------------------------

def generate_price_series(n_steps: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Gera uma série temporal de preços sintética com tendência cíclica e ruído.

    O preço é composto por:
      - Uma componente sinusoidal de longa duração (tendência de mercado)
      - Uma componente sinusoidal de curta duração (volatilidade intermédia)
      - Ruído gaussiano (variação diária aleatória)

    Parâmetros
    ----------
    n_steps : int
        Número de passos temporais (dias de mercado) a gerar.
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    np.ndarray
        Array de preços normalizados entre ~50 e ~150.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_steps)

    trend  = 20 * np.sin(t)
    medium = 10 * np.sin(3 * t + np.pi / 4)
    noise  = rng.normal(0, 3, n_steps)
    prices = 100 + trend + medium + noise

    return np.clip(prices, 10, None)


# ---------------------------------------------------------------------------
# Discretização do Espaço de Estados
# ---------------------------------------------------------------------------

def discretize_state(price_change_pct: float,
                     distance_to_ma_pct: float,
                     holding: bool,
                     unrealized_pnl_pct: float = 0.0) -> tuple:
    """
    Converte variáveis contínuas do mercado num estado discreto para a Q-Table.

    O estado é um tuplo de 4 componentes:
      1. Variação de preço     : -1 (queda), 0 (estável), +1 (subida)
      2. Distância à MA-10     : -1 (abaixo), 0 (próximo), +1 (acima)
      3. Posição atual         :  0 (sem ação),  1 (com ação em carteira)
      4. Lucro/Perda não real. : -1 (em perda), 0 (neutro/sem posição), +1 (em lucro)

    Esta 4ª componente é crítica: permite que o agente distinga entre
    'estou a perder dinheiro nesta posição' e 'estou a ganhar', possibilitando
    estratégias de venda baseadas em P&L não realizado.

    Parâmetros
    ----------
    price_change_pct    : float – variação % do preço face ao dia anterior
    distance_to_ma_pct  : float – diferença % entre preço atual e MA-10
    holding             : bool  – True se o agente possui uma unidade em carteira
    unrealized_pnl_pct  : float – % de lucro/perda não realizado (0 se não holding)

    Retorna
    -------
    tuple – (variacao_preco, dist_ma, posicao, pnl_bin)
    """
    if price_change_pct > 1.0:
        pc_bin = 1
    elif price_change_pct < -1.0:
        pc_bin = -1
    else:
        pc_bin = 0

    if distance_to_ma_pct > 2.0:
        ma_bin = 1
    elif distance_to_ma_pct < -2.0:
        ma_bin = -1
    else:
        ma_bin = 0

    if not holding:
        pnl_bin = 0
    elif unrealized_pnl_pct > 1.5:
        pnl_bin = 1
    elif unrealized_pnl_pct < -1.5:
        pnl_bin = -1
    else:
        pnl_bin = 0

    return (pc_bin, ma_bin, int(holding), pnl_bin)


# ---------------------------------------------------------------------------
# Ambiente de Mercado
# ---------------------------------------------------------------------------

class MarketEnvironment:
    """
    Ambiente de mercado simulado compatível com a interface gymnasium.

    O agente dispõe de 3 ações:
      0 – HOLD  : não fazer nada
      1 – BUY   : comprar 1 unidade (se não tiver em carteira)
      2 – SELL  : vender 1 unidade (se tiver em carteira)

    O estado observável é um tuplo discreto de 4 componentes produzido por
    `discretize_state`. A inclusão do P&L não realizado como componente
    de estado é uma decisão de desenho fundamental: sem esta informação,
    o agente não consegue distinguir 'devo vender porque estou em lucro'
    de 'devo aguentar porque estou em perda'.
    """

    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(self, prices: np.ndarray, window: int = 10):
        self.prices    = prices
        self.window    = window
        self.n_actions = 3
        self._step      = 0
        self._holding   = False
        self._buy_price = 0.0
        self._cash      = 1000.0
        self._portfolio_values: list[float] = []

    def reset(self) -> tuple:
        """Reinicia o ambiente para o início da série temporal."""
        self._step      = self.window
        self._holding   = False
        self._buy_price = 0.0
        self._cash      = 1000.0
        self._portfolio_values = [self._cash]
        return self._observe(), {}

    def step(self, action: int) -> tuple:
        """
        Executa uma ação no mercado e avança um dia.

        Parâmetros
        ----------
        action : int  – 0=HOLD, 1=BUY, 2=SELL

        Retorna
        -------
        (next_state, reward, done, truncated, info)
        """
        price  = self.prices[self._step]
        reward = self._apply_action(action, price)

        portfolio_val = self._cash + (price if self._holding else 0.0)
        self._portfolio_values.append(portfolio_val)

        self._step += 1
        done = self._step >= len(self.prices) - 1

        return self._observe(), reward, done, False, {"portfolio": portfolio_val}

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _observe(self) -> tuple:
        """Calcula e devolve o estado atual discretizado (4 componentes)."""
        if self._step == 0:
            return (0, 0, 0, 0)

        price      = self.prices[self._step]
        prev_price = self.prices[self._step - 1]
        ma         = np.mean(self.prices[self._step - self.window: self._step])

        change_pct = (price - prev_price) / prev_price * 100
        ma_dist    = (price - ma) / ma * 100

        if self._holding and self._buy_price > 0:
            pnl_pct = (price - self._buy_price) / self._buy_price * 100
        else:
            pnl_pct = 0.0

        return discretize_state(change_pct, ma_dist, self._holding, pnl_pct)

    def _apply_action(self, action: int, price: float) -> float:
        """Executa a ação e devolve a recompensa base (lucro bruto na venda)."""
        reward = 0.0

        if action == self.BUY and not self._holding:
            self._holding   = True
            self._buy_price = price
            self._cash     -= price

        elif action == self.SELL and self._holding:
            profit = price - self._buy_price
            reward = profit
            self._cash   += price
            self._holding = False
            self._buy_price = 0.0

        return reward

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def portfolio_values(self) -> list[float]:
        return self._portfolio_values

    @property
    def current_price(self) -> float:
        return self.prices[self._step]

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def n_steps(self) -> int:
        return len(self.prices) - self.window - 1
