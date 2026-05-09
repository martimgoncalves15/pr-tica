"""
Smart Trader - Agente Q-Learning com Personalidades
=====================================================
Define o agente base (Q-Learning) e três subclasses que implementam
funções de recompensa distintas, dando ao agente uma "personalidade"
de trading diferente.

Personalidades
--------------
SafeAgent      – Averso ao risco: penaliza perdas fortemente.
BoldAgent      – Arrojado: recompensa lucros altos mas tolera perdas.
DayTraderAgent – Impaciente: penaliza manter posições abertas por tempo.
"""

import numpy as np
from collections import defaultdict
from environment import MarketEnvironment


# ---------------------------------------------------------------------------
# Agente Base: Q-Learning
# ---------------------------------------------------------------------------

class QLearningTrader:
    """
    Agente de Q-Learning para trading de recursos.

    Implementa o algoritmo off-policy Q-Learning (Watkins, 1989):

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

    A estratégia de exploração é ε-greedy com decaimento linear de ε ao
    longo dos episódios, transitando de exploração pura para exploração
    maioritariamente gananciosa (greedy).

    Parâmetros
    ----------
    env            : MarketEnvironment
    alpha          : float – taxa de aprendizagem (0 < α ≤ 1)
    gamma          : float – fator de desconto (0 ≤ γ ≤ 1)
    epsilon_start  : float – ε inicial (exploração máxima)
    epsilon_end    : float – ε final (exploração mínima)
    epsilon_decay  : float – decaimento linear de ε por episódio
    name           : str   – identificador da personalidade
    """

    def __init__(self,
                 env: MarketEnvironment,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float   = 0.05,
                 epsilon_decay: float = 0.001,
                 name: str = "Base"):

        self.env           = env
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.name          = name

        # Q-Table: mapeamento defaultdict  estado → array de valores por ação
        self.Q: dict = defaultdict(lambda: np.zeros(env.n_actions))

        # Histórico de treino
        self.episode_rewards: list[float] = []
        self.portfolio_history: list[list[float]] = []

    # ------------------------------------------------------------------
    # Política ε-greedy
    # ------------------------------------------------------------------

    def choose_action(self, state: tuple) -> int:
        """
        Seleciona uma ação com política ε-greedy.

        Com probabilidade ε escolhe aleatoriamente (exploração).
        Com probabilidade 1−ε escolhe a ação de maior valor Q (exploração).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Transformação da Recompensa (sobrescrita pelas personalidades)
    # ------------------------------------------------------------------

    def shape_reward(self,
                     raw_reward: float,
                     action: int,
                     holding_days: int) -> float:
        """
        Transforma a recompensa bruta do ambiente.

        Na classe base, a recompensa não é modificada.
        As subclasses sobrescrevem este método para implementar
        as diferentes personalidades de risco.

        Parâmetros
        ----------
        raw_reward   : float – recompensa devolvida pelo ambiente
        action       : int   – ação tomada (0=HOLD, 1=BUY, 2=SELL)
        holding_days : int   – número de dias com posição aberta

        Retorna
        -------
        float – recompensa modificada
        """
        return raw_reward

    # ------------------------------------------------------------------
    # Ciclo de Treino
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 800, verbose: bool = True) -> None:
        """
        Executa n_episodes episódios de treino com Q-Learning.

        Em cada episódio o agente percorre toda a série temporal de preços
        desde o início, atualizando a Q-Table a cada transição.

        Parâmetros
        ----------
        n_episodes : int  – número de episódios de treino
        verbose    : bool – se True imprime progresso a cada 100 episódios
        """
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done           = False
            total_reward   = 0.0
            holding_days   = 0

            while not done:
                action = self.choose_action(state)

                # Tracking de dias com posição aberta
                if self.env.is_holding:
                    holding_days += 1
                else:
                    holding_days = 0

                next_state, raw_reward, done, _, _ = self.env.step(action)

                # Recompensa moldada pela personalidade
                reward = self.shape_reward(raw_reward, action, holding_days)
                total_reward += reward

                # Atualização da Q-Table  (equação de Bellman)
                best_next = np.max(self.Q[next_state])
                td_target = reward + self.gamma * best_next * (not done)
                td_error  = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                state = next_state

            # Decaimento de ε
            self.epsilon = max(self.epsilon_end,
                               self.epsilon - self.epsilon_decay)

            self.episode_rewards.append(total_reward)
            self.portfolio_history.append(self.env.portfolio_values[:])

            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(self.episode_rewards[-100:])
                print(f"[{self.name}] Episódio {episode+1:>4d}/{n_episodes}"
                      f"  |  Recompensa média (últimos 100): {avg:+.2f}"
                      f"  |  ε = {self.epsilon:.3f}")

    # ------------------------------------------------------------------
    # Avaliação (sem exploração)
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Corre um episódio de avaliação com política puramente gananciosa (ε=0).

        Retorna um dicionário com métricas de performance:
          - portfolio_values : evolução do valor de carteira
          - total_return     : retorno total em €
          - return_pct       : retorno percentual
          - n_trades         : número de operações de compra/venda
        """
        saved_eps   = self.epsilon
        self.epsilon = 0.0          # sem exploração na avaliação

        state, _     = self.env.reset()
        done         = False
        n_trades     = 0
        actions_log  = []

        while not done:
            action = self.choose_action(state)
            actions_log.append(action)
            if action in (self.env.BUY, self.env.SELL):
                n_trades += 1
            state, _, done, _, _ = self.env.step(action)

        self.epsilon = saved_eps

        final_val   = self.env.portfolio_values[-1]
        initial_val = self.env.portfolio_values[0]
        total_return = final_val - initial_val

        return {
            "portfolio_values": self.env.portfolio_values,
            "total_return":     total_return,
            "return_pct":       total_return / initial_val * 100,
            "n_trades":         n_trades,
            "actions_log":      actions_log,
        }

    # ------------------------------------------------------------------
    # Inspeção da Q-Table
    # ------------------------------------------------------------------

    def print_q_table(self) -> None:
        """Imprime a Q-Table aprendida de forma legível."""
        actions = ["HOLD", "BUY ", "SELL"]
        print(f"\n{'─'*60}")
        print(f"  Q-Table – Agente: {self.name}")
        print(f"{'─'*60}")
        print(f"  {'Estado':<25}  {'HOLD':>8}  {'BUY':>8}  {'SELL':>8}")
        print(f"{'─'*60}")
        for state, values in sorted(self.Q.items()):
            best = np.argmax(values)
            row  = f"  {str(state):<25}  "
            row += "  ".join(
                f"{'▶' if i == best else ' '}{v:+7.3f}"
                for i, v in enumerate(values)
            )
            print(row)
        print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Personalidade 1: Agente Seguro (Averso ao Risco)
# ---------------------------------------------------------------------------

class SafeAgent(QLearningTrader):
    """
    Agente conservador que penaliza perdas com intensidade dupla.

    Estratégia de recompensa:
      - Lucro  (+) → recompensa normal
      - Perda  (−) → penalização amplificada (×2.5)
      - Lógica: o agente aprende a sair rapidamente de posições perdedoras
                e a preferir pequenos ganhos seguros a apostas arriscadas.
    """

    def __init__(self, env: MarketEnvironment, **kwargs):
        super().__init__(env, name="Seguro", **kwargs)

    def shape_reward(self, raw_reward: float, action: int,
                     holding_days: int) -> float:
        if raw_reward < 0:
            return raw_reward * 2.5     # penalização ampliada para perdas
        return raw_reward


# ---------------------------------------------------------------------------
# Personalidade 2: Agente Arrojado (Tolerante ao Risco)
# ---------------------------------------------------------------------------

class BoldAgent(QLearningTrader):
    """
    Agente agressivo que recompensa lucros elevados com bónus exponencial.

    Estratégia de recompensa:
      - Lucro  (+) → recompensa amplificada (×1.5), com bónus extra para lucros altos
      - Perda  (−) → penalização suavizada (×0.5), encoraja a aguentar a descida
      - Lógica: o agente aprende a "HODL" (manter posição) à espera de ganhos maiores.
    """

    def __init__(self, env: MarketEnvironment, **kwargs):
        super().__init__(env, name="Arrojado", **kwargs)

    def shape_reward(self, raw_reward: float, action: int,
                     holding_days: int) -> float:
        if raw_reward > 0:
            bonus = 0.5 if raw_reward > 10 else 0   # bónus para lucros grandes
            return raw_reward * 1.5 + bonus
        return raw_reward * 0.5                       # penalização suavizada


# ---------------------------------------------------------------------------
# Personalidade 3: Day Trader (Impaciente)
# ---------------------------------------------------------------------------

class DayTraderAgent(QLearningTrader):
    """
    Agente impaciente que penaliza manter posições abertas por muitos dias.

    Estratégia de recompensa:
      - Cada dia com posição aberta → penalização de -0.3 × dias
      - Lucro realizado (SELL)      → recompensa normal
      - Lógica: o agente aprende a fazer operações curtas e frequentes,
                nunca deixando uma posição aberta por demasiado tempo.
    """

    def __init__(self, env: MarketEnvironment, **kwargs):
        super().__init__(env, name="Day Trader", **kwargs)

    def shape_reward(self, raw_reward: float, action: int,
                     holding_days: int) -> float:
        holding_penalty = -0.3 * holding_days if holding_days > 0 else 0
        return raw_reward + holding_penalty
