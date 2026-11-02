"""
modelos.py
----------
Implementação (apenas com numpy) dos modelos de Redes Neurais Artificiais
pedidos no trabalho: Perceptron Simples, ADALINE e Perceptron de Múltiplas
Camadas (MLP).

Convenções adotadas (formulação dos slides da disciplina):
  * X tem dimensão (p+1) x N. Cada COLUNA é uma amostra e a PRIMEIRA linha é
    a entrada de bias, sempre igual a -1 (o peso associado a ela faz o papel
    do limiar de ativação θ).
  * Y tem dimensão C x N com rótulos bipolares (+1/-1). No problema binário
    C = 1; no reconhecimento facial C = 20 (one-hot bipolar).
  * Todos os modelos têm critério de parada por número máximo de épocas.
  * O Erro Quadrático Médio (EQM) de cada época é guardado para montar a
    curva de aprendizado:
            EQM = 1/(2N) * soma_t soma_j ( d_j(t) - y_j(t) )^2
    Se um conjunto de teste for passado ao fit(), o EQM dele também é
    registrado, mas SOMENTE para o gráfico: ele nunca participa do ajuste dos
    pesos nem da decisão de parada.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------
def degrau_bipolar(u):
    """Função de ativação degrau bipolar: +1 se u >= 0, senão -1."""
    return np.where(u >= 0, 1.0, -1.0)


def calcular_eqm(D, Y_saida):
    """EQM = 1/(2N) * soma dos erros quadráticos de todas as saídas e amostras."""
    N = D.shape[1]
    return float(np.sum((D - Y_saida) ** 2) / (2.0 * N))


def pesos_aleatorios(n_linhas, n_entradas, rng):
    """Pesos iniciais uniformes em [-a, a], com a = 1/sqrt(n_entradas).

    ADAPTAÇÃO (justificada): os slides sugerem valores aleatórios pequenos,
    tipicamente em [-0.5, 0.5]. Aqui o limite é dividido pela raiz do número
    de entradas do neurônio (incluindo o bias). Para a espiral (3 entradas)
    isso dá a = 0.577, praticamente a faixa clássica. Para as imagens
    (901 entradas) dá a = 0.033, o que impede que o potencial de ativação
    comece muito grande e sature a tangente hiperbólica logo no início
    (regra proposta por LeCun et al., "Efficient BackProp", 1998).
    """
    limite = 1.0 / np.sqrt(n_entradas)
    return rng.uniform(-limite, limite, size=(n_linhas, n_entradas))


def _como_matriz_rotulos(Y):
    """Aceita rótulos como vetor (N,) ou matriz (C, N) e devolve sempre (C, N)."""
    Y = np.asarray(Y, dtype=float)
    return Y.reshape(1, -1) if Y.ndim == 1 else Y


def _decidir(saida):
    """Transforma a saída contínua da rede em decisão.

    * 1 neurônio de saída (binário): degrau bipolar -> +1 ou -1.
    * C neurônios (multiclasse): índice do neurônio com maior saída (argmax).
    """
    if saida.shape[0] == 1:
        return degrau_bipolar(saida[0])
    return np.argmax(saida, axis=0)


# -----------------------------------------------------------------------------
# Perceptron Simples
# -----------------------------------------------------------------------------
class PerceptronSimples:
    """Perceptron Simples (Rosenblatt) com regra de aprendizado clássica.

        u(t) = w^T x(t)
        y(t) = degrau_bipolar(u(t))
        e(t) = d(t) - y(t)
        w    = w + η e(t) x(t)

    O treino é feito amostra a amostra, como nos slides, porque cada correção
    depende dos pesos deixados pela amostra anterior (não é possível
    vetorizar a época inteira sem mudar o algoritmo).

    ADAPTAÇÃO exigida pelo enunciado: o Perceptron original só para quando
    uma época inteira passa sem erros, o que nunca acontece em dados não
    linearmente separáveis (caso da espiral). Por isso o laço também é
    interrompido ao atingir `max_epocas`.

    No caso multiclasse (C saídas), cada linha de W^T é um perceptron
    independente que aprende a separar "sua" pessoa das demais.
    """

    nome = "Perceptron Simples"

    def __init__(self, taxa_aprendizado=0.01, max_epocas=100, semente=None):
        self.taxa_aprendizado = taxa_aprendizado
        self.max_epocas = max_epocas
        self.semente = semente  # não é usada: os pesos começam em zero (slides)

    def fit(self, X, Y, X_teste=None, Y_teste=None):
        Y = _como_matriz_rotulos(Y)
        n_entradas, N = X.shape
        n_saidas = Y.shape[0]
        # Pesos iniciais nulos, como no algoritmo dos slides. Com w(0) = 0 a
        # fronteira obtida não depende do valor de η (todos os pesos ficam
        # multiplicados pelo mesmo fator), então η só altera a escala de w.
        self.W = np.zeros((n_entradas, n_saidas))
        self.eqm_treino, self.eqm_teste = [], []

        amostras = X.T.copy()   # linha t = amostra x(t) (acesso contíguo na memória)
        desejados = Y.T.copy()  # linha t = d(t)
        eta = self.taxa_aprendizado

        for epoca in range(self.max_epocas):
            houve_erro = False
            for t in range(N):
                x_t = amostras[t]
                y_t = degrau_bipolar(x_t @ self.W)
                e_t = desejados[t] - y_t
                if e_t.any():  # só corrige quando a amostra foi classificada errada
                    self.W += eta * np.outer(x_t, e_t)
                    houve_erro = True
            self._registrar(X, Y, X_teste, Y_teste)
            if not houve_erro:  # convergiu: época inteira sem erros
                break
        self.epocas_treinadas = epoca + 1
        return self

    def saida(self, X):
        """Potencial de ativação u = W^T X (C x N)."""
        return self.W.T @ X

    def predict(self, X):
        """Binário: degrau bipolar de u. Multiclasse: argmax de u.

        No multiclasse vários perceptrons podem disparar +1 ao mesmo tempo (ou
        nenhum). Para desempatar usa-se o maior potencial u, que é a saída
        do neurônio antes do degrau.
        """
        return _decidir(self.saida(X))

    def _registrar(self, X, Y, X_teste, Y_teste):
        # O EQM do Perceptron é calculado sobre a saída do degrau.
        self.eqm_treino.append(calcular_eqm(Y, degrau_bipolar(self.saida(X))))
        if X_teste is not None:
            Yt = _como_matriz_rotulos(Y_teste)
            self.eqm_teste.append(calcular_eqm(Yt, degrau_bipolar(self.saida(X_teste))))


# -----------------------------------------------------------------------------
# ADALINE
# -----------------------------------------------------------------------------
class Adaline:
    """ADAptive LINear Element treinado pela regra delta (LMS) sobre o EQM.

        u(t) = w^T x(t)                 (saída linear usada no treino)
        e(t) = d(t) - u(t)
        w    = w + η e(t) x(t)          (passo de gradiente do EQM)

    Parada (slides): |EQM_atual - EQM_anterior| <= precisão, ou ao atingir
    `max_epocas`. Depois de treinado, a classificação usa o degrau bipolar
    sobre u (binário) ou o argmax de u (multiclasse).
    """

    nome = "ADALINE"

    def __init__(self, taxa_aprendizado=0.001, max_epocas=200, precisao=1e-6, semente=None):
        self.taxa_aprendizado = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.semente = semente

    def fit(self, X, Y, X_teste=None, Y_teste=None):
        rng = np.random.default_rng(self.semente)
        Y = _como_matriz_rotulos(Y)
        n_entradas, N = X.shape
        n_saidas = Y.shape[0]
        self.W = pesos_aleatorios(n_saidas, n_entradas, rng).T  # (p+1) x C
        self.eqm_treino, self.eqm_teste = [], []

        amostras = X.T.copy()
        desejados = Y.T.copy()
        eta = self.taxa_aprendizado
        eqm_anterior = calcular_eqm(Y, self.saida(X))

        for epoca in range(self.max_epocas):
            for t in range(N):
                x_t = amostras[t]
                e_t = desejados[t] - x_t @ self.W
                self.W += eta * np.outer(x_t, e_t)
            eqm_atual = self._registrar(X, Y, X_teste, Y_teste)
            if not np.isfinite(eqm_atual):  # proteção caso η seja grande demais
                break
            if abs(eqm_atual - eqm_anterior) <= self.precisao:
                break
            eqm_anterior = eqm_atual
        self.epocas_treinadas = epoca + 1
        return self

    def saida(self, X):
        """Saída linear u = W^T X (C x N)."""
        return self.W.T @ X

    def predict(self, X):
        return _decidir(self.saida(X))

    def _registrar(self, X, Y, X_teste, Y_teste):
        # O EQM do ADALINE é o da saída linear: é exatamente a função de custo
        # que a regra LMS minimiza.
        eqm = calcular_eqm(Y, self.saida(X))
        self.eqm_treino.append(eqm)
        if X_teste is not None:
            Yt = _como_matriz_rotulos(Y_teste)
            self.eqm_teste.append(calcular_eqm(Yt, self.saida(X_teste)))
        return eqm


# -----------------------------------------------------------------------------
# Perceptron de Múltiplas Camadas (MLP)
# -----------------------------------------------------------------------------
class MLP:
    """MLP com tangente hiperbólica em todos os neurônios, treinada por
    backpropagation e gradiente descendente.

    Notação (slides): W[l] é a matriz de pesos da camada l, com dimensão
    q_l x (q_{l-1} + 1); a primeira coluna corresponde ao bias (-1).

    Fase forward, para cada camada l:
        i[l] = W[l] y[l-1]          (y[0] = X, já com a linha de -1)
        y[l] = tanh(i[l])           (e recebe a linha -1 para a camada seguinte)

    Fase backward (g'(i) = 1 - tanh(i)^2 = 1 - y^2):
        δ[L] = g'(i[L]) ⊙ (D - y[L])                       (camada de saída)
        δ[l] = g'(i[l]) ⊙ (W[l+1] sem a coluna do bias)^T δ[l+1]
        W[l] = W[l] + (η / N) δ[l] y[l-1]^T

    ADAPTAÇÃO (justificada): os slides apresentam o backpropagation amostra a
    amostra. Aqui os gradientes das N amostras são calculados de uma vez com
    operações matriciais (gradiente descendente em lote, "batch"). A regra de
    atualização é a mesma, apenas somada sobre as amostras e dividida por N,
    e corresponde ao gradiente exato do EQM do conjunto de treino. Isso
    torna viável executar centenas de rodadas (R = 500) da validação.

    Parada: EQM de treino <= precisão, ou ao atingir `max_epocas`.
    """

    nome = "MLP"

    def __init__(self, camadas_ocultas=(10,), taxa_aprendizado=0.1, max_epocas=1000,
                 precisao=1e-4, semente=None):
        self.camadas_ocultas = tuple(camadas_ocultas)
        self.taxa_aprendizado = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.semente = semente

    def descricao(self):
        return "MLP " + "-".join(str(q) for q in self.camadas_ocultas)

    def _inicializar(self, n_entradas, n_saidas, rng):
        """Cria W[l] para cada camada: q_l linhas e (q_{l-1} + 1) colunas."""
        neuronios = [n_entradas - 1] + list(self.camadas_ocultas) + [n_saidas]
        self.W = [pesos_aleatorios(neuronios[l + 1], neuronios[l] + 1, rng)
                  for l in range(len(neuronios) - 1)]

    def _forward(self, X):
        """Propaga X pela rede.

        Retorna as entradas de cada camada (já com a linha -1 do bias) e as
        saídas y[l] de cada camada (sem o bias).
        """
        entradas, saidas = [X], []
        y = X
        for l, W in enumerate(self.W):
            y = np.tanh(W @ y)
            saidas.append(y)
            if l < len(self.W) - 1:
                y = np.vstack((-np.ones((1, y.shape[1])), y))
                entradas.append(y)
        return entradas, saidas

    def fit(self, X, Y, X_teste=None, Y_teste=None):
        rng = np.random.default_rng(self.semente)
        Y = _como_matriz_rotulos(Y)
        Yt = _como_matriz_rotulos(Y_teste) if X_teste is not None else None
        N = X.shape[1]
        self._inicializar(X.shape[0], Y.shape[0], rng)
        self.eqm_treino, self.eqm_teste = [], []
        passo = self.taxa_aprendizado / N

        for epoca in range(self.max_epocas):
            # ---------- forward ----------
            entradas, saidas = self._forward(X)
            erro = Y - saidas[-1]
            eqm = float(np.sum(erro ** 2) / (2.0 * N))
            self.eqm_treino.append(eqm)
            if Yt is not None:
                self.eqm_teste.append(calcular_eqm(Yt, self.saida(X_teste)))
            if eqm <= self.precisao:
                break

            # ---------- backward ----------
            delta = (1.0 - saidas[-1] ** 2) * erro
            for l in range(len(self.W) - 1, -1, -1):
                gradiente = delta @ entradas[l].T
                if l > 0:
                    # δ da camada anterior usa W[l] ANTES da atualização e
                    # descarta a coluna 0 (peso do bias).
                    delta = (1.0 - saidas[l - 1] ** 2) * (self.W[l][:, 1:].T @ delta)
                self.W[l] += passo * gradiente
        self.epocas_treinadas = epoca + 1
        return self

    def saida(self, X):
        """Saída da última camada (valores contínuos em (-1, 1)), C x N."""
        return self._forward(X)[1][-1]

    def predict(self, X):
        return _decidir(self.saida(X))
