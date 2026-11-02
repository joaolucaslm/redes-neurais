"""
etapa1.py
---------
Primeira etapa do trabalho: classificação binária do conjunto spiral_d.csv
com Perceptron Simples, ADALINE e MLP.

O script executa, em ordem:
  1. organização dos dados (X com dimensão (p+1) x N, rótulos bipolares) e
     gráfico de espalhamento;
  2. busca de hiperparâmetros em partições treino/validação;
  3. estudo de underfitting/overfitting da MLP (três topologias);
  4. validação por amostragem aleatória (R rodadas, 80% treino / 20% teste);
  5. matrizes de confusão e curvas de aprendizado das rodadas extremas;
  6. tabelas (uma por métrica) e boxplot/violin plot.

Figuras são gravadas em figuras/etapa1/ e tabelas em resultados/etapa1/.

Uso:
    python etapa1.py                # execução completa (R = 500)
    python etapa1.py --R 50         # menos rodadas, para um teste rápido
    python etapa1.py --sem-ajuste   # pula a busca e usa os hiperparâmetros padrão
"""

import argparse
import json
import os
import time

# As matrizes destes modelos são pequenas; nesse caso, o paralelismo interno do
# BLAS (biblioteca usada pelo numpy) gasta mais tempo sincronizando threads do
# que calculando. Uma única thread foi medida como igual ou mais rápida. A
# variável precisa ser definida antes do primeiro "import numpy".
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils as U
from modelos import PerceptronSimples, Adaline, MLP

RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_FIG = os.path.join(RAIZ, "figuras", "etapa1")
PASTA_RES = os.path.join(RAIZ, "resultados", "etapa1")

SEMENTE = 42               # semente fixa para que os resultados sejam reproduzíveis
PROPORCAO_TREINO = 0.8     # 80% treino / 20% teste
METRICAS = ("acuracia", "sensibilidade", "especificidade")
NOME_METRICA = {"acuracia": "Acurácia", "sensibilidade": "Sensibilidade",
                "especificidade": "Especificidade"}
ROTULOS_BINARIOS = ["+1", "-1"]   # ordem da matriz: classe positiva primeiro

# -----------------------------------------------------------------------------
# Hiperparâmetros padrão. São os valores escolhidos pela busca (seção 2) e
# servem como reserva quando o script é executado com --sem-ajuste.
# -----------------------------------------------------------------------------
HP_PERCEPTRON = dict(taxa_aprendizado=0.01, max_epocas=100)
HP_ADALINE = dict(taxa_aprendizado=1e-4, max_epocas=500, precisao=1e-6)
HP_MLP = dict(camadas_ocultas=(5, 5), taxa_aprendizado=2.0, max_epocas=2000, precisao=1e-3)

# Grades exploradas na busca.
GRADE_ETA_PERCEPTRON = (0.001, 0.01, 0.1, 1.0)
GRADE_ETA_ADALINE = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
GRADE_TOPOLOGIAS_MLP = ((5,), (10,), (20,), (5, 5), (10, 10), (20, 20))
GRADE_ETA_MLP = (0.5, 1.0, 2.0)
PARTICOES_BUSCA = 5        # nº de partições treino/validação por configuração
# 0,25 ponto percentual: com 5 x 280 = 1400 amostras de validação, equivale a
# cerca de 3 amostras, ou seja, diferenças desse tamanho são tratadas como empate.
TOLERANCIA_PARCIMONIA = 0.0025

# Topologias do estudo de under/overfitting (a "adequada" vem da busca).
TOPOLOGIA_SUB = dict(camadas_ocultas=(2,), taxa_aprendizado=2.0, max_epocas=2000, precisao=1e-3)
TOPOLOGIA_SUPER = dict(camadas_ocultas=(60, 60, 60), taxa_aprendizado=1.0, max_epocas=15000,
                       precisao=0.0)  # precisão 0: treina todas as épocas, sem parada antecipada
PARTICOES_TOPOLOGIAS = 5   # partições usadas no estudo de under/overfitting


# =============================================================================
# Funções auxiliares
# =============================================================================
def avaliar(modelo, X, y_bipolar):
    """Matriz de confusão (contagem) e métricas de um modelo treinado."""
    reais = U.bipolar_para_indice(y_bipolar)
    previstos = U.bipolar_para_indice(modelo.predict(X))
    matriz = U.matriz_confusao(reais, previstos, 2)
    return matriz, U.metricas_binarias(matriz)


def criar_modelos(hp_perceptron, hp_adaline, hp_mlp, semente):
    """Instancia os três modelos com os hiperparâmetros definidos."""
    return {
        "Perceptron Simples": PerceptronSimples(**hp_perceptron),
        "ADALINE": Adaline(**hp_adaline, semente=semente),
        "MLP": MLP(**hp_mlp, semente=semente + 1),
    }


def descrever_hp(hp):
    """Texto curto com os hiperparâmetros (usado em títulos e no terminal)."""
    partes = []
    if "camadas_ocultas" in hp:
        partes.append("ocultas=" + "-".join(str(q) for q in hp["camadas_ocultas"]))
    partes.append(f"η={hp['taxa_aprendizado']:g}")
    partes.append(f"épocas máx.={hp['max_epocas']}")
    if "precisao" in hp:
        partes.append(f"ε={hp['precisao']:g}")
    return ", ".join(partes)


def numero_de_pesos(camadas_ocultas, p=2, c=1):
    """Total de pesos (incluindo bias) de uma MLP p -> ocultas -> c."""
    neuronios = [p] + list(camadas_ocultas) + [c]
    return sum((neuronios[i] + 1) * neuronios[i + 1] for i in range(len(neuronios) - 1))


# =============================================================================
# 1. Dados e visualização inicial
# =============================================================================
def secao_dados():
    X_bruto, rotulos = U.carregar_espiral(os.path.join(RAIZ, "spiral_d.csv"))
    print(f"  Valores encontrados na coluna de rótulos: {np.unique(rotulos)}")
    y = U.converter_para_bipolar(rotulos)
    n_pos, n_neg = int((y == 1).sum()), int((y == -1).sum())
    print(f"  N = {y.size} amostras, p = {X_bruto.shape[0]} atributos "
          f"| classe +1: {n_pos} | classe -1: {n_neg}")
    # Montagem completa apenas para conferir as dimensões pedidas no enunciado.
    # Nas rodadas, a normalização é refeita usando somente o treino.
    X_exemplo = U.adicionar_bias(U.normalizar(X_bruto, *U.ajustar_normalizacao(X_bruto)))
    print(f"  X organizado com dimensão (p+1) x N = {X_exemplo.shape}; Y com dimensão {y.shape}")
    U.plot_dispersao(X_bruto, y, os.path.join(PASTA_FIG, "01_dispersao.png"),
                     "Conjunto spiral_d: gráfico de espalhamento")
    return X_bruto, y


# =============================================================================
# 2. Busca de hiperparâmetros
# =============================================================================
def particoes_de_busca(N, rng):
    """Partições treino/validação (80/20) usadas somente na busca."""
    return [U.dividir_treino_teste(N, PROPORCAO_TREINO, rng) for _ in range(PARTICOES_BUSCA)]


def escolher_mlp(resultados):
    """Regra de escolha da topologia da MLP (critério de parcimônia).

    Entre as configurações cuja acurácia média de validação está a no máximo
    0,25 ponto percentual da melhor, fica a que tem MENOS pesos e, em caso de
    empate, a de maior acurácia. A ideia é não usar uma rede maior do que o
    necessário, o que reduz o custo e o risco de overfitting.
    """
    melhor = max(r["acuracia"] for r in resultados)
    candidatas = [r for r in resultados if r["acuracia"] >= melhor - TOLERANCIA_PARCIMONIA]
    return min(candidatas, key=lambda r: (r["pesos"], -r["acuracia"]))


def busca_hiperparametros(X_bruto, y):
    rng = np.random.default_rng(SEMENTE + 1000)  # sequência diferente da validação final
    particoes = particoes_de_busca(y.size, rng)
    dados = []
    for itr, ival in particoes:
        Xtr, Xval, _, _ = U.preparar_particao(X_bruto, itr, ival)
        dados.append((Xtr, y[itr], Xval, y[ival]))

    # ---- Perceptron: efeito de η (pesos iniciais nulos) ----
    linhas_p = []
    for eta in GRADE_ETA_PERCEPTRON:
        accs = [avaliar(PerceptronSimples(eta, HP_PERCEPTRON["max_epocas"]).fit(Xtr, ytr),
                        Xval, yval)[1]["acuracia"] for Xtr, ytr, Xval, yval in dados]
        linhas_p.append([eta, np.mean(accs), np.std(accs)])
        print(f"  Perceptron η={eta:<6g} acurácia de validação = {np.mean(accs):.4f}")
    np.savetxt(os.path.join(PASTA_RES, "ajuste_perceptron.csv"), np.array(linhas_p),
               fmt="%.6g", delimiter=";", encoding="utf-8", header="eta;acuracia_media;desvio", comments="")

    # ---- ADALINE: η, com precisão e máximo de épocas fixos ----
    linhas_a, curvas_a = [], {}
    for eta in GRADE_ETA_ADALINE:
        accs, eqms, epocas = [], [], []
        for k, (Xtr, ytr, Xval, yval) in enumerate(dados):
            modelo = Adaline(eta, HP_ADALINE["max_epocas"], HP_ADALINE["precisao"], semente=k)
            modelo.fit(Xtr, ytr, Xval, yval)
            accs.append(avaliar(modelo, Xval, yval)[1]["acuracia"])
            eqms.append(modelo.eqm_teste[-1])
            epocas.append(modelo.epocas_treinadas)
            if k == 0:
                curvas_a[eta] = modelo.eqm_treino
        linhas_a.append([eta, np.mean(accs), np.mean(eqms), np.mean(epocas)])
        print(f"  ADALINE    η={eta:<6g} acurácia val. = {np.mean(accs):.4f} | "
              f"EQM val. = {np.mean(eqms):.4f} | épocas = {np.mean(epocas):.0f}")
    linhas_a = np.array(linhas_a)
    np.savetxt(os.path.join(PASTA_RES, "ajuste_adaline.csv"), linhas_a, fmt="%.6g", delimiter=";",
               encoding="utf-8", header="eta;acuracia_media_val;eqm_medio_val;epocas_medias", comments="")
    # Critério do ADALINE: menor EQM de validação (é a função que ele minimiza).
    eta_adaline = float(linhas_a[np.argmin(linhas_a[:, 2]), 0])

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for eta, curva in curvas_a.items():
        ax.plot(np.arange(1, len(curva) + 1), curva, label=f"η = {eta:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Época (escala log)")
    ax.set_ylabel("EQM de treino")
    ax.set_title("ADALINE: efeito da taxa de aprendizado")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "02_ajuste_adaline.png"), dpi=200)
    plt.close(fig)

    # ---- MLP: topologia x η ----
    resultados_mlp = []
    inicio = time.time()
    for topologia in GRADE_TOPOLOGIAS_MLP:
        for eta in GRADE_ETA_MLP:
            accs, epocas = [], []
            for k, (Xtr, ytr, Xval, yval) in enumerate(dados):
                modelo = MLP(topologia, eta, HP_MLP["max_epocas"], HP_MLP["precisao"], semente=k)
                modelo.fit(Xtr, ytr)
                accs.append(avaliar(modelo, Xval, yval)[1]["acuracia"])
                epocas.append(modelo.epocas_treinadas)
            resultados_mlp.append(dict(topologia=topologia, eta=eta, acuracia=float(np.mean(accs)),
                                       desvio=float(np.std(accs)), epocas=float(np.mean(epocas)),
                                       pesos=numero_de_pesos(topologia)))
            print(f"  MLP {'-'.join(map(str, topologia)):>6} η={eta:<4g} "
                  f"acurácia val. = {np.mean(accs):.4f} | épocas = {np.mean(epocas):.0f} "
                  f"| {time.time() - inicio:5.1f}s")
    linhas_m = [["-".join(map(str, r["topologia"])), f"{r['eta']:g}", f"{r['acuracia']:.4f}",
                 f"{r['desvio']:.4f}", f"{r['epocas']:.0f}", str(r["pesos"])] for r in resultados_mlp]
    np.savetxt(os.path.join(PASTA_RES, "ajuste_mlp.csv"), np.array(linhas_m, dtype=object), fmt="%s",
               delimiter=";", encoding="utf-8",
               header="topologia;eta;acuracia_media_val;desvio;epocas_medias;n_pesos",
               comments="")
    escolhida = escolher_mlp(resultados_mlp)

    # Mapa de calor da acurácia de validação (topologia x η).
    grade = np.array([[r["acuracia"] for r in resultados_mlp if r["topologia"] == t]
                      for t in GRADE_TOPOLOGIAS_MLP])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(grade, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                xticklabels=[f"η={e:g}" for e in GRADE_ETA_MLP],
                yticklabels=["-".join(map(str, t)) for t in GRADE_TOPOLOGIAS_MLP])
    ax.set_ylabel("Neurônios nas camadas ocultas")
    ax.set_title("MLP: acurácia média de validação")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "03_ajuste_mlp.png"), dpi=200)
    plt.close(fig)

    hp_perceptron = dict(HP_PERCEPTRON)
    hp_adaline = dict(HP_ADALINE, taxa_aprendizado=eta_adaline)
    hp_mlp = dict(HP_MLP, camadas_ocultas=escolhida["topologia"], taxa_aprendizado=escolhida["eta"])
    return hp_perceptron, hp_adaline, hp_mlp


# =============================================================================
# 3. Underfitting x overfitting (MLP)
# =============================================================================
def estudo_topologias(X_bruto, y, hp_adequada):
    """Treina as três topologias nas MESMAS partições 80/20.

    Uma única partição pode ser enganosa (por exemplo, uma inicialização ruim
    que prende a rede em um platô). Por isso cada topologia é treinada em
    PARTICOES_TOPOLOGIAS partições; a tabela traz média e desvio-padrão e as
    figuras mostram a 1ª partição.
    """
    rng = np.random.default_rng(SEMENTE + 2000)
    particoes = [U.dividir_treino_teste(y.size, PROPORCAO_TREINO, rng) for _ in range(PARTICOES_TOPOLOGIAS)]
    casos = [("Subdimensionada", TOPOLOGIA_SUB), ("Adequada", hp_adequada),
             ("Superdimensionada", TOPOLOGIA_SUPER)]

    linhas, treinados = [], []
    por_particao = []
    for nome, hp in casos:
        topologia = "2-" + "-".join(map(str, hp["camadas_ocultas"])) + "-1"
        registros = []
        for k, (itr, ite) in enumerate(particoes):
            Xtr, Xte, minimo, maximo = U.preparar_particao(X_bruto, itr, ite)
            inicio = time.time()
            modelo = MLP(**hp, semente=SEMENTE + k).fit(Xtr, y[itr], Xte, y[ite])
            duracao = time.time() - inicio
            _, met_tr = avaliar(modelo, Xtr, y[itr])
            M_te, met_te = avaliar(modelo, Xte, y[ite])
            registro = [modelo.epocas_treinadas, modelo.eqm_treino[-1], modelo.eqm_teste[-1],
                        met_tr["acuracia"], met_te["acuracia"], met_te["sensibilidade"],
                        met_te["especificidade"], duracao]
            registros.append(registro)
            por_particao.append([nome, str(k + 1)] + [f"{v:.6g}" for v in registro])
            print(f"  {nome:<17} {topologia:<12} partição {k + 1}: épocas={modelo.epocas_treinadas:<6} "
                  f"EQM tr={modelo.eqm_treino[-1]:.5f} te={modelo.eqm_teste[-1]:.5f} | "
                  f"acur. tr={met_tr['acuracia']:.4f} te={met_te['acuracia']:.4f} "
                  f"sens={met_te['sensibilidade']:.4f} esp={met_te['especificidade']:.4f} ({duracao:.1f}s)")
            if k == 0:  # a 1ª partição (a mesma para as três topologias) é usada nas figuras
                treinados.append((nome, hp, modelo, M_te, met_te))
                minimo_fig, maximo_fig = minimo, maximo
        registros = np.array(registros)
        media, desvio = registros.mean(axis=0), registros.std(axis=0, ddof=1)
        linha = [nome, topologia, str(numero_de_pesos(hp["camadas_ocultas"])), f"{hp['taxa_aprendizado']:g}",
                 f"{media[0]:.0f}"]
        for j in range(1, 7):
            linha += [f"{media[j]:.6f}", f"{desvio[j]:.6f}"]
        linha.append(f"{media[7]:.2f}")
        linhas.append(linha)

        # Figura individual (1ª partição): matriz de confusão, curva e fronteira.
        nome_, hp_, modelo, M_te, met_te = treinados[-1]
        fig, eixos = plt.subplots(1, 3, figsize=(12, 3.8))
        U.plot_matriz_confusao(eixos[0], M_te, ROTULOS_BINARIOS,
                               f"Matriz de confusão (teste)\nacur.={met_te['acuracia']:.4f} "
                               f"sens.={met_te['sensibilidade']:.4f} esp.={met_te['especificidade']:.4f}")
        U.plot_curva_aprendizado(eixos[1], modelo.eqm_treino, modelo.eqm_teste,
                                 "Curva de aprendizado", escala_log=True)
        U.plot_fronteira_decisao(eixos[2], modelo, X_bruto, y, minimo_fig, maximo_fig, "Fronteira de decisão")
        fig.suptitle(f"MLP {nome.lower()}: {topologia} ({descrever_hp(hp)})")
        fig.tight_layout()
        fig.savefig(os.path.join(PASTA_FIG, f"04_topologia_{nome.lower()}.png"), dpi=200)
        plt.close(fig)

    np.savetxt(os.path.join(PASTA_RES, "topologias_mlp.csv"), np.array(linhas, dtype=object), fmt="%s",
               delimiter=";", encoding="utf-8", comments="",
               header="caso;topologia;n_pesos;eta;epocas_media;eqm_treino;eqm_treino_dp;eqm_teste;"
                      "eqm_teste_dp;acuracia_treino;acuracia_treino_dp;acuracia_teste;acuracia_teste_dp;"
                      "sensibilidade_teste;sensibilidade_teste_dp;especificidade_teste;"
                      "especificidade_teste_dp;tempo_medio_s")
    np.savetxt(os.path.join(PASTA_RES, "topologias_mlp_por_particao.csv"),
               np.array(por_particao, dtype=object), fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="caso;particao;epocas;eqm_treino;eqm_teste;acuracia_treino;acuracia_teste;"
                      "sensibilidade_teste;especificidade_teste;tempo_s")

    # Figura comparativa (1ª partição): curvas (linha de cima) e fronteiras (linha de baixo).
    fig, eixos = plt.subplots(2, 3, figsize=(10, 6.8))
    for j, (nome, hp, modelo, M_te, met_te) in enumerate(treinados):
        topologia = "2-" + "-".join(map(str, hp["camadas_ocultas"])) + "-1"
        U.plot_curva_aprendizado(eixos[0, j], modelo.eqm_treino, modelo.eqm_teste,
                                 f"{nome} ({topologia})", escala_log=True)
        melhor_epoca = int(np.argmin(modelo.eqm_teste)) + 1
        eixos[0, j].axvline(melhor_epoca, color="gray", ls=":", lw=1)
        U.plot_fronteira_decisao(eixos[1, j], modelo, X_bruto, y, minimo_fig, maximo_fig,
                                 f"{nome}\nacurácia de teste = {met_te['acuracia']:.4f}")
    fig.suptitle("MLP: subdimensionamento x topologia adequada x superdimensionamento")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "05_topologias_comparacao.png"), dpi=200)
    plt.close(fig)

    # Matrizes de confusão lado a lado (útil para o relatório).
    fig, eixos = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, (nome, hp, modelo, M_te, met_te) in zip(eixos, treinados):
        U.plot_matriz_confusao(ax, M_te, ROTULOS_BINARIOS,
                               f"{nome}\nacur.={met_te['acuracia']:.3f} sens.={met_te['sensibilidade']:.3f} "
                               f"esp.={met_te['especificidade']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "06_topologias_matrizes.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# 4. Validação por amostragem aleatória
# =============================================================================
def validacao_monte_carlo(X_bruto, y, hps, R):
    """Executa R rodadas com partições aleatórias 80/20.

    Em cada rodada os dados são embaralhados, normalizados com os parâmetros
    do treino e apresentados aos três modelos. Guardam-se as métricas, a
    matriz de confusão e as curvas de aprendizado de todas as rodadas.
    """
    rng = np.random.default_rng(SEMENTE)
    nomes = list(criar_modelos(*hps, semente=0).keys())
    resultados = {n: dict(metricas=np.zeros((R, len(METRICAS))), matrizes=np.zeros((R, 2, 2), int),
                          curvas=[], epocas=np.zeros(R, int), tempo=0.0) for n in nomes}
    exemplo = {}  # modelos da 1ª rodada, usados na figura de fronteiras
    progresso = U.Progresso(R, "Rodadas")
    for r in range(R):
        itr, ite = U.dividir_treino_teste(y.size, PROPORCAO_TREINO, rng)
        Xtr, Xte, minimo, maximo = U.preparar_particao(X_bruto, itr, ite)
        semente_rodada = int(rng.integers(0, 2 ** 31 - 1))
        for nome, modelo in criar_modelos(*hps, semente=semente_rodada).items():
            inicio = time.time()
            modelo.fit(Xtr, y[itr], Xte, y[ite])
            resultados[nome]["tempo"] += time.time() - inicio
            M, met = avaliar(modelo, Xte, y[ite])
            resultados[nome]["metricas"][r] = [met[m] for m in METRICAS]
            resultados[nome]["matrizes"][r] = M
            resultados[nome]["curvas"].append((np.array(modelo.eqm_treino, dtype=np.float32),
                                               np.array(modelo.eqm_teste, dtype=np.float32)))
            resultados[nome]["epocas"][r] = modelo.epocas_treinadas
            if r == 0:
                exemplo[nome] = (modelo, minimo, maximo)
        medias = " ".join(f"{n.split()[0][:4]}={resultados[n]['metricas'][:r + 1, 0].mean():.3f}"
                          for n in nomes)
        progresso.atualizar(r + 1, f"| acurácia média: {medias}")
    return resultados, exemplo


# =============================================================================
# 5. Rodadas extremas, tabelas e gráficos
# =============================================================================
def figuras_extremos(resultados):
    """Matriz de confusão e curva de aprendizado da rodada de maior e de menor
    valor de cada métrica, para cada modelo."""
    registros = []
    for nome, res in resultados.items():
        sufixo = nome.lower().replace(" ", "_")
        log = nome == "MLP"
        painel, eixos_p = plt.subplots(len(METRICAS), 4, figsize=(11, 2.8 * len(METRICAS)))
        for i, metrica in enumerate(METRICAS):
            valores = res["metricas"][:, i]
            # Em caso de empate (ex.: várias rodadas com sensibilidade 1,0) o
            # argmax/argmin devolve a primeira rodada em que o valor ocorre.
            r_max, r_min = int(np.argmax(valores)), int(np.argmin(valores))
            registros.append([nome, NOME_METRICA[metrica], r_max + 1, f"{valores[r_max]:.4f}",
                              r_min + 1, f"{valores[r_min]:.4f}"])
            fig, eixos = plt.subplots(2, 2, figsize=(8, 6.6))
            for coluna, (rotulo, r) in enumerate((("Maior", r_max), ("Menor", r_min))):
                titulo = f"{rotulo} {NOME_METRICA[metrica].lower()}: {valores[r]:.4f} (rodada {r + 1})"
                # Mesmo conteúdo na figura individual (fonte 10) e no painel (fonte 8).
                for ax_cm, ax_lc, fonte in ((eixos[0, coluna], eixos[1, coluna], 10),
                                            (eixos_p[i, 2 * coluna], eixos_p[i, 2 * coluna + 1], 8)):
                    U.plot_matriz_confusao(ax_cm, res["matrizes"][r], ROTULOS_BINARIOS, titulo,
                                           tamanho_fonte=fonte, tamanho_titulo=fonte)
                    U.plot_curva_aprendizado(ax_lc, res["curvas"][r][0], res["curvas"][r][1],
                                             f"Curva de aprendizado (rodada {r + 1})", escala_log=log,
                                             tamanho_titulo=fonte)
            fig.suptitle(f"{nome}: rodadas de maior e menor {NOME_METRICA[metrica].lower()}")
            fig.tight_layout()
            fig.savefig(os.path.join(PASTA_FIG, f"08_extremos_{sufixo}_{metrica}.png"), dpi=200)
            plt.close(fig)
        painel.suptitle(f"{nome}: rodadas extremas de cada métrica")
        painel.tight_layout()
        painel.savefig(os.path.join(PASTA_FIG, f"09_painel_extremos_{sufixo}.png"), dpi=150)
        plt.close(painel)
    np.savetxt(os.path.join(PASTA_RES, "rodadas_extremas.csv"), np.array(registros, dtype=object),
               fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="modelo;metrica;rodada_maior;valor_maior;rodada_menor;valor_menor")


def tabelas_e_distribuicoes(resultados, R):
    nomes = list(resultados.keys())
    for i, metrica in enumerate(METRICAS):
        valores = [resultados[n]["metricas"][:, i] for n in nomes]
        U.salvar_tabela_resumo(os.path.join(PASTA_RES, f"tabela_{metrica}"), nomes, valores,
                               f"{NOME_METRICA[metrica]} - {R} rodadas (80% treino / 20% teste)")
        print()
        U.plot_box_violin(valores, nomes, f"{NOME_METRICA[metrica]} nas {R} rodadas",
                          NOME_METRICA[metrica], os.path.join(PASTA_FIG, f"10_boxplot_{metrica}.png"))

    # Figura única: boxplots (linha de cima) e violinos (linha de baixo) das 3 métricas.
    fig, eixos = plt.subplots(2, len(METRICAS), figsize=(10, 6))
    for i, metrica in enumerate(METRICAS):
        valores = [resultados[n]["metricas"][:, i] for n in nomes]
        U.desenhar_box_violin(eixos[0, i], eixos[1, i], valores, ["Perceptron", "ADALINE", "MLP"],
                              NOME_METRICA[metrica])
        eixos[0, i].set_title(f"{NOME_METRICA[metrica]} - boxplot", fontsize=10)
        eixos[1, i].set_title(f"{NOME_METRICA[metrica]} - violin plot", fontsize=10)
    fig.suptitle(f"Distribuição das métricas nas {R} rodadas")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "11_box_violin_metricas.png"), dpi=200)
    plt.close(fig)

    # Todas as métricas de todas as rodadas (permite refazer qualquer análise).
    cabecalho = ["rodada"] + [f"{n.split()[0].lower()}_{m}" for n in nomes for m in METRICAS]
    tabela = np.column_stack([np.arange(1, R + 1)] + [resultados[n]["metricas"] for n in nomes])
    np.savetxt(os.path.join(PASTA_RES, "metricas_por_rodada.csv"), tabela, delimiter=";", encoding="utf-8",
               fmt=["%d"] + ["%.6f"] * (tabela.shape[1] - 1), header=";".join(cabecalho), comments="")

    # Tempo e épocas médias de treinamento por modelo.
    linhas = [[n, f"{resultados[n]['epocas'].mean():.1f}", f"{resultados[n]['epocas'].min()}",
               f"{resultados[n]['epocas'].max()}", f"{resultados[n]['tempo'] / R:.4f}"] for n in nomes]
    np.savetxt(os.path.join(PASTA_RES, "custo_treinamento.csv"), np.array(linhas, dtype=object),
               fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="modelo;epocas_media;epocas_min;epocas_max;tempo_medio_por_rodada_s")


def figura_fronteiras(exemplo, X_bruto, y):
    """Fronteiras dos três modelos treinados na 1ª rodada da validação."""
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (nome, (modelo, minimo, maximo)) in zip(eixos, exemplo.items()):
        U.plot_fronteira_decisao(ax, modelo, X_bruto, y, minimo, maximo, nome)
    fig.suptitle("Fronteiras de decisão (1ª rodada da validação)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG, "07_fronteiras_modelos.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# Programa principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Etapa 1 - spiral_d.csv")
    parser.add_argument("--R", type=int, default=500, help="número de rodadas (padrão: 500)")
    parser.add_argument("--sem-ajuste", action="store_true",
                        help="pula a busca de hiperparâmetros e usa os valores padrão")
    args = parser.parse_args()
    U.configurar_terminal()
    U.garantir_pasta(PASTA_FIG)
    U.garantir_pasta(PASTA_RES)
    inicio = time.time()

    print("[1/5] Organização dos dados e gráfico de espalhamento")
    X_bruto, y = secao_dados()

    print("[2/5] Escolha de hiperparâmetros")
    if args.sem_ajuste:
        hps = (dict(HP_PERCEPTRON), dict(HP_ADALINE), dict(HP_MLP))
        print("  (busca ignorada: usando valores padrão)")
    else:
        hps = busca_hiperparametros(X_bruto, y)
    for nome, hp in zip(("Perceptron Simples", "ADALINE", "MLP"), hps):
        print(f"  -> {nome}: {descrever_hp(hp)}")
    with open(os.path.join(PASTA_RES, "hiperparametros_escolhidos.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(("perceptron", "adaline", "mlp"), hps)), f, indent=2, ensure_ascii=False)

    print("[3/5] Underfitting x overfitting da MLP")
    estudo_topologias(X_bruto, y, hps[2])

    print(f"[4/5] Validação por amostragem aleatória (R = {args.R})")
    resultados, exemplo = validacao_monte_carlo(X_bruto, y, hps, args.R)
    figura_fronteiras(exemplo, X_bruto, y)

    print("[5/5] Rodadas extremas, tabelas e gráficos")
    figuras_extremos(resultados)
    tabelas_e_distribuicoes(resultados, args.R)
    print(f"Concluído em {time.time() - inicio:.1f}s. Figuras em {PASTA_FIG} | resultados em {PASTA_RES}")


if __name__ == "__main__":
    main()
