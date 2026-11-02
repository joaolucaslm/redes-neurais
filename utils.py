"""
utils.py
--------
Funções de apoio usadas nas duas etapas do trabalho:

  * carregamento dos dados (CSV da espiral e imagens do reconhecimento facial);
  * pré-processamento: normalização, inclusão do bias, divisão treino/teste e
    codificação one-hot bipolar;
  * avaliação: matriz de confusão (calculada por contagem) e métricas;
  * gráficos (matplotlib/seaborn) e gravação de tabelas (numpy).

Bibliotecas utilizadas: numpy, matplotlib, seaborn e opencv-python (este último
somente para ler e redimensionar as imagens).
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # gera os arquivos de imagem sem abrir janelas
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import cv2


# =============================================================================
# 1. Carregamento dos dados
# =============================================================================
def carregar_espiral(caminho_csv):
    """Lê o spiral_d.csv com numpy.

    Colunas 1 e 2 -> atributos (x1, x2); coluna 3 -> rótulo da classe.
    Retorna X_bruto com dimensão p x N (cada coluna é uma amostra) e o vetor
    de rótulos com N posições.
    """
    dados = np.loadtxt(caminho_csv, delimiter=",")
    X_bruto = dados[:, :2].T.copy()
    rotulos = dados[:, 2].copy()
    return X_bruto, rotulos


def converter_para_bipolar(rotulos):
    """Garante rótulos +1/-1.

    Se o arquivo já estiver em {-1, +1} nada muda. Caso venha em {0, 1}
    (ou qualquer outro par de valores), o maior valor vira +1 e o menor vira -1.
    """
    valores = np.unique(rotulos)
    if valores.size != 2:
        raise ValueError(f"Esperadas 2 classes, encontradas {valores.size}: {valores}")
    if np.array_equal(valores, np.array([-1.0, 1.0])):
        return rotulos.astype(float)
    return np.where(rotulos == valores.max(), 1.0, -1.0)


def localizar_pasta_faces(raiz_repositorio):
    """Encontra a pasta das imagens faciais, aceitando 'RecFac' ou 'ReCFac'.

    Se a pasta tiver apenas uma subpasta e nenhuma imagem (por exemplo
    RecFac/RecFac/, efeito comum ao descompactar um .zip), o código desce
    um nível até chegar à pasta que contém uma subpasta por pessoa.
    """
    candidatas = [d for d in os.listdir(raiz_repositorio)
                  if d.lower() == "recfac" and os.path.isdir(os.path.join(raiz_repositorio, d))]
    if not candidatas:
        raise FileNotFoundError("Pasta 'RecFac' não encontrada na raiz do repositório.")
    pasta = os.path.join(raiz_repositorio, candidatas[0])
    while True:
        itens = os.listdir(pasta)
        subpastas = [i for i in itens if os.path.isdir(os.path.join(pasta, i))]
        arquivos = [i for i in itens if os.path.isfile(os.path.join(pasta, i))]
        if len(subpastas) == 1 and len(arquivos) == 0:
            pasta = os.path.join(pasta, subpastas[0])
        else:
            return pasta


def carregar_faces(raiz_repositorio, lado):
    """Lê todas as imagens em escala de cinza e as redimensiona para lado x lado.

    O nome de cada subpasta é usado como rótulo (classe) da pessoa.
    Retorna:
        X_bruto  -> matriz p x N (p = lado*lado pixels, N = nº de imagens)
        classes  -> vetor com o índice (0..C-1) da pessoa de cada imagem
        nomes    -> lista com os nomes das C pessoas (ordem alfabética)
    """
    pasta = localizar_pasta_faces(raiz_repositorio)
    nomes = sorted(d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d)))
    extensoes = (".png", ".pgm", ".jpg", ".jpeg", ".bmp")
    vetores, classes = [], []
    for indice, pessoa in enumerate(nomes):
        pasta_pessoa = os.path.join(pasta, pessoa)
        for arquivo in sorted(os.listdir(pasta_pessoa)):
            if not arquivo.lower().endswith(extensoes):
                continue
            imagem = cv2.imread(os.path.join(pasta_pessoa, arquivo), cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue
            # INTER_AREA calcula a média dos pixels de cada região, o que é o
            # recomendado para reduzir imagens (evita serrilhado/aliasing).
            reduzida = cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)
            vetores.append(reduzida.astype(float).ravel())  # vetorização da imagem
            classes.append(indice)
    X_bruto = np.array(vetores).T
    return X_bruto, np.array(classes, dtype=int), nomes, pasta


# =============================================================================
# 2. Pré-processamento
# =============================================================================
def ajustar_normalizacao(X_bruto):
    """Calcula o mínimo e o máximo de cada atributo (linha de X).

    É chamada APENAS com os dados de treino da rodada; os mesmos valores são
    depois aplicados ao teste. Assim nenhuma informação do teste vaza para o
    treinamento.
    """
    return X_bruto.min(axis=1, keepdims=True), X_bruto.max(axis=1, keepdims=True)


def normalizar(X_bruto, minimo, maximo):
    """Normalização min-max para o intervalo [-1, 1].

    x' = 2 (x - min) / (max - min) - 1

    O intervalo [-1, 1] foi escolhido por coincidir com a faixa da tangente
    hiperbólica, dos rótulos bipolares e da entrada de bias (-1). Atributos
    constantes no treino (max = min) recebem o valor 0 para evitar divisão
    por zero.
    """
    amplitude = maximo - minimo
    constante = amplitude == 0
    amplitude = np.where(constante, 1.0, amplitude)
    X_norm = 2.0 * (X_bruto - minimo) / amplitude - 1.0
    if np.any(constante):
        X_norm[constante[:, 0], :] = 0.0
    return X_norm


def adicionar_bias(X):
    """Acrescenta a linha de -1 no topo, resultando em X com dimensão (p+1) x N."""
    return np.vstack((-np.ones((1, X.shape[1])), X))


def dividir_treino_teste(N, proporcao_treino, rng):
    """Embaralha os índices 0..N-1 e separa a proporção pedida para treino."""
    indices = rng.permutation(N)
    n_treino = int(round(proporcao_treino * N))
    return indices[:n_treino], indices[n_treino:]


def preparar_particao(X_bruto, indices_treino, indices_teste):
    """Monta X_treino e X_teste já normalizados e com o bias (-1).

    Retorna também o mínimo e o máximo usados, necessários para desenhar a
    fronteira de decisão no espaço original dos dados.
    """
    minimo, maximo = ajustar_normalizacao(X_bruto[:, indices_treino])
    X_treino = adicionar_bias(normalizar(X_bruto[:, indices_treino], minimo, maximo))
    X_teste = adicionar_bias(normalizar(X_bruto[:, indices_teste], minimo, maximo))
    return X_treino, X_teste, minimo, maximo


def one_hot_bipolar(classes, n_classes):
    """Codificação one-hot bipolar: +1 na posição da classe e -1 nas demais.

    Retorna Y com dimensão C x N (uma coluna por amostra).
    """
    Y = -np.ones((n_classes, classes.size))
    Y[classes, np.arange(classes.size)] = 1.0
    return Y


# =============================================================================
# 3. Avaliação: matriz de confusão e métricas
# =============================================================================
def matriz_confusao(reais, previstos, n_classes):
    """Matriz de confusão construída pela contagem das amostras.

    Linha = classe real, coluna = classe prevista. Cada amostra soma 1 na
    célula correspondente ao par (real, previsto).
    """
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for real, previsto in zip(reais, previstos):
        matriz[int(real), int(previsto)] += 1
    return matriz


def bipolar_para_indice(rotulos):
    """Converte +1 -> 0 e -1 -> 1.

    Com essa ordem a matriz binária fica [[VP, FN], [FP, VN]], tomando a
    classe +1 como a classe positiva.
    """
    return np.where(np.asarray(rotulos).ravel() > 0, 0, 1)


def acuracia(matriz):
    """Fração de acertos: soma da diagonal dividida pelo total de amostras."""
    return np.trace(matriz) / matriz.sum()


def metricas_binarias(matriz):
    """Acurácia, sensibilidade e especificidade a partir da matriz 2x2.

    sensibilidade = VP / (VP + FN)  -> taxa de acerto da classe positiva (+1)
    especificidade = VN / (VN + FP) -> taxa de acerto da classe negativa (-1)
    """
    VP, FN = matriz[0]
    FP, VN = matriz[1]
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0.0
    return {
        "acuracia": acuracia(matriz),
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
    }


def resumo_estatistico(valores):
    """Média, desvio-padrão amostral (ddof=1), maior e menor valor."""
    valores = np.asarray(valores, dtype=float)
    desvio = valores.std(ddof=1) if valores.size > 1 else 0.0
    return valores.mean(), desvio, valores.max(), valores.min()


# =============================================================================
# 4. Saída de resultados (tabelas e progresso)
# =============================================================================
def configurar_terminal():
    """Força UTF-8 na saída para que acentos e letras gregas (η, ε) não gerem
    erro quando a saída do script é redirecionada para arquivo no Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def garantir_pasta(caminho):
    os.makedirs(caminho, exist_ok=True)
    return caminho


def salvar_tabela_resumo(caminho_sem_extensao, nomes_modelos, valores_por_modelo, titulo):
    """Grava a tabela Média | Desvio-Padrão | Maior | Menor de uma métrica.

    São gerados dois arquivos com numpy: um .csv (fácil de reaproveitar) e um
    .txt alinhado (fácil de ler). Retorna a tabela numérica (modelos x 4).
    """
    tabela = np.array([resumo_estatistico(v) for v in valores_por_modelo])
    cabecalho = "Modelo;Media;Desvio-Padrao;Maior Valor;Menor Valor"
    linhas = np.array([[nome] + [f"{x:.4f}" for x in linha]
                       for nome, linha in zip(nomes_modelos, tabela)], dtype=object)
    np.savetxt(caminho_sem_extensao + ".csv", linhas, fmt="%s", delimiter=";",
               header=cabecalho, comments="", encoding="utf-8")

    largura = max(len(n) for n in nomes_modelos) + 2
    texto = [titulo, "-" * (largura + 52),
             f"{'Modelo':<{largura}}{'Media':>12}{'Desvio-Padrao':>15}{'Maior':>12}{'Menor':>12}",
             "-" * (largura + 52)]
    for nome, linha in zip(nomes_modelos, tabela):
        texto.append(f"{nome:<{largura}}{linha[0]:>12.4f}{linha[1]:>15.4f}"
                     f"{linha[2]:>12.4f}{linha[3]:>12.4f}")
    with open(caminho_sem_extensao + ".txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(texto) + "\n")
    print("\n".join(texto))
    return tabela


class Progresso:
    """Mostra no terminal o andamento das rodadas e uma estimativa de tempo."""

    def __init__(self, total, descricao):
        self.total = total
        self.descricao = descricao
        self.inicio = time.time()

    def atualizar(self, feitos, extra=""):
        decorrido = time.time() - self.inicio
        restante = decorrido / feitos * (self.total - feitos) if feitos else 0.0
        sys.stdout.write(f"\r  {self.descricao}: {feitos}/{self.total} "
                         f"| {decorrido:6.1f}s decorridos | ~{restante:6.1f}s restantes {extra}   ")
        sys.stdout.flush()
        if feitos == self.total:
            sys.stdout.write("\n")


# =============================================================================
# 5. Gráficos
# =============================================================================
CORES_CLASSES = {1.0: "#1f77b4", -1.0: "#d62728"}


def plot_dispersao(X_bruto, rotulos, caminho, titulo="Gráfico de espalhamento"):
    """Espalhamento dos dois atributos, colorido por classe."""
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    for classe, marcador in ((1.0, "o"), (-1.0, "s")):
        mascara = rotulos == classe
        ax.scatter(X_bruto[0, mascara], X_bruto[1, mascara], s=10, marker=marcador,
                   c=CORES_CLASSES[classe], alpha=0.75, edgecolors="none",
                   label=f"Classe {int(classe):+d} (N = {mascara.sum()})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(titulo)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def plot_matriz_confusao(ax, matriz, rotulos, titulo, anotar=True, tamanho_fonte=10, tamanho_titulo=10):
    """Desenha a matriz (já calculada por contagem) como heatmap do seaborn.

    Nas matrizes grandes (20 x 20) só as células diferentes de zero recebem
    número, para não poluir o gráfico.
    """
    if anotar and len(rotulos) > 2:
        anotacoes = np.where(matriz > 0, matriz.astype(str), "")
        fmt = ""
    else:
        anotacoes, fmt = anotar, "d"
    sns.heatmap(matriz, annot=anotacoes, fmt=fmt, cmap="Blues", cbar=len(rotulos) > 2,
                xticklabels=rotulos, yticklabels=rotulos, ax=ax, square=True,
                linewidths=0.5, linecolor="white", annot_kws={"size": tamanho_fonte})
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo, fontsize=tamanho_titulo)


def plot_curva_aprendizado(ax, eqm_treino, eqm_teste, titulo, escala_log=False, tamanho_titulo=10):
    """Curva de aprendizado: EQM por época no treino e no teste."""
    epocas = np.arange(1, len(eqm_treino) + 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))  # épocas inteiras
    ax.plot(epocas, eqm_treino, label="EQM treino", color="#1f77b4", lw=1.4)
    if eqm_teste is not None and len(eqm_teste) > 0:
        ax.plot(epocas, eqm_teste, label="EQM teste", color="#ff7f0e", lw=1.4, ls="--")
    ax.set_xlabel("Época")
    ax.set_ylabel("EQM")
    if escala_log:
        ax.set_yscale("log")
    ax.set_title(titulo, fontsize=tamanho_titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_fronteira_decisao(ax, modelo, X_bruto, rotulos, minimo, maximo, titulo, passo=0.15):
    """Fronteira de decisão no espaço original (x1, x2).

    Uma grade de pontos é normalizada com os MESMOS parâmetros do treino,
    recebe o bias e é classificada pelo modelo.
    """
    margem = 1.5
    x1 = np.arange(X_bruto[0].min() - margem, X_bruto[0].max() + margem, passo)
    x2 = np.arange(X_bruto[1].min() - margem, X_bruto[1].max() + margem, passo)
    G1, G2 = np.meshgrid(x1, x2)
    grade = np.vstack((G1.ravel(), G2.ravel()))
    previsto = modelo.predict(adicionar_bias(normalizar(grade, minimo, maximo)))
    Z = previsto.reshape(G1.shape)
    ax.contourf(G1, G2, Z, levels=[-2, 0, 2], colors=["#f4b6b6", "#b7d3ee"], alpha=0.8)
    ax.contour(G1, G2, Z, levels=[0], colors="k", linewidths=0.8)
    for classe, marcador in ((1.0, "o"), (-1.0, "s")):
        m = rotulos == classe
        ax.scatter(X_bruto[0, m], X_bruto[1, m], s=6, marker=marcador,
                   c=CORES_CLASSES[classe], edgecolors="none", alpha=0.8,
                   label=f"Classe {int(classe):+d}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal", adjustable="box")


def desenhar_box_violin(ax_box, ax_violin, valores_por_modelo, nomes, rotulo_y):
    """Boxplot e violin plot de um conjunto de modelos em dois eixos dados."""
    paleta = sns.color_palette("Set2", len(nomes))
    ax_box.boxplot(list(valores_por_modelo), tick_labels=nomes, patch_artist=True,
                   boxprops=dict(alpha=0.8), medianprops=dict(color="k"),
                   flierprops=dict(markersize=3))
    for caixa, cor in zip(ax_box.patches, paleta):
        caixa.set_facecolor(cor)
    # Formato "longo": um vetor com os valores e outro com o nome do modelo.
    valores_longos = np.concatenate([np.asarray(v, dtype=float) for v in valores_por_modelo])
    grupos = np.concatenate([[n] * len(v) for n, v in zip(nomes, valores_por_modelo)])
    sns.violinplot(x=grupos, y=valores_longos, hue=grupos, palette=paleta, legend=False,
                   ax=ax_violin, cut=0, inner="quartile", linewidth=0.8)
    for ax in (ax_box, ax_violin):
        ax.set_ylabel(rotulo_y)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)


def plot_box_violin(valores_por_modelo, nomes, titulo, rotulo_y, caminho):
    """Boxplot e violin plot lado a lado para comparar a distribuição das rodadas."""
    fig, eixos = plt.subplots(1, 2, figsize=(8, 3.4))
    desenhar_box_violin(eixos[0], eixos[1], valores_por_modelo, nomes, rotulo_y)
    eixos[0].set_title("Boxplot", fontsize=10)
    eixos[1].set_title("Violin plot", fontsize=10)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
