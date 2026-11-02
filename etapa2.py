"""
etapa2.py
---------
Segunda etapa do trabalho: reconhecimento facial de 20 pessoas (640 imagens)
com Perceptron Simples, ADALINE e MLP, cada um com 20 neurônios de saída.

O enunciado traz uma divergência no número de rodadas: o item 5 pede
R = 10 rodadas e o item 7 fala em R = 100 rodadas. Por isso R é um parâmetro
e, por padrão, o script executa AS DUAS configurações, gravando as saídas em
pastas separadas:
    figuras/etapa2_R10/   resultados/etapa2_R10/
    figuras/etapa2_R100/  resultados/etapa2_R100/
Itens comuns (exemplos de imagens, busca de hiperparâmetros e comparação
entre R = 10 e R = 100) ficam em figuras/etapa2/ e resultados/etapa2/.

Uso:
    python etapa2.py                  # R = 10 e R = 100
    python etapa2.py --R 10           # apenas uma configuração
    python etapa2.py --sem-ajuste     # pula a busca e usa os valores padrão
"""

import argparse
import json
import os
import time

# Uma thread no BLAS: com matrizes pequenas é igual ou mais rápido (ver etapa1.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils as U
from modelos import PerceptronSimples, Adaline, MLP

RAIZ = os.path.dirname(os.path.abspath(__file__))
PASTA_FIG_COMUM = os.path.join(RAIZ, "figuras", "etapa2")
PASTA_RES_COMUM = os.path.join(RAIZ, "resultados", "etapa2")

SEMENTE = 42
PROPORCAO_TREINO = 0.8
# Imagens redimensionadas para 30 x 30 = 900 atributos. Entre as opções do
# enunciado (30, 40, 50 ou 80) é a menor: mantém a estrutura do rosto
# (ver figura de exemplos) e deixa o treino amostra a amostra do Perceptron e
# do ADALINE e o treino da MLP viáveis para 110 rodadas no total.
LADO = 30
N_CLASSES = 20

# -----------------------------------------------------------------------------
# Hiperparâmetros padrão (iguais aos escolhidos pela busca; usados com
# --sem-ajuste). A justificativa de cada valor está no relatório e em
# busca_hiperparametros().
# -----------------------------------------------------------------------------
HP_PERCEPTRON = dict(taxa_aprendizado=0.01, max_epocas=100)
HP_ADALINE = dict(taxa_aprendizado=2e-3, max_epocas=300, precisao=1e-3)
HP_MLP = dict(camadas_ocultas=(50, 50), taxa_aprendizado=0.1, max_epocas=1500, precisao=1e-3)

GRADE_ADALINE = ((5e-4, 1e-3), (1e-3, 1e-3), (2e-3, 1e-3), (1e-3, 1e-4))  # (η, ε)
GRADE_TOPOLOGIAS_MLP = ((20,), (50,), (100,), (50, 50))
GRADE_ETA_MLP = (0.05, 0.1)
PARTICOES_BUSCA = 3
TOLERANCIA_PARCIMONIA = 0.0025  # mesma regra da Etapa 1 (0,25 ponto percentual)


# =============================================================================
# Funções auxiliares
# =============================================================================
def avaliar(modelo, X, classes):
    """Matriz de confusão 20 x 20 (contagem) e acurácia."""
    matriz = U.matriz_confusao(classes, modelo.predict(X), N_CLASSES)
    return matriz, U.acuracia(matriz)


def criar_modelos(hps, semente):
    hp_p, hp_a, hp_m = hps
    return {
        "Perceptron Simples": PerceptronSimples(**hp_p),
        "ADALINE": Adaline(**hp_a, semente=semente),
        "MLP": MLP(**hp_m, semente=semente + 1),
    }


def descrever_hp(hp):
    partes = []
    if "camadas_ocultas" in hp:
        partes.append("ocultas=" + "-".join(str(q) for q in hp["camadas_ocultas"]))
    partes.append(f"η={hp['taxa_aprendizado']:g}")
    partes.append(f"épocas máx.={hp['max_epocas']}")
    if "precisao" in hp:
        partes.append(f"ε={hp['precisao']:g}")
    return ", ".join(partes)


def numero_de_pesos(camadas_ocultas, p=LADO * LADO, c=N_CLASSES):
    neuronios = [p] + list(camadas_ocultas) + [c]
    return sum((neuronios[i] + 1) * neuronios[i + 1] for i in range(len(neuronios) - 1))


def nome_arquivo(nome_modelo):
    return nome_modelo.lower().replace(" ", "_")


# =============================================================================
# 1. Dados
# =============================================================================
def secao_dados():
    X_bruto, classes, nomes, pasta = U.carregar_faces(RAIZ, LADO)
    print(f"  Pasta das imagens: {pasta}")
    print(f"  {len(nomes)} pessoas, {classes.size} imagens; imagens por pessoa: "
          f"{np.bincount(classes).min()} a {np.bincount(classes).max()}")
    # Montagem completa apenas para conferir as dimensões pedidas no enunciado.
    # Nas rodadas, a normalização é refeita usando somente o treino.
    X = U.adicionar_bias(U.normalizar(X_bruto, *U.ajustar_normalizacao(X_bruto)))
    Y = U.one_hot_bipolar(classes, len(nomes))
    print(f"  X com dimensão (p+1) x N = {X.shape} | Y com dimensão C x N = {Y.shape}")
    print(f"  Exemplo de rótulo one-hot ({nomes[0]}): {Y[:, 0].astype(int)}")

    # Uma imagem de cada pessoa já reduzida para 30 x 30.
    fig, eixos = plt.subplots(2, 10, figsize=(10, 2.8))
    for indice, ax in enumerate(eixos.ravel()):
        primeira = np.flatnonzero(classes == indice)[0]
        ax.imshow(X_bruto[:, primeira].reshape(LADO, LADO), cmap="gray")
        ax.set_title(nomes[indice], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Uma imagem de cada pessoa após o redimensionamento para {LADO}x{LADO}")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG_COMUM, "01_exemplos_faces.png"), dpi=200)
    plt.close(fig)
    return X_bruto, classes, nomes


# =============================================================================
# 2. Busca de hiperparâmetros
# =============================================================================
def escolher_mlp(resultados):
    """Mesma regra da Etapa 1: a rede com menos pesos entre as que ficam a até
    0,25 ponto percentual da melhor acurácia média de validação."""
    melhor = max(r["acuracia"] for r in resultados)
    candidatas = [r for r in resultados if r["acuracia"] >= melhor - TOLERANCIA_PARCIMONIA]
    return min(candidatas, key=lambda r: (r["pesos"], -r["acuracia"]))


def busca_hiperparametros(X_bruto, classes):
    rng = np.random.default_rng(SEMENTE + 1000)
    dados = []
    for _ in range(PARTICOES_BUSCA):
        itr, ival = U.dividir_treino_teste(classes.size, PROPORCAO_TREINO, rng)
        Xtr, Xval, _, _ = U.preparar_particao(X_bruto, itr, ival)
        dados.append((Xtr, classes[itr], Xval, classes[ival]))

    # Limite teórico de estabilidade da regra LMS: 0 < η < 2 / λ_max, em que
    # λ_max é o maior autovalor da matriz de correlação R = X X^T / N.
    Xtr0 = dados[0][0]
    lambda_max = float(np.linalg.eigvalsh(Xtr0 @ Xtr0.T / Xtr0.shape[1]).max())
    print(f"  ADALINE: λ_max(R) = {lambda_max:.2f} -> estabilidade exige η < {2 / lambda_max:.5f}")

    linhas_a = []
    for eta, eps in GRADE_ADALINE:
        accs, epocas, tempos = [], [], []
        for k, (Xtr, ctr, Xval, cval) in enumerate(dados):
            inicio = time.time()
            modelo = Adaline(eta, HP_ADALINE["max_epocas"], eps, semente=k)
            modelo.fit(Xtr, U.one_hot_bipolar(ctr, N_CLASSES))
            tempos.append(time.time() - inicio)
            accs.append(avaliar(modelo, Xval, cval)[1])
            epocas.append(modelo.epocas_treinadas)
        linhas_a.append([eta, eps, np.mean(accs), np.mean(epocas), np.mean(tempos)])
        print(f"  ADALINE η={eta:<7g} ε={eps:<7g} acurácia val. = {np.mean(accs):.4f} "
              f"| épocas = {np.mean(epocas):.0f} | {np.mean(tempos):.1f}s por treino")
    linhas_a = np.array(linhas_a)
    np.savetxt(os.path.join(PASTA_RES_COMUM, "ajuste_adaline.csv"), linhas_a, fmt="%.6g", delimiter=";",
               encoding="utf-8", header="eta;precisao;acuracia_media_val;epocas_medias;tempo_medio_s",
               comments="")
    # Escolha: maior acurácia; em empate (diferença < 0,25 p.p.), o mais rápido.
    melhor = linhas_a[:, 2].max()
    candidatos = np.flatnonzero(linhas_a[:, 2] >= melhor - TOLERANCIA_PARCIMONIA)
    escolhido = candidatos[np.argmin(linhas_a[candidatos, 4])]
    hp_adaline = dict(HP_ADALINE, taxa_aprendizado=float(linhas_a[escolhido, 0]),
                      precisao=float(linhas_a[escolhido, 1]))

    resultados = []
    for topologia in GRADE_TOPOLOGIAS_MLP:
        for eta in GRADE_ETA_MLP:
            accs, tempos = [], []
            for k, (Xtr, ctr, Xval, cval) in enumerate(dados):
                inicio = time.time()
                modelo = MLP(topologia, eta, HP_MLP["max_epocas"], HP_MLP["precisao"], semente=k)
                modelo.fit(Xtr, U.one_hot_bipolar(ctr, N_CLASSES))
                tempos.append(time.time() - inicio)
                accs.append(avaliar(modelo, Xval, cval)[1])
            resultados.append(dict(topologia=topologia, eta=eta, acuracia=float(np.mean(accs)),
                                   desvio=float(np.std(accs)), tempo=float(np.mean(tempos)),
                                   pesos=numero_de_pesos(topologia)))
            print(f"  MLP {'-'.join(map(str, topologia)):>6} η={eta:<5g} acurácia val. = "
                  f"{np.mean(accs):.4f} | {np.mean(tempos):.1f}s por treino")
    linhas_m = [["-".join(map(str, r["topologia"])), f"{r['eta']:g}", f"{r['acuracia']:.4f}",
                 f"{r['desvio']:.4f}", f"{r['tempo']:.2f}", str(r["pesos"])] for r in resultados]
    np.savetxt(os.path.join(PASTA_RES_COMUM, "ajuste_mlp.csv"), np.array(linhas_m, dtype=object),
               fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="topologia;eta;acuracia_media_val;desvio;tempo_medio_s;n_pesos")
    escolhida = escolher_mlp(resultados)
    hp_mlp = dict(HP_MLP, camadas_ocultas=escolhida["topologia"], taxa_aprendizado=escolhida["eta"])
    with open(os.path.join(PASTA_RES_COMUM, "estabilidade_adaline.txt"), "w", encoding="utf-8") as f:
        f.write(f"lambda_max(R) = {lambda_max:.4f}\neta_max = 2/lambda_max = {2 / lambda_max:.6f}\n")
    return dict(HP_PERCEPTRON), hp_adaline, hp_mlp


# =============================================================================
# 3. Validação por Monte Carlo
# =============================================================================
def monte_carlo(X_bruto, classes, hps, R):
    """R rodadas com partição aleatória 80/20; métrica: acurácia.

    A semente é a mesma para qualquer R; portanto as 10 rodadas da
    configuração R = 10 são exatamente as 10 primeiras da configuração
    R = 100. A diferença entre as duas reflete apenas o aumento do número
    de rodadas.
    """
    rng = np.random.default_rng(SEMENTE)
    nomes = list(criar_modelos(hps, 0).keys())
    res = {n: dict(acuracias=np.zeros(R), matrizes=np.zeros((R, N_CLASSES, N_CLASSES), int),
                   curvas=[], epocas=np.zeros(R, int), tempo=0.0) for n in nomes}
    progresso = U.Progresso(R, f"Rodadas (R = {R})")
    for r in range(R):
        itr, ite = U.dividir_treino_teste(classes.size, PROPORCAO_TREINO, rng)
        Xtr, Xte, _, _ = U.preparar_particao(X_bruto, itr, ite)
        Ytr = U.one_hot_bipolar(classes[itr], N_CLASSES)
        Yte = U.one_hot_bipolar(classes[ite], N_CLASSES)
        semente_rodada = int(rng.integers(0, 2 ** 31 - 1))
        for nome, modelo in criar_modelos(hps, semente_rodada).items():
            inicio = time.time()
            modelo.fit(Xtr, Ytr, Xte, Yte)
            res[nome]["tempo"] += time.time() - inicio
            matriz, acc = avaliar(modelo, Xte, classes[ite])
            res[nome]["acuracias"][r] = acc
            res[nome]["matrizes"][r] = matriz
            res[nome]["curvas"].append((np.array(modelo.eqm_treino), np.array(modelo.eqm_teste)))
            res[nome]["epocas"][r] = modelo.epocas_treinadas
        medias = " ".join(f"{n.split()[0][:4]}={res[n]['acuracias'][:r + 1].mean():.3f}" for n in nomes)
        progresso.atualizar(r + 1, f"| acurácia média: {medias}")
    return res


# =============================================================================
# 4. Figuras, tabelas e análise por classe
# =============================================================================
def figuras_extremos(res, nomes_pessoas, R, pasta_fig, pasta_res):
    registros = []
    fig_c, eixos_c = plt.subplots(len(res), 2, figsize=(9, 2.5 * len(res)))
    for i, (nome, dados) in enumerate(res.items()):
        acc = dados["acuracias"]
        r_max, r_min = int(np.argmax(acc)), int(np.argmin(acc))
        registros.append([nome, r_max + 1, f"{acc[r_max]:.4f}", r_min + 1, f"{acc[r_min]:.4f}"])

        fig, eixos = plt.subplots(1, 2, figsize=(13, 6.4))
        for ax, (rotulo, r) in zip(eixos, (("Maior", r_max), ("Menor", r_min))):
            U.plot_matriz_confusao(ax, dados["matrizes"][r], nomes_pessoas,
                                   f"{rotulo} acurácia: {acc[r]:.4f} (rodada {r + 1})", tamanho_fonte=9,
                                   tamanho_titulo=12)
            ax.tick_params(axis="x", labelrotation=90, labelsize=9)
            ax.tick_params(axis="y", labelrotation=0, labelsize=9)
        fig.suptitle(f"{nome} - matrizes de confusão das rodadas extremas (R = {R})", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(pasta_fig, f"matrizes_extremos_{nome_arquivo(nome)}.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

        log = nome == "MLP"
        for j, (rotulo, r) in enumerate((("Maior", r_max), ("Menor", r_min))):
            treino, teste = dados["curvas"][r]
            U.plot_curva_aprendizado(eixos_c[i, j], treino, teste,
                                     f"{nome} - {rotulo.lower()} acurácia ({acc[r]:.4f}, rodada {r + 1})",
                                     escala_log=log, tamanho_titulo=9)
    fig_c.suptitle(f"Curvas de aprendizado das rodadas de maior e menor acurácia (R = {R})", fontsize=11)
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(pasta_fig, "curvas_extremos.png"), dpi=200)
    plt.close(fig_c)
    np.savetxt(os.path.join(pasta_res, "rodadas_extremas.csv"), np.array(registros, dtype=object),
               fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="modelo;rodada_maior;acuracia_maior;rodada_menor;acuracia_menor")


def analise_por_classe(res, nomes_pessoas, R, pasta_fig, pasta_res):
    """Soma as matrizes de confusão de todas as rodadas e calcula a taxa de
    acerto de cada pessoa (diagonal / total da linha). Também lista as
    confusões mais frequentes (fora da diagonal)."""
    taxas, texto = [], [f"Análise por classe - soma das matrizes de confusão das {R} rodadas", ""]
    for nome, dados in res.items():
        acumulada = dados["matrizes"].sum(axis=0)
        np.savetxt(os.path.join(pasta_res, f"matriz_acumulada_{nome_arquivo(nome)}.csv"), acumulada,
                   fmt="%d", delimiter=";", encoding="utf-8", header=";".join(nomes_pessoas), comments="")
        taxa = np.diag(acumulada) / np.maximum(acumulada.sum(axis=1), 1)
        taxas.append(taxa)
        ordem = np.argsort(taxa)
        fora = acumulada.copy()
        np.fill_diagonal(fora, 0)
        pares = np.dstack(np.unravel_index(np.argsort(fora.ravel())[::-1][:5], fora.shape))[0]
        texto.append(f"{nome}:")
        texto.append("  piores: " + ", ".join(f"{nomes_pessoas[k]} ({taxa[k]:.3f})" for k in ordem[:4]))
        texto.append("  melhores: " + ", ".join(f"{nomes_pessoas[k]} ({taxa[k]:.3f})"
                                                for k in ordem[::-1][:4]))
        texto.append(f"  pessoas com 100% de acerto: {int(np.sum(taxa == 1.0))} de {len(taxa)}")
        texto.append("  confusões mais frequentes (real -> previsto: vezes): " +
                     ", ".join(f"{nomes_pessoas[a]}->{nomes_pessoas[b]}: {fora[a, b]}"
                               for a, b in pares if fora[a, b] > 0))
        texto.append("")
    taxas = np.array(taxas)
    # Padrão entre modelos: correlação entre as taxas de acerto por pessoa.
    texto.append("Correlação (Pearson) entre as taxas de acerto por pessoa dos modelos:")
    nomes_modelos = list(res.keys())
    correlacao = np.corrcoef(taxas)
    for a in range(len(nomes_modelos)):
        for b in range(a + 1, len(nomes_modelos)):
            texto.append(f"  {nomes_modelos[a]} x {nomes_modelos[b]}: {correlacao[a, b]:.3f}")
    with open(os.path.join(pasta_res, "analise_classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(texto) + "\n")
    print("\n".join(texto))

    linhas = np.column_stack([np.array(nomes_pessoas, dtype=object)] +
                             [np.char.mod("%.4f", t).astype(object) for t in taxas])
    np.savetxt(os.path.join(pasta_res, "acerto_por_classe.csv"), linhas, fmt="%s", delimiter=";",
               encoding="utf-8", header="pessoa;" + ";".join(nomes_modelos), comments="")

    fig, ax = plt.subplots(figsize=(10, 2.9))
    sns.heatmap(taxas, annot=True, fmt=".2f", cmap="RdYlGn", vmin=min(0.8, taxas.min()), vmax=1.0,
                xticklabels=nomes_pessoas, yticklabels=nomes_modelos, ax=ax, annot_kws={"size": 8},
                cbar_kws={"label": "taxa de acerto"})
    ax.tick_params(axis="x", labelrotation=60, labelsize=9)
    ax.set_title(f"Taxa de acerto por pessoa (soma das {R} rodadas)")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, "acerto_por_classe.png"), dpi=200)
    plt.close(fig)


def tabelas(res, R, pasta_fig, pasta_res):
    nomes = list(res.keys())
    valores = [res[n]["acuracias"] for n in nomes]
    U.salvar_tabela_resumo(os.path.join(pasta_res, "tabela_acuracia"), nomes, valores,
                           f"Acurácia - Monte Carlo com R = {R} rodadas (80% treino / 20% teste)")
    U.plot_box_violin(valores, nomes, f"Acurácia nas {R} rodadas", "Acurácia",
                      os.path.join(pasta_fig, "boxplot_acuracia.png"))
    tabela = np.column_stack([np.arange(1, R + 1)] + valores)
    np.savetxt(os.path.join(pasta_res, "acuracia_por_rodada.csv"), tabela, delimiter=";", encoding="utf-8",
               fmt=["%d"] + ["%.6f"] * len(nomes), comments="",
               header="rodada;" + ";".join(nomes))
    linhas = [[n, f"{res[n]['epocas'].mean():.1f}", f"{res[n]['epocas'].min()}", f"{res[n]['epocas'].max()}",
               f"{res[n]['tempo'] / R:.3f}"] for n in nomes]
    np.savetxt(os.path.join(pasta_res, "custo_treinamento.csv"), np.array(linhas, dtype=object), fmt="%s",
               delimiter=";", comments="", encoding="utf-8",
               header="modelo;epocas_media;epocas_min;epocas_max;tempo_medio_s")


def comparar_configuracoes():
    """Compara R = 10 e R = 100 a partir dos CSVs gravados (se ambos existirem)."""
    arquivos = {R: os.path.join(RAIZ, "resultados", f"etapa2_R{R}", "acuracia_por_rodada.csv")
                for R in (10, 100)}
    if not all(os.path.exists(a) for a in arquivos.values()):
        return
    with open(arquivos[10], encoding="utf-8") as f:
        nomes = f.readline().strip().split(";")[1:]
    dados = {R: np.loadtxt(a, delimiter=";", skiprows=1)[:, 1:] for R, a in arquivos.items()}

    linhas = []
    for j, nome in enumerate(nomes):
        for R in (10, 100):
            media, desvio, maior, menor = U.resumo_estatistico(dados[R][:, j])
            # Erro-padrão da média: desvio / sqrt(R) -> mede a incerteza da estimativa da média.
            linhas.append([nome, str(R), f"{media:.4f}", f"{desvio:.4f}", f"{maior:.4f}", f"{menor:.4f}",
                           f"{desvio / np.sqrt(R):.4f}"])
    np.savetxt(os.path.join(PASTA_RES_COMUM, "comparacao_R10_R100.csv"), np.array(linhas, dtype=object),
               fmt="%s", delimiter=";", encoding="utf-8", comments="",
               header="modelo;R;media;desvio;maior;menor;erro_padrao_media")

    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.6))
    valores, grupos, configs = [], [], []
    for j, nome in enumerate(nomes):
        for R in (10, 100):
            valores.append(dados[R][:, j])
            grupos += [nome] * len(dados[R])
            configs += [f"R = {R}"] * len(dados[R])
    valores = np.concatenate(valores)
    sns.boxplot(x=grupos, y=valores, hue=configs, ax=eixos[0], palette="Set2")
    sns.violinplot(x=grupos, y=valores, hue=configs, ax=eixos[1], palette="Set2", split=True,
                   cut=0, inner="quartile")
    for ax, titulo in zip(eixos, ("Boxplot", "Violin plot")):
        ax.set_title(titulo)
        ax.set_ylabel("Acurácia")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Acurácia na Etapa 2: R = 10 x R = 100")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG_COMUM, "comparacao_R10_R100.png"), dpi=200)
    plt.close(fig)

    # Média acumulada: como a estimativa da média evolui com o número de rodadas.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    rodadas = np.arange(1, len(dados[100]) + 1)
    for j, nome in enumerate(nomes):
        ax.plot(rodadas, np.cumsum(dados[100][:, j]) / rodadas, label=nome)
    ax.axvline(10, color="gray", ls=":", lw=1)
    ax.text(10.5, ax.get_ylim()[0], " R = 10", color="gray", va="bottom")
    ax.set_xlabel("Número de rodadas")
    ax.set_ylabel("Acurácia média acumulada")
    ax.set_title("Estabilização da média com o aumento de R")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_FIG_COMUM, "media_acumulada.png"), dpi=200)
    plt.close(fig)
    print("  Comparação R = 10 x R = 100 gravada em", PASTA_RES_COMUM)


# =============================================================================
# Programa principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Etapa 2 - reconhecimento facial")
    parser.add_argument("--R", type=int, nargs="+", default=[10, 100],
                        help="número(s) de rodadas do Monte Carlo (padrão: 10 100)")
    parser.add_argument("--sem-ajuste", action="store_true",
                        help="pula a busca de hiperparâmetros e usa os valores padrão")
    args = parser.parse_args()
    U.configurar_terminal()
    U.garantir_pasta(PASTA_FIG_COMUM)
    U.garantir_pasta(PASTA_RES_COMUM)
    inicio = time.time()

    print(f"[1/4] Leitura das imagens ({LADO}x{LADO}, escala de cinza)")
    X_bruto, classes, nomes_pessoas = secao_dados()

    print("[2/4] Escolha de hiperparâmetros")
    if args.sem_ajuste:
        hps = (dict(HP_PERCEPTRON), dict(HP_ADALINE), dict(HP_MLP))
        print("  (busca ignorada: usando valores padrão)")
    else:
        hps = busca_hiperparametros(X_bruto, classes)
    for nome, hp in zip(("Perceptron Simples", "ADALINE", "MLP"), hps):
        print(f"  -> {nome}: {descrever_hp(hp)}")
    with open(os.path.join(PASTA_RES_COMUM, "hiperparametros_escolhidos.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(("perceptron", "adaline", "mlp"), hps)), f, indent=2, ensure_ascii=False)

    for R in args.R:
        print(f"[3/4] Monte Carlo com R = {R}")
        pasta_fig = U.garantir_pasta(os.path.join(RAIZ, "figuras", f"etapa2_R{R}"))
        pasta_res = U.garantir_pasta(os.path.join(RAIZ, "resultados", f"etapa2_R{R}"))
        res = monte_carlo(X_bruto, classes, hps, R)
        tabelas(res, R, pasta_fig, pasta_res)
        figuras_extremos(res, nomes_pessoas, R, pasta_fig, pasta_res)
        analise_por_classe(res, nomes_pessoas, R, pasta_fig, pasta_res)

    print("[4/4] Comparação entre as configurações de R")
    comparar_configuracoes()
    print(f"Concluído em {time.time() - inicio:.1f}s.")


if __name__ == "__main__":
    main()
