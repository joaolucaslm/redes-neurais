"""
gerar_relatorio.py
------------------
Monta o relatório (Relatorio_AV2.docx) a partir do modelo IEEE do repositório,
inserindo as figuras e os números REAIS gravados por etapa1.py e etapa2.py
em figuras/ e resultados/.

Este script é apenas uma ferramenta de apoio para escrever o relatório. Ele
usa python-docx e fica fora do código avaliado (modelos e experimentos), que
utiliza somente numpy, matplotlib, seaborn e opencv-python.

Uso (depois de executar etapa1.py e etapa2.py):
    pip install -r relatorio/requirements.txt
    python relatorio/gerar_relatorio.py
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx_ieee import RelatorioIEEE, Eq  # noqa: E402
from conteudo_resultados import escrever_resultados, escrever_conclusoes, resumo_texto  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO = os.path.join(RAIZ, "conference-template-a4.docx")
SAIDA = os.path.join(RAIZ, "relatorio", "Relatorio_AV2.docx")
r, t, b = Eq.r, Eq.t, Eq.b


# =============================================================================
# Leitura dos resultados
# =============================================================================
def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.reader(f, delimiter=";"))
    return linhas[0], linhas[1:]


def ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def carregar_dados():
    e1 = os.path.join(RAIZ, "resultados", "etapa1")
    e2 = os.path.join(RAIZ, "resultados", "etapa2")
    d = {"raiz": RAIZ}
    d["hp1"] = ler_json(os.path.join(e1, "hiperparametros_escolhidos.json"))
    d["tab1"] = {m: ler_csv(os.path.join(e1, f"tabela_{m}.csv"))[1]
                 for m in ("acuracia", "sensibilidade", "especificidade")}
    d["topologias"] = ler_csv(os.path.join(e1, "topologias_mlp.csv"))[1]
    d["topologias_part"] = ler_csv(os.path.join(e1, "topologias_mlp_por_particao.csv"))[1]
    d["ajuste_p1"] = ler_csv(os.path.join(e1, "ajuste_perceptron.csv"))[1]
    d["ajuste_a1"] = ler_csv(os.path.join(e1, "ajuste_adaline.csv"))[1]
    d["ajuste_m1"] = ler_csv(os.path.join(e1, "ajuste_mlp.csv"))[1]
    d["extremos1"] = ler_csv(os.path.join(e1, "rodadas_extremas.csv"))[1]
    d["custo1"] = ler_csv(os.path.join(e1, "custo_treinamento.csv"))[1]
    d["rodadas1"] = np.loadtxt(os.path.join(e1, "metricas_por_rodada.csv"), delimiter=";", skiprows=1)

    d["hp2"] = ler_json(os.path.join(e2, "hiperparametros_escolhidos.json"))
    d["ajuste_a2"] = ler_csv(os.path.join(e2, "ajuste_adaline.csv"))[1]
    d["ajuste_m2"] = ler_csv(os.path.join(e2, "ajuste_mlp.csv"))[1]
    d["comparacao"] = ler_csv(os.path.join(e2, "comparacao_R10_R100.csv"))[1]
    with open(os.path.join(e2, "estabilidade_adaline.txt"), encoding="utf-8") as f:
        d["lambda_max"] = float(f.readline().split("=")[1])
    for R in (10, 100):
        pasta = os.path.join(RAIZ, "resultados", f"etapa2_R{R}")
        d[f"tab2_{R}"] = ler_csv(os.path.join(pasta, "tabela_acuracia.csv"))[1]
        d[f"extremos2_{R}"] = ler_csv(os.path.join(pasta, "rodadas_extremas.csv"))[1]
        d[f"classes_{R}"] = ler_csv(os.path.join(pasta, "acerto_por_classe.csv"))
        d[f"custo2_{R}"] = ler_csv(os.path.join(pasta, "custo_treinamento.csv"))[1]
        d[f"rodadas2_{R}"] = np.loadtxt(os.path.join(pasta, "acuracia_por_rodada.csv"),
                                        delimiter=";", skiprows=1)
        d[f"acumuladas_{R}"] = {}
        for nome, arquivo in (("Perceptron Simples", "perceptron_simples"), ("ADALINE", "adaline"),
                              ("MLP", "mlp")):
            cab, linhas = ler_csv(os.path.join(pasta, f"matriz_acumulada_{arquivo}.csv"))
            d[f"acumuladas_{R}"][nome] = np.array(linhas, dtype=int)
        d["pessoas"] = cab
    return d


def potencia(x):
    """Escreve 0,001 como 10^{−3} (marcação de sobrescrito do docx_ieee)."""
    k = np.log10(x)
    if abs(k - round(k)) < 1e-9 and k < -1:
        return f"10^{{−{-int(round(k))}}}"
    return f"{x:g}".replace(".", ",")


def num(x, casas=4):
    return f"{float(x):.{casas}f}".replace(".", ",")


def pct(x, casas=2):
    return f"{100 * float(x):.{casas}f}".replace(".", ",") + "%"


def fig(*partes):
    return os.path.join(RAIZ, "figuras", *partes)


# =============================================================================
# Metodologia
# =============================================================================
def escrever_metodologia(rel, d):
    hp1, hp2 = d["hp1"], d["hp2"]
    ocultas1 = "-".join(str(q) for q in hp1["mlp"]["camadas_ocultas"])
    ocultas2 = "-".join(str(q) for q in hp2["mlp"]["camadas_ocultas"])

    rel.secao("II", "Metodologia")
    rel.subsecao("A", "Ambiente de desenvolvimento e reprodutibilidade")
    rel.paragrafo(
        "As implementações foram escritas em Python 3.12 usando somente as bibliotecas permitidas: "
        "numpy (álgebra linear, leitura do CSV, sorteios e estatísticas), matplotlib e seaborn "
        "(gráficos; o seaborn foi usado apenas para desenhar heatmaps e violin plots) e opencv-python "
        "(apenas leitura e redimensionamento das imagens). A separação treino/teste, a normalização, a "
        "codificação one-hot, a matriz de confusão e as métricas foram implementadas pela equipe. O código "
        "está dividido em *modelos.py* (classes Perceptron Simples, ADALINE e MLP, com os métodos *fit* e "
        "*predict* e o histórico do EQM por época), *utils.py* (funções de apoio) e *etapa1.py*/*etapa2.py* "
        "(experimentos). Todos os sorteios usam um gerador do numpy com semente fixa (42), o que torna os "
        "resultados reprodutíveis.")

    # ------------------------------------------------------------------ dados
    rel.subsecao("B", "Organização e pré-processamento dos dados")
    rel.paragrafo(
        "O arquivo spiral_d.csv possui N = 1400 amostras com p = 2 atributos (colunas 1 e 2) e o rótulo na "
        "coluna 3. A inspeção dos rótulos mostrou que eles já estão no formato bipolar {−1, +1}, com 1000 "
        "amostras da classe +1 e 400 da classe −1 (71,4% e 28,6%; classes desbalanceadas). A classe +1 foi "
        "adotada como positiva no cálculo da sensibilidade e da especificidade. Seguindo a notação dos "
        "slides da disciplina [7], cada amostra é uma coluna de X e a primeira linha é a entrada de bias, "
        "constante e igual a −1:", titulo_em_linha="1) Etapa 1")
    rel.equacao(b("X") + t("=") + Eq.par(Eq.sub(b("x"), t("1")) + t("  ") + Eq.sub(b("x"), t("2")) + t(" ⋯ ")
                                         + Eq.sub(b("x"), r("N")), "[", "]")
                + t("∈") + Eq.sup(t("ℝ"), Eq.par(r("p") + t("+1")) + t("×") + r("N")) + t(",   ")
                + b("x") + Eq.par(r("t")) + t("=") + Eq.sup(Eq.par(t("−1,  ") + Eq.sub(r("x"), t("1"))
                                                                   + Eq.par(r("t")) + t(",  ")
                                                                   + Eq.sub(r("x"), t("2")) + Eq.par(r("t")),
                                                                   "[", "]"), r("T")))
    rel.paragrafo(
        "Assim, na Etapa 1, X tem dimensão 3 × 1400 e o vetor de rótulos d tem 1400 posições.")
    rel.paragrafo(
        "A pasta RecFac contém 20 subpastas, uma por pessoa, com 32 imagens PNG de 120 × 128 pixels cada "
        "(640 imagens). Cada pessoa aparece em quatro poses (esquerda, direita, frente e para cima), quatro "
        "expressões e com ou sem óculos escuros. No repositório a pasta está aninhada "
        "(RecFac/RecFac/<pessoa>), e o código desce automaticamente até o nível que contém as pessoas. O nome "
        "de cada subpasta é usado como rótulo. As imagens foram lidas em escala de cinza (cv2.imread) e "
        "redimensionadas para 30 × 30 pixels com interpolação por área (cv2.INTER_AREA), que calcula a média "
        "dos pixels de cada região e evita serrilhado. Entre as opções do enunciado (30 × 30, 40 × 40, "
        "50 × 50 e 80 × 80), escolheu-se a menor: ela ainda preserva o contorno do rosto, o cabelo e a "
        "presença dos óculos (Fig. 1) e reduz o custo do treinamento, que é repetido 110 vezes (R = 10 e "
        "R = 100) para cada modelo. Cada imagem foi vetorizada em 900 valores, de modo que X tem dimensão "
        "901 × 640. Os rótulos foram codificados em one-hot bipolar: a pessoa k recebe um vetor com +1 na "
        "posição k e −1 nas outras 19 posições; com as pessoas em ordem alfabética (an2i, at33, boland, bpm, "
        "ch4f, …), a mesma do enunciado, Y tem dimensão 20 × 640:", titulo_em_linha="2) Etapa 2")
    rel.equacao(Eq.sub(b("y"), t("an2i")) + t("=") + Eq.sup(Eq.par(t("+1, −1, −1, ⋯, −1"), "[", "]"), r("T"))
                + t(",   ") + b("Y") + t("∈") + Eq.sup(t("ℝ"), r("C") + t("×") + r("N")) + t(",  ")
                + r("C") + t("=20"))
    rel.figura_larga(fig("etapa2", "01_exemplos_faces.png"),
                     "Uma imagem de cada uma das 20 pessoas após a conversão para escala de cinza e o "
                     "redimensionamento para 30 × 30 pixels.")
    rel.paragrafo(
        "Antes do treinamento, cada atributo foi normalizado para o intervalo [−1, 1] por min-max:",
        titulo_em_linha="3) Normalização")
    rel.equacao(Eq.sup(r("x"), t("′")) + t("=2 ") + Eq.frac(r("x") + t("−") + Eq.sub(r("x"), t("mín")),
                                                            Eq.sub(r("x"), t("máx")) + t("−")
                                                            + Eq.sub(r("x"), t("mín"))) + t("−1"))
    rel.paragrafo(
        "Esse intervalo foi escolhido por coincidir com a faixa da tangente hiperbólica, com os rótulos "
        "bipolares e com o valor da entrada de bias, deixando todas as entradas na mesma escala. O mínimo "
        "e o máximo de cada atributo são calculados somente com as amostras de treino de cada rodada e "
        "depois aplicados ao teste, para que nenhuma informação do teste participe do treinamento. Pixels "
        "constantes no treino recebem o valor 0, evitando divisão por zero.")

    # ---------------------------------------------------------------- modelos
    rel.subsecao("C", "Modelos de redes neurais")
    rel.paragrafo(
        "Os três modelos recebem X ∈ ℝ^{(p+1)×N} e a matriz de rótulos D ∈ ℝ^{C×N}, com C = 1 na Etapa 1 e "
        "C = 20 na Etapa 2. Em todos eles o treinamento é acompanhado pelo erro quadrático médio (EQM) de "
        "cada época,")
    rel.equacao(t("EQM=") + Eq.frac(t("1"), t("2") + r("N")) + Eq.soma(r("t") + t("=1"), r("N"),
                Eq.soma(r("j") + t("=1"), r("C"), Eq.sup(Eq.par(Eq.sub(r("d"), r("j")) + Eq.par(r("t")) + t("−")
                                                               + Eq.sub(r("y"), r("j")) + Eq.par(r("t"))), t("2")))))
    rel.paragrafo(
        "que também é calculado no conjunto de teste, apenas para desenhar a curva de aprendizado. O teste "
        "nunca influencia o ajuste dos pesos nem o critério de parada. Todos os modelos param, no máximo, "
        "ao atingir um número máximo de épocas.")

    rel.paragrafo(
        "Para cada amostra calcula-se o potencial de ativação e a saída pela função degrau bipolar [2]:",
        titulo_em_linha="1) Perceptron Simples")
    rel.equacao(r("u") + Eq.par(r("t")) + t("=") + Eq.sup(b("w"), r("T")) + b("x") + Eq.par(r("t")) + t(",   ")
                + r("y") + Eq.par(r("t")) + t("=") + Eq.casos(t("+1,   se ") + r("u") + Eq.par(r("t")) + t("≥0"),
                                                             t("−1,   se ") + r("u") + Eq.par(r("t")) + t("<0")))
    rel.paragrafo("Quando a saída difere do rótulo desejado d(t), os pesos são corrigidos pela regra de "
                  "aprendizado do Perceptron:")
    rel.equacao(b("w") + t("←") + b("w") + t("+") + r("η") + Eq.par(r("d") + Eq.par(r("t")) + t("−") + r("y")
                                                                 + Eq.par(r("t"))) + b("x") + Eq.par(r("t")))
    rel.paragrafo(
        "Os pesos partem de zero, como nos slides. O algoritmo clássico repete as épocas até que uma época "
        "inteira ocorra sem erros, o que nunca acontece com dados não linearmente separáveis, como a "
        "espiral. Por isso, conforme exige o enunciado, foi acrescentado o critério de parada por número "
        "máximo de épocas. Uma consequência da inicialização nula é que a fronteira obtida não depende de "
        "η: todas as correções são múltiplos de η, então w é sempre o mesmo vetor multiplicado por η e o "
        "sinal de wᵀx não muda. Na Etapa 2 são usados 20 perceptrons independentes (W ∈ ℝ^{901×20}), cada "
        "um treinado para responder +1 apenas para a sua pessoa. O EQM do Perceptron é calculado sobre a "
        "saída do degrau; cada amostra classificada errada contribui com 2/N.")

    rel.paragrafo(
        "O ADALINE [3] tem a mesma estrutura, mas o erro é medido na saída linear u(t), e os pesos são "
        "ajustados pela regra delta (LMS), que é um passo de gradiente descendente sobre o EQM:",
        titulo_em_linha="2) ADALINE")
    rel.equacao(b("w") + t("←") + b("w") + t("+") + r("η") + Eq.par(r("d") + Eq.par(r("t")) + t("−") + r("u")
                                                                 + Eq.par(r("t"))) + b("x") + Eq.par(r("t")))
    rel.paragrafo(
        "O treinamento é feito amostra a amostra, como nos slides, e termina quando a variação do EQM entre "
        "duas épocas consecutivas fica abaixo da precisão ε, ou ao atingir o número máximo de épocas:")
    rel.equacao(Eq.par(t("EQM") + Eq.sub(t(""), t("atual")) + t("−EQM") + Eq.sub(t(""), t("anterior")), "|", "|")
                + t("≤") + r("ε"))
    rel.paragrafo(
        "Depois de treinado, o ADALINE classifica pelo degrau bipolar aplicado a u (Etapa 1) ou pelo argmax "
        "de u (Etapa 2). A regra LMS só converge se a taxa de aprendizado respeitar")
    rel.equacao(t("0<") + r("η") + t("<") + Eq.frac(t("2"), Eq.sub(r("λ"), t("máx"))) + t(",   ") + b("R") + t("=")
                + Eq.frac(t("1"), r("N")) + b("X") + Eq.sup(b("X"), r("T")))
    rel.paragrafo(
        "em que λ_{máx} é o maior autovalor da matriz de correlação das entradas [1]. Esse limite foi usado "
        "para delimitar os valores de η investigados na Etapa 2.")

    rel.paragrafo(
        "A MLP usa a tangente hiperbólica em todos os neurônios (ocultos e de saída), cuja derivada pode ser "
        "escrita em função da própria saída: g′(i) = 1 − g(i)². Sendo W^{(l)} a matriz de pesos da camada l, "
        "com q_{l} linhas e q_{l−1} + 1 colunas (a primeira associada ao bias), a fase *forward* calcula, "
        "para l = 1, …, L,", titulo_em_linha="3) Perceptron de Múltiplas Camadas")
    rel.equacao(Eq.sup(b("i"), Eq.par(r("l"))) + t("=") + Eq.sup(b("W"), Eq.par(r("l")))
                + Eq.sup(b("y"), Eq.par(r("l") + t("−1"))) + t(",   ")
                + Eq.sup(b("y"), Eq.par(r("l"))) + t("=tanh") + Eq.par(Eq.sup(b("i"), Eq.par(r("l")))))
    rel.paragrafo(
        "com y^{(0)} = X e uma linha de −1 acrescentada à saída de cada camada oculta (bias da camada "
        "seguinte). A fase *backward* (retropropagação [4]) calcula os gradientes locais da saída para a "
        "entrada,")
    rel.equacao(Eq.sup(b("δ"), Eq.par(r("L"))) + t("=") + Eq.par(t("1−") + Eq.sup(b("y"), Eq.par(r("L"))) + t("⊙")
                                                              + Eq.sup(b("y"), Eq.par(r("L"))))
                + t("⊙") + Eq.par(b("D") + t("−") + Eq.sup(b("y"), Eq.par(r("L")))))
    rel.equacao(Eq.sup(b("δ"), Eq.par(r("l"))) + t("=") + Eq.par(t("1−") + Eq.sup(b("y"), Eq.par(r("l"))) + t("⊙")
                                                              + Eq.sup(b("y"), Eq.par(r("l"))))
                + t("⊙") + Eq.par(Eq.sup(Eq.acento(b("W")), Eq.par(r("l") + t("+1")) + r("T"))
                                  + Eq.sup(b("δ"), Eq.par(r("l") + t("+1")))))
    rel.paragrafo(
        "em que ⊙ é o produto elemento a elemento e W̃ é a matriz de pesos sem a coluna do bias. Por fim, os "
        "pesos são atualizados por gradiente descendente:")
    rel.equacao(Eq.sup(b("W"), Eq.par(r("l"))) + t("←") + Eq.sup(b("W"), Eq.par(r("l"))) + t("+")
                + Eq.frac(r("η"), r("N")) + Eq.sup(b("δ"), Eq.par(r("l")))
                + Eq.sup(b("y"), Eq.par(r("l") + t("−1")) + r("T")))
    rel.paragrafo(
        "Os slides apresentam o *backpropagation* amostra a amostra. Aqui as N amostras de treino são "
        "propagadas juntas, como colunas de uma matriz, e o gradiente usado é a média dos gradientes "
        "individuais (gradiente descendente em lote). A regra de atualização é a mesma, apenas acumulada "
        "ao longo da época, e corresponde ao gradiente exato do EQM de treino. Essa adaptação permite usar "
        "operações matriciais do numpy e tornou viável repetir o treinamento 500 vezes na Etapa 1. A MLP para "
        "quando o EQM de treino fica abaixo da precisão ε, ou ao atingir o número máximo de épocas.")
    rel.paragrafo(
        "Os pesos do ADALINE e da MLP são sorteados de uma distribuição uniforme")
    rel.equacao(r("w") + t("~") + r("U") + Eq.par(t("−") + Eq.frac(t("1"), Eq.raiz(Eq.sub(r("n"), r("e")))) + t(",  ")
                                                  + Eq.frac(t("1"), Eq.raiz(Eq.sub(r("n"), r("e"))))))
    rel.paragrafo(
        "em que n_{e} é o número de entradas do neurônio, incluindo o bias. Os slides sugerem valores "
        "pequenos, tipicamente em [−0,5; 0,5]. Para a espiral (n_{e} = 3) o limite é 0,577, praticamente a "
        "faixa clássica; para as imagens (n_{e} = 901) ele cai para 0,033. Isso impede que o potencial de "
        "ativação inicial seja grande demais e sature a tangente hiperbólica, situação em que g′(i) ≈ 0 e o "
        "aprendizado praticamente para [5].")
    rel.paragrafo(
        "Na Etapa 1, a classe prevista é o sinal da saída (degrau bipolar). Na Etapa 2, é o índice do "
        "neurônio de saída com maior valor:", titulo_em_linha="4) Regra de decisão")
    rel.equacao(Eq.acento(r("c"), "̂") + Eq.par(r("t")) + t("=") + Eq.sub(t("arg max"), r("j") + t("=1,…,") + r("C"))
                + t(" ") + Eq.sub(r("y"), r("j")) + Eq.par(r("t")))
    rel.paragrafo(
        "No Perceptron, várias saídas do degrau podem valer +1 ao mesmo tempo (ou nenhuma). Por isso o "
        "argmax é aplicado ao potencial u, que é a saída do neurônio antes do degrau.")

    # ------------------------------------------------------- hiperparâmetros
    p1, a1, m1 = hp1["perceptron"], hp1["adaline"], hp1["mlp"]
    p2, a2, m2 = hp2["perceptron"], hp2["adaline"], hp2["mlp"]
    n_val = 5 * 280
    rel.subsecao("D", "Escolha dos hiperparâmetros")
    rel.paragrafo(
        "Os hiperparâmetros foram escolhidos por busca em grade. Cada configuração foi avaliada em "
        "partições treino/validação (80/20) sorteadas com uma semente diferente da usada na validação "
        "final (cinco partições na Etapa 1 e três na Etapa 2, cujo treinamento é mais caro) e comparada "
        "pela média da métrica de validação.")
    rel.paragrafo(
        "Para o Perceptron testou-se η ∈ {0,001; 0,01; 0,1; 1}. Como previsto pela inicialização nula, as "
        "quatro taxas resultaram exatamente na mesma acurácia, e η = 0,01 foi mantido. O limite de 100 épocas "
        "foi fixado observando que, na espiral, o EQM do Perceptron apenas oscila, sem tendência de queda. "
        "Para o ADALINE testou-se η ∈ {10^{−4}; 5·10^{−4}; 10^{−3}; 5·10^{−3}; 10^{−2}}, com ε = 10^{−6} e no "
        "máximo 500 épocas, escolhendo o η de menor EQM de validação, que é a função que o ADALINE "
        "minimiza. Para a MLP foram testadas seis topologias (uma camada oculta com 5, 10 ou 20 neurônios; "
        "duas camadas com 5-5, 10-10 ou 20-20) e três taxas (η ∈ {0,5; 1; 2}), com no máximo 2000 épocas e "
        f"ε = 10^{{−3}}. Adotou-se um critério de parcimônia: entre as configurações cuja acurácia média de "
        f"validação fica a até 0,25 ponto percentual da melhor (cerca de 3 das {n_val} amostras de "
        "validação), escolhe-se a de menor número de pesos, o que reduz o custo e o risco de overfitting. "
        "A precisão ε = 10^{−3} corresponde a um erro RMS por saída de cerca de 0,045 (√(2·10^{−3})), ou seja, "
        "saídas da tanh já muito próximas de ±1. Continuar o treinamento a partir daí apenas aumentaria a "
        "magnitude dos pesos, sem ganho de classificação.", titulo_em_linha="1) Etapa 1")
    rel.paragrafo(
        "Na Etapa 2 foi seguida a mesma lógica, aproveitando a experiência da Etapa 1. (i) O Perceptron foi "
        "mantido com η = 0,01 e 100 épocas; como p = 900 é maior que o número de amostras de treino "
        "(N = 512), os dados de treino são linearmente separáveis e ele converge em poucas épocas. (ii) Para "
        f"o ADALINE, o maior autovalor de R é λ_{{máx}} = {num(d['lambda_max'], 1)}, o que exige "
        f"η < {num(2 / d['lambda_max'], 4)}. Foram testados η ∈ {{5·10^{{−4}}; 10^{{−3}}; 2·10^{{−3}}}} com "
        "ε = 10^{−3} e η = 10^{−3} com ε = 10^{−4}, escolhendo a maior acurácia de validação e, em caso de "
        "empate (até 0,25 p.p.), a mais rápida. (iii) Para a MLP, a taxa de aprendizado precisou ser bem "
        "menor que na espiral. Em testes preliminares, η ≥ 0,5 levou as 20 saídas à saturação em −1 "
        "(EQM = 2), porque 19 dos 20 alvos de cada amostra valem −1 e o gradiente empurra rapidamente todas "
        "as saídas para esse valor. Foram então testadas as topologias 20, 50, 100 e 50-50 com "
        "η ∈ {0,05; 0,1}, no máximo 1500 épocas e ε = 10^{−3}, aplicando o mesmo critério de parcimônia. "
        "A Tabela I resume os valores finais.", titulo_em_linha="2) Etapa 2")

    def ep(h):
        return str(h["max_epocas"])

    rel.tabela("Hiperparâmetros escolhidos", ["Modelo / etapa", "Neurônios ocultos", "η", "Épocas máx.", "ε"], [
        ["Perceptron – E1", "—", num(p1["taxa_aprendizado"], 2), ep(p1), "—"],
        ["ADALINE – E1", "—", f"{a1['taxa_aprendizado']:g}".replace(".", ","), ep(a1),
         potencia(a1['precisao'])],
        ["MLP – E1", ocultas1, f"{m1['taxa_aprendizado']:g}".replace(".", ","), ep(m1),
         potencia(m1['precisao'])],
        ["Perceptron – E2", "—", num(p2["taxa_aprendizado"], 2), ep(p2), "—"],
        ["ADALINE – E2", "—", f"{a2['taxa_aprendizado']:g}".replace(".", ","), ep(a2),
         potencia(a2['precisao'])],
        ["MLP – E2", ocultas2, f"{m2['taxa_aprendizado']:g}".replace(".", ","), ep(m2),
         potencia(m2['precisao'])],
    ], larguras=[1500, 1000, 700, 850, 760])
    rel.paragrafo(
        "Em todas as MLPs a camada de saída tem a quantidade de neurônios igual ao número de classes "
        "codificadas: 1 neurônio na Etapa 1 (saída bipolar) e 20 neurônios na Etapa 2 (one-hot).")

    # ------------------------------------------------------ under/overfitting
    topo = {linha[0]: linha for linha in d["topologias"]}
    sub, sup = topo["Subdimensionada"], topo["Superdimensionada"]
    rel.subsecao("E", "Estudo de underfitting e overfitting")
    rel.paragrafo(
        "Para identificar underfitting e overfitting, a MLP foi treinada com três topologias nas mesmas "
        "cinco partições 80/20 (uma única partição poderia ser enganosa, por exemplo, por uma inicialização "
        "ruim): "
        f"(i) subdimensionada, {sub[1]} (uma camada oculta com 2 neurônios e {sub[2]} pesos), "
        f"η = {sub[3]} e até 2000 épocas; (ii) adequada, a topologia {topo['Adequada'][1]} escolhida na busca "
        f"({topo['Adequada'][2]} pesos); (iii) superdimensionada, {sup[1]} (três camadas ocultas com 60 "
        f"neurônios e {sup[2]} pesos), η = {sup[3]}, 15000 épocas e ε = 0, isto é, sem parada antecipada, "
        "forçando a rede a continuar ajustando o treino. Para cada caso foram calculadas, no teste, a "
        "acurácia, a sensibilidade e a especificidade (média e desvio-padrão das cinco partições) e, para a "
        "primeira partição, a matriz de confusão, a curva de aprendizado (EQM de treino e de teste por "
        "época) e a fronteira de decisão.")

    # -------------------------------------------------------------- validação
    rel.subsecao("F", "Estratégia de validação e métricas")
    rel.paragrafo(
        "Foi usada validação por amostragem aleatória com R = 500 rodadas. Em cada rodada os índices das "
        "1400 amostras são embaralhados; as primeiras 1120 (80%) formam o treino e as 280 restantes (20%) o "
        "teste. A normalização é recalculada e os três modelos são treinados do zero com os hiperparâmetros "
        "da Tabela I. No teste foram medidas a acurácia, a sensibilidade e a especificidade:",
        titulo_em_linha="1) Etapa 1")
    rel.equacao(t("Acc=") + Eq.frac(t("VP+VN"), t("VP+VN+FP+FN")) + t(",  Sens=") + Eq.frac(t("VP"), t("VP+FN"))
                + t(",  Esp=") + Eq.frac(t("VN"), t("VN+FP")))
    rel.paragrafo(
        "VP, VN, FP e FN são os verdadeiros/falsos positivos/negativos, obtidos da matriz de confusão. A "
        "matriz é calculada pela equipe, somando 1 na célula (classe real, classe prevista) de cada amostra; "
        "as linhas correspondem à classe real e as colunas à prevista, e o seaborn é usado apenas para "
        "desenhar o heatmap. Para cada modelo e métrica foram identificadas a rodada de maior e a de menor "
        "valor (em caso de empate, a primeira rodada em que o valor ocorre), e para elas foram plotadas a "
        "matriz de confusão e a curva de aprendizado. Ao final das R rodadas calcularam-se a média, o "
        "desvio-padrão amostral, o maior e o menor valor de cada métrica,")
    rel.equacao(Eq.barra(r("a")) + t("=") + Eq.frac(t("1"), r("R")) + Eq.soma(r("k") + t("=1"), r("R"), Eq.sub(r("a"), r("k")))
                + t(",   ") + r("s") + t("=") + Eq.raiz(Eq.frac(t("1"), r("R") + t("−1"))
                                                    + Eq.soma(r("k") + t("=1"), r("R"),
                                                              Eq.sup(Eq.par(Eq.sub(r("a"), r("k")) + t("−")
                                                                            + Eq.barra(r("a"))), t("2")))))
    rel.paragrafo("apresentados em uma tabela por métrica e complementados por boxplots e violin plots.")
    rel.paragrafo(
        "Foi usada validação por simulações de Monte Carlo com a mesma partição 80/20 (512 imagens de "
        "treino e 128 de teste por rodada), tendo a acurácia como métrica. O enunciado apresenta uma "
        "divergência quanto ao número de rodadas: o item 5 define R = 10, enquanto o item 7 pede as "
        "estatísticas \"ao final das R = 100 rodadas\". Como não é possível saber qual valor era o "
        "pretendido, a equipe optou por executar e reportar as duas configurações. O script recebe R como "
        "parâmetro e grava as saídas em pastas separadas (etapa2_R10 e etapa2_R100). Como a semente é a "
        "mesma, as 10 rodadas da configuração R = 10 coincidem com as 10 primeiras da configuração R = 100, "
        "e a comparação entre as duas mostra exatamente o efeito de aumentar o número de rodadas sobre as "
        "estimativas. Para cada modelo e configuração foram geradas as matrizes de confusão 20 × 20 e as "
        "curvas de aprendizado das rodadas de maior e de menor acurácia. Para identificar as pessoas mais e "
        "menos reconhecidas, as matrizes de todas as rodadas foram somadas e calculou-se a taxa de acerto de "
        "cada pessoa (diagonal dividida pelo total da linha).", titulo_em_linha="2) Etapa 2")

    # ------------------------------------------------------------- adaptações
    rel.subsecao("G", "Resumo das adaptações em relação à formulação dos slides")
    for texto in (
        "**Critério de parada por épocas no Perceptron**: exigido pelo enunciado, pois o algoritmo original "
        "não termina em dados não linearmente separáveis.",
        "**Gradiente descendente em lote na MLP**: a mesma regra do backpropagation, calculada de uma vez "
        "para todas as amostras (média dos gradientes); permite vetorizar e executar 500 rodadas em tempo "
        "razoável.",
        "**Pesos iniciais uniformes e pequenos** no ADALINE e na MLP, em U(−1/√n_{e}, 1/√n_{e}): preserva a "
        "escala clássica "
        "(±0,5) na espiral e evita a saturação da tanh com 900 entradas.",
        "**Normalização min-max em [−1, 1] calculada só com o treino** de cada rodada: mesma faixa da tanh, "
        "dos rótulos e do bias; sem vazamento de informação do teste.",
        "**Argmax do potencial u no Perceptron multiclasse**: desempata os casos em que várias saídas do "
        "degrau valem +1 (ou nenhuma).",
        "**ε = 0 e 15000 épocas na topologia superdimensionada**: desliga a parada antecipada para tornar o "
        "overfitting observável.",
        "**BLAS com uma única thread**: apenas desempenho; com matrizes pequenas, uma thread foi mais rápida "
        "e garante resultados idênticos a cada execução.",
    ):
        rel.item(texto)


# =============================================================================
# Programa principal
# =============================================================================
def main():
    d = carregar_dados()
    rel = RelatorioIEEE(MODELO)
    rel.titulo("Redes Neurais Artificiais Lineares e Não Lineares na Classificação de uma Espiral e no "
               "Reconhecimento Facial")
    rel.autores([
        ("João Lucas Lima Maia", "Ciência da Computação", "Universidade de Fortaleza (UNIFOR)",
         "Fortaleza, Brasil", "joaolucaslimamaia@gmail.com"),
        ("Nome do(a) integrante 2", "Curso", "Universidade de Fortaleza (UNIFOR)", "Fortaleza, Brasil",
         "e-mail"),
        ("Nome do(a) integrante 3", "Curso", "Universidade de Fortaleza (UNIFOR)", "Fortaleza, Brasil",
         "e-mail"),
    ])
    rel.resumo(resumo_texto(d), "Perceptron Simples, ADALINE, perceptron de múltiplas camadas, "
                                "retropropagação, reconhecimento facial, validação Monte Carlo")
    rel.secao("I", "Introdução")
    rel.paragrafo(
        "Redes neurais artificiais são modelos inspirados no funcionamento do cérebro que aprendem a partir "
        "de exemplos [1]. Este trabalho compara três delas: o Perceptron Simples [2] e o ADALINE [3], que "
        "são modelos lineares (um único neurônio por saída, com fronteira de decisão linear), e o Perceptron "
        "de Múltiplas Camadas (MLP) treinado por retropropagação do erro [4], que é não linear. Na primeira "
        "etapa, os modelos são validados em um problema bidimensional sintético de duas classes com formato "
        "de espiral. Na segunda, são aplicados ao reconhecimento de 20 pessoas a partir de imagens faciais "
        "[6]. Em ambas as etapas o desempenho é estimado por validação com várias partições aleatórias "
        "treino/teste.")
    escrever_metodologia(rel, d)
    escrever_resultados(rel, d)
    escrever_conclusoes(rel, d)
    rel.titulo_referencias()
    for referencia in (
        "S. Haykin, Neural Networks and Learning Machines, 3rd ed. Upper Saddle River, NJ, USA: Pearson, 2009.",
        "F. Rosenblatt, “The perceptron: a probabilistic model for information storage and organization in "
        "the brain,” Psychological Review, vol. 65, no. 6, pp. 386–408, 1958.",
        "B. Widrow and M. E. Hoff, “Adaptive switching circuits,” in IRE WESCON Convention Record, part 4, "
        "pp. 96–104, 1960.",
        "D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating "
        "errors,” Nature, vol. 323, pp. 533–536, 1986.",
        "Y. LeCun, L. Bottou, G. B. Orr, and K.-R. Müller, “Efficient BackProp,” in Neural Networks: Tricks of "
        "the Trade, LNCS 1524. Berlin, Germany: Springer, 1998, pp. 9–50.",
        "T. M. Mitchell, Machine Learning. New York, NY, USA: McGraw-Hill, 1997.",
        "P. C. S. Barbosa, “Inteligência Artificial Computacional: slides de aula,” Universidade de Fortaleza "
        "(UNIFOR), Fortaleza, Brasil, 2026.",
    ):
        rel.referencia(referencia)
    rel.salvar(SAIDA)
    print("Relatório gravado em", SAIDA)


if __name__ == "__main__":
    main()
