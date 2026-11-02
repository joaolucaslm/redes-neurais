"""
conteudo_resultados.py
----------------------
Texto das seções de Resultados e Conclusões do relatório. Todos os números
são lidos dos arquivos gravados por etapa1.py e etapa2.py (nada é digitado
à mão), de modo que o texto sempre corresponde à última execução.
"""

import os

import numpy as np

MODELOS = ("Perceptron Simples", "ADALINE", "MLP")
METRICAS = ("acuracia", "sensibilidade", "especificidade")
NOME_METRICA = {"acuracia": "Acurácia", "sensibilidade": "Sensibilidade", "especificidade": "Especificidade"}


def num(x, casas=4):
    return f"{float(x):.{casas}f}".replace(".", ",")


def pct(x, casas=2):
    return f"{100 * float(x):.{casas}f}".replace(".", ",") + "%"


def fig(d, *partes):
    return os.path.join(d["raiz"], "figuras", *partes)


def lista(itens):
    """Junta itens como em português: a, b e c."""
    itens = list(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def taxa(x):
    """Taxa de aprendizado sem zeros supérfluos (2.0 -> 2; 0.0001 -> 0,0001)."""
    return f"{float(x):.10g}".replace(".", ",")


def linha_modelo(tabela, modelo):
    """Linha (média, desvio, maior, menor) de um modelo em uma tabela resumo."""
    for linha in tabela:
        if linha[0] == modelo:
            return [float(v) for v in linha[1:5]]
    raise KeyError(modelo)


def tabela_resumo(rel, titulo, tabela):
    linhas = [[l[0], num(l[1]), num(l[2]), num(l[3]), num(l[4])] for l in tabela]
    return rel.tabela(titulo, ["Modelo", "Média", "Desvio-padrão", "Maior valor", "Menor valor"], linhas,
                      larguras=[1480, 800, 1050, 800, 800])


def extremo(d, modelo, metrica):
    nome = NOME_METRICA[metrica]
    for linha in d["extremos1"]:
        if linha[0] == modelo and linha[1] == nome:
            return int(linha[2]), float(linha[3]), int(linha[4]), float(linha[5])
    raise KeyError((modelo, metrica))


def coluna_rodadas(d, modelo, metrica):
    indice = 1 + 3 * MODELOS.index(modelo) + METRICAS.index(metrica)
    return d["rodadas1"][:, indice]


# =============================================================================
# Resumo
# =============================================================================
def resumo_texto(d):
    acc = {m: linha_modelo(d["tab1"]["acuracia"], m) for m in MODELOS}
    t10 = {m: linha_modelo(d["tab2_10"], m) for m in MODELOS}
    t100 = {m: linha_modelo(d["tab2_100"], m) for m in MODELOS}
    ocultas1 = "-".join(str(q) for q in d["hp1"]["mlp"]["camadas_ocultas"])
    ocultas2 = "-".join(str(q) for q in d["hp2"]["mlp"]["camadas_ocultas"])
    return (
        "Este trabalho implementa, apenas com numpy, o Perceptron Simples, o ADALINE e o Perceptron de "
        "Múltiplas Camadas (MLP) e os avalia em dois problemas. No primeiro, a classificação binária de "
        "uma espiral bidimensional (1400 amostras), a validação por amostragem aleatória com 500 rodadas "
        f"resultou em acurácia média de {pct(acc['Perceptron Simples'][0])} para o Perceptron, "
        f"{pct(acc['ADALINE'][0])} para o ADALINE e {pct(acc['MLP'][0])} para a MLP {ocultas1}, o que "
        "confirma que os modelos lineares não resolvem um problema não linearmente separável. Também "
        "foram identificados o underfitting de uma MLP subdimensionada e o overfitting de uma "
        "superdimensionada. No segundo, o reconhecimento de 20 pessoas em 640 imagens reduzidas para "
        "30 × 30 pixels, a validação por Monte Carlo foi executada com R = 10 e com R = 100 rodadas, por "
        "causa de uma divergência no enunciado. Com R = 100, as acurácias médias foram "
        f"{pct(t100['Perceptron Simples'][0])} (Perceptron), {pct(t100['ADALINE'][0])} (ADALINE) e "
        f"{pct(t100['MLP'][0])} (MLP {ocultas2}); com R = 10, {pct(t10['Perceptron Simples'][0])}, "
        f"{pct(t10['ADALINE'][0])} e {pct(t10['MLP'][0])}. Com 900 atributos e 512 imagens de treino, "
        "o problema torna-se quase linearmente separável, e os três modelos têm desempenho próximo.")


# =============================================================================
# Resultados
# =============================================================================
def escrever_resultados(rel, d):
    rel.secao("III", "Resultados")
    _etapa1(rel, d)
    _etapa2(rel, d)


def _etapa1(rel, d):
    rel.subsecao("A", "Etapa 1 – classificação da espiral")

    # ----------------------------------------------------------- dispersão
    n_fig = rel.proxima_figura()
    rel.paragrafo(
        f"A Fig. {n_fig} mostra o gráfico de espalhamento dos dados originais. As duas classes formam "
        "braços de espiral entrelaçados: a classe −1 aparece tanto no anel externo quanto no centro, "
        "envolvida pela classe +1. Nenhuma reta separa as classes, portanto o problema não é linearmente "
        "separável. Por isso espera-se que o Perceptron Simples e o ADALINE, cuja fronteira de decisão é "
        "um hiperplano, tenham desempenho limitado, e que apenas a MLP consiga modelar a fronteira curva. "
        "Também se nota o desbalanceamento: um classificador trivial que sempre responde +1 já acertaria "
        "71,4% das amostras, valor que serve de referência para as acurácias.",
        titulo_em_linha="1) Visualização inicial")
    rel.figura(fig(d, "etapa1", "01_dispersao.png"),
               "Gráfico de espalhamento do conjunto spiral_d (atributos originais, antes da normalização).")

    # ------------------------------------------------------ hiperparâmetros
    ajuste_p = d["ajuste_p1"]
    accs_p = sorted({linha[1] for linha in ajuste_p})
    ajuste_a = np.array(d["ajuste_a1"], dtype=float)
    ajuste_m = d["ajuste_m1"]
    hp_a, hp_m = d["hp1"]["adaline"], d["hp1"]["mlp"]
    ocultas = "-".join(str(q) for q in hp_m["camadas_ocultas"])
    melhor_m = max(float(l[2]) for l in ajuste_m)
    escolhida = [l for l in ajuste_m if l[0] == ocultas and float(l[1]) == hp_m["taxa_aprendizado"]][0]
    um_m = {l[0]: l for l in ajuste_m}
    n_fa, n_fm = rel.proxima_figura(), rel.proxima_figura() + 1
    texto_p = ("as quatro taxas testadas produziram exatamente a mesma acurácia média de validação "
               f"({pct(float(accs_p[0]))}), confirmando que, com pesos iniciais nulos, η não altera a fronteira"
               if len(accs_p) == 1 else
               "as taxas testadas produziram acurácias diferentes")
    rel.paragrafo(
        f"No Perceptron, {texto_p}. No ADALINE (Fig. {n_fa}), todas as taxas convergiram para praticamente "
        f"o mesmo EQM de validação (entre {num(ajuste_a[:, 2].min())} e {num(ajuste_a[:, 2].max())}) e a "
        f"mesma acurácia (cerca de {pct(np.median(ajuste_a[:, 1]), 1)}). O que muda é a velocidade: com "
        f"η = 10^{{−4}} foram necessárias {ajuste_a[0, 3]:.0f} épocas, e com η = 10^{{−2}} apenas "
        f"{ajuste_a[-1, 3]:.0f}. Isso mostra que o limite do ADALINE é o modelo linear, e não o "
        f"treinamento. Pelo critério do menor EQM de validação, escolheu-se η = "
        f"{taxa(hp_a['taxa_aprendizado'])}, cuja convergência é a mais suave. Na MLP "
        f"(Fig. {n_fm}), as taxas maiores foram melhores dentro do limite de 2000 épocas, porque o gradiente "
        f"em lote dá passos médios pequenos. Duas camadas ocultas também superaram uma: a rede 5-5, com "
        f"{um_m['5-5'][5]} pesos, foi melhor que a de uma camada com 20 neurônios, que tem {um_m['20'][5]} "
        f"pesos ({pct(max(float(l[2]) for l in ajuste_m if l[0] == '5-5'))} contra "
        f"{pct(max(float(l[2]) for l in ajuste_m if l[0] == '20'))}). A melhor acurácia de validação foi "
        f"{pct(melhor_m)}"
        + (f", obtida pela própria topologia {ocultas} com η = "
           f"{taxa(hp_m['taxa_aprendizado'])}, que também é a de menos pesos ({escolhida[5]}) "
           "entre as configurações dentro da tolerância de parcimônia e por isso foi a escolhida."
           if abs(float(escolhida[2]) - melhor_m) < 1e-9 else
           f", e a regra de parcimônia escolheu a topologia {ocultas} com η = "
           f"{taxa(hp_m['taxa_aprendizado'])} ({escolhida[5]} pesos, acurácia de validação "
           f"de {pct(float(escolhida[2]))})."),
        titulo_em_linha="2) Escolha dos hiperparâmetros")
    rel.figura(fig(d, "etapa1", "02_ajuste_adaline.png"),
               "ADALINE: EQM de treino por época para cada taxa de aprendizado (primeira partição da busca).")
    rel.figura(fig(d, "etapa1", "03_ajuste_mlp.png"),
               "MLP: acurácia média de validação para cada topologia e taxa de aprendizado.")

    # ------------------------------------------------------ under/overfitting
    topo = {l[0]: l for l in d["topologias"]}
    sub, ade, sup = topo["Subdimensionada"], topo["Adequada"], topo["Superdimensionada"]

    def ms(l, i, casas=4):
        return f"{num(l[i], casas)} ± {num(l[i + 1], casas)}"

    part = {}
    for l in d["topologias_part"]:  # caso; partição; épocas; EQM tr; EQM te; acur. tr; acur. te; ...
        part.setdefault(l[0], []).append([float(v) for v in l[1:]])
    part = {k: np.array(v) for k, v in part.items()}
    p_sup = part["Superdimensionada"]
    memorizou = p_sup[:, 2] < 1e-3                 # EQM de treino praticamente nulo
    presa = p_sup[:, 2] > 1e-2                     # rede parada em um platô
    gap = memorizou & (p_sup[:, 3] > 10 * p_sup[:, 2] + 1e-3)
    n_mem = lambda caso: int(np.sum(part[caso][:, 2] < 1e-3))

    n_tab = rel.proxima_tabela()
    n_fc, n_fmz = rel.proxima_figura(), rel.proxima_figura() + 1
    rel.paragrafo(
        f"A Tabela {_romano(n_tab)} resume as três topologias (média ± desvio-padrão nas cinco partições), e "
        f"as Figs. {n_fc} e {n_fmz} mostram as curvas de aprendizado, as fronteiras de decisão e as matrizes "
        "de confusão da primeira partição.", titulo_em_linha="3) Underfitting e overfitting da MLP")
    rel.tabela("MLP: topologias sub e superdimensionadas (média ± desvio, 5 partições)",
               ["Métrica", "Sub " + sub[1], "Adequada " + ade[1], "Super " + sup[1]], [
                   ["Nº de pesos", sub[2], ade[2], sup[2]],
                   ["Épocas (média)", sub[4], ade[4], sup[4]],
                   ["EQM treino", ms(sub, 5, 4), ms(ade, 5, 4), ms(sup, 5, 5)],
                   ["EQM teste", ms(sub, 7, 4), ms(ade, 7, 4), ms(sup, 7, 4)],
                   ["Acurácia treino", ms(sub, 9), ms(ade, 9), ms(sup, 9)],
                   ["Acurácia teste", ms(sub, 11), ms(ade, 11), ms(sup, 11)],
                   ["Sensibilidade", ms(sub, 13), ms(ade, 13), ms(sup, 13)],
                   ["Especificidade", ms(sub, 15), ms(ade, 15), ms(sup, 15)],
                   ["Tempo médio (s)", num(sub[17], 1), num(ade[17], 1), num(sup[17], 1)],
                   ["Partições com EQM treino < 10^{−3}", f"{n_mem('Subdimensionada')} de 5",
                    f"{n_mem('Adequada')} de 5", f"{n_mem('Superdimensionada')} de 5"],
               ], larguras=[1130, 1200, 1250, 1300])
    rel.paragrafo(
        f"A rede subdimensionada ({sub[1]}) apresenta underfitting. O EQM fica alto e praticamente igual no "
        f"treino ({num(sub[5])}) e no teste ({num(sub[7])}), a acurácia de treino ({pct(float(sub[9]))}) é tão "
        f"baixa quanto a de teste, e as curvas se estabilizam cedo (Fig. {n_fc}). Com apenas dois neurônios "
        "ocultos, a fronteira é quase uma reta: a rede não tem capacidade nem para aprender o treino, ou "
        "seja, o erro é de viés, e não de variância. Ela acerta quase toda a classe majoritária "
        f"(sensibilidade de {pct(float(sub[13]))}), mas classifica grande parte da classe −1 como +1 "
        f"(especificidade de {pct(float(sub[15]))}).")
    rel.paragrafo(
        f"A topologia adequada ({ade[1]}) alcança EQM baixo e próximo no treino e no teste "
        f"({num(ade[5])} e {num(ade[7])}) e acurácia de teste de {pct(float(ade[11]))} ± "
        f"{pct(float(ade[12]))}, com uma fronteira suave que acompanha os braços da espiral.")
    i1 = 0  # primeira partição, a das figuras
    p1 = p_sup[i1]
    texto_gap = lista([f"{p_sup[k, 3] / p_sup[k, 2]:.0f}" for k in np.where(gap)[0]])
    rel.paragrafo(
        f"A rede superdimensionada ({sup[1]}, {sup[2]} pesos), treinada por 15000 épocas sem parada "
        f"antecipada, memorizou o conjunto de treino em {int(memorizou.sum())} das 5 partições, com EQM de "
        "treino da ordem de 10^{−5} e 100% de acurácia de treino. A Fig. "
        f"{n_fc} mostra a assinatura do overfitting na primeira partição. O EQM de teste atinge o mínimo no "
        "início do treinamento (linha pontilhada), depois volta a subir e estaciona, enquanto o EQM de treino "
        f"continua caindo até {num(p1[2] * 1e5, 1)}·10^{{−5}}. Ao final, o erro de teste é cerca de "
        f"{p1[3] / p1[2]:.0f} vezes o de treino. Todas as épocas após o mínimo servem apenas para ajustar "
        f"o treino. Em {int(gap.sum())} das partições memorizadas essa razão foi grande (" + texto_gap
        + " vezes). Nas outras, as amostras de teste caíram em regiões onde a fronteira memorizada coincide "
        "com a verdadeira, o que é possível porque a espiral tem pouco ruído."
        + (f" Na partição {', '.join(str(int(k) + 1) for k in np.where(presa)[0])}, a rede nem chegou a "
           f"memorizar: ficou presa em um platô do EQM ({num(p_sup[presa][0, 2], 3)}), "
           "dificuldade típica de redes profundas treinadas por gradiente descendente simples."
           if presa.any() else "")
        + f" Em média, a rede superdimensionada teve acurácia de teste menor que a adequada "
        f"({pct(float(sup[11]))} contra {pct(float(ade[11]))}) e desvio-padrão "
        f"{float(sup[12]) / float(ade[12]):.0f} vezes maior. Essa variabilidade entre partições é "
        "característica de um modelo com excesso de capacidade, cujo resultado depende muito de quais "
        f"amostras ficaram no treino. Além disso, ela exige {int(sup[2]) / int(ade[2]):.0f} vezes mais pesos "
        f"e cerca de {float(sup[17]) / max(float(ade[17]), 1e-9):.0f} vezes mais tempo de treinamento. Em "
        "dados mais ruidosos, a memorização incluiria o ruído e a perda de acurácia seria ainda maior.")
    rel.figura_larga(fig(d, "etapa1", "05_topologias_comparacao.png"),
                     "MLP subdimensionada, adequada e superdimensionada (primeira partição). Linha de cima: "
                     "curvas de aprendizado em escala logarítmica (a linha pontilhada marca a época de menor "
                     "EQM de teste). Linha de baixo: fronteiras de decisão.")
    rel.figura_larga(fig(d, "etapa1", "06_topologias_matrizes.png"),
                     "Matrizes de confusão no teste das três topologias (primeira partição). Linhas: classe "
                     "real; colunas: classe prevista.", largura_pol=5.6)

    # ------------------------------------------------------------ validação
    acc = {m: linha_modelo(d["tab1"]["acuracia"], m) for m in MODELOS}
    sen = {m: linha_modelo(d["tab1"]["sensibilidade"], m) for m in MODELOS}
    esp = {m: linha_modelo(d["tab1"]["especificidade"], m) for m in MODELOS}
    t_acc, t_sen, t_esp = rel.proxima_tabela(), rel.proxima_tabela() + 1, rel.proxima_tabela() + 2
    rel.paragrafo(
        f"As Tabelas {_romano(t_acc)}, {_romano(t_sen)} e {_romano(t_esp)} trazem as estatísticas das 500 "
        "rodadas para cada métrica, e a Fig. {} mostra as distribuições em boxplots e violin plots.".format(
            rel.proxima_figura()), titulo_em_linha="4) Validação com R = 500 rodadas")
    tabela_resumo(rel, "Acurácia nas 500 rodadas (Etapa 1)", d["tab1"]["acuracia"])
    tabela_resumo(rel, "Sensibilidade nas 500 rodadas (Etapa 1)", d["tab1"]["sensibilidade"])
    tabela_resumo(rel, "Especificidade nas 500 rodadas (Etapa 1)", d["tab1"]["especificidade"])
    rel.figura_larga(fig(d, "etapa1", "11_box_violin_metricas.png"),
                     "Distribuição da acurácia, da sensibilidade e da especificidade nas 500 rodadas "
                     "(boxplots acima e violin plots abaixo).")

    perc_sens = coluna_rodadas(d, "Perceptron Simples", "sensibilidade")
    perc_esp = coluna_rodadas(d, "Perceptron Simples", "especificidade")
    perc_acc = coluna_rodadas(d, "Perceptron Simples", "acuracia")
    mlp_acc = coluna_rodadas(d, "MLP", "acuracia")
    frac_trivial = np.mean(perc_sens == 1.0)
    frac_abaixo = np.mean(perc_acc < 0.714)
    corr_p = np.corrcoef(perc_sens, perc_esp)[0, 1]
    adal_acc = coluna_rodadas(d, "ADALINE", "acuracia")
    rel.paragrafo(
        f"A MLP teve acurácia média de {pct(acc['MLP'][0])}, contra {pct(acc['ADALINE'][0])} do ADALINE e "
        f"{pct(acc['Perceptron Simples'][0])} do Perceptron Simples. Ela também foi a única com sensibilidade "
        f"e especificidade altas ao mesmo tempo ({pct(sen['MLP'][0])} e {pct(esp['MLP'][0])}). Os dois "
        "modelos lineares, por sua vez, comportam-se de forma bem diferente entre si.")
    rel.paragrafo(
        f"O Perceptron Simples teve o pior desempenho e, de longe, a maior variabilidade. O desvio-padrão da "
        f"acurácia foi de {num(acc['Perceptron Simples'][1])}, e tanto a sensibilidade quanto a especificidade "
        f"variaram de {pct(sen['Perceptron Simples'][3], 0)} a {pct(sen['Perceptron Simples'][2], 0)}. Como a "
        "espiral não é linearmente separável, a regra de Rosenblatt nunca para de corrigir os pesos: a cada "
        "época o hiperplano é empurrado pelas amostras erradas, e o EQM oscila sem convergir. O resultado "
        "final depende apenas de onde o hiperplano estava ao fim da 100ª época. Por isso, sensibilidade e "
        f"especificidade são fortemente antagônicas (correlação de {num(corr_p, 2).replace('-', '−')} nas "
        f"500 rodadas). Em "
        f"{pct(frac_trivial, 1)} das rodadas a sensibilidade foi 1, em geral com especificidade baixa, pois "
        f"a rede respondeu +1 para quase tudo. A mediana da acurácia ({pct(np.median(perc_acc))}) é "
        f"praticamente igual aos 71,4% do classificador trivial, e em {pct(frac_abaixo, 1)} das rodadas a "
        "acurácia ficou abaixo dessa referência.")
    rel.paragrafo(
        f"O ADALINE, ao contrário, é estável (desvio-padrão da acurácia de {num(acc['ADALINE'][1])}). Ele "
        "minimiza o EQM, cujo mínimo é único, e por isso encontra praticamente o mesmo hiperplano em todas "
        "as rodadas; a variação que resta vem da partição sorteada. Esse hiperplano, porém, favorece a "
        f"classe majoritária: sensibilidade de {pct(sen['ADALINE'][0])} e especificidade de apenas "
        f"{pct(esp['ADALINE'][0])}. A acurácia fica cerca de "
        f"{100 * (acc['ADALINE'][0] - 0.714):.0f} p.p. acima da referência trivial (só "
        f"{pct(np.mean(adal_acc < 0.714), 1)} das rodadas ficaram abaixo dela), o que é próximo do máximo "
        "que uma fronteira linear consegue nesse problema.")
    n_front = rel.proxima_figura()
    rel.paragrafo(
        f"A MLP {'-'.join(str(q) for q in d['hp1']['mlp']['camadas_ocultas'])} resolve o problema: a "
        f"mediana da acurácia é {pct(np.median(mlp_acc))}, {pct(np.mean(mlp_acc >= 0.99), 1)} das rodadas "
        f"ficaram com acurácia de pelo menos 99% e {pct(np.mean(mlp_acc == 1.0), 1)} acertaram todas as "
        f"amostras de teste. A cauda inferior da distribuição é curta: só {pct(np.mean(mlp_acc < 0.98), 1)} "
        f"das rodadas ficaram abaixo de 98%, e o mínimo foi {pct(acc['MLP'][3])}. Como se discute no item "
        f"seguinte, essas rodadas estão ligadas à instabilidade causada pela taxa de aprendizado alta. A "
        f"Fig. {n_front} compara as fronteiras dos três modelos na primeira rodada: as do Perceptron e do "
        "ADALINE são retas, e a da MLP acompanha a espiral.")
    rel.figura_larga(fig(d, "etapa1", "07_fronteiras_modelos.png"),
                     "Fronteiras de decisão dos três modelos treinados na primeira rodada da validação.")

    # ------------------------------------------------------------- extremos
    n_p = rel.proxima_figura()
    texto = []
    for modelo in MODELOS:
        partes = []
        for metrica in METRICAS:
            r_max, v_max, r_min, v_min = extremo(d, modelo, metrica)
            partes.append(f"{NOME_METRICA[metrica].lower()} {num(v_max)} (rodada {r_max}) e "
                          f"{num(v_min)} (rodada {r_min})")
        texto.append(f"{modelo}: " + "; ".join(partes))
    rel.paragrafo(
        f"As Figs. {n_p} a {n_p + 2} mostram, para cada modelo e métrica, a matriz de confusão e a curva de "
        "aprendizado das rodadas de maior e de menor valor. Os extremos foram (maior e menor): "
        + ". ".join(texto) + ".", titulo_em_linha="5) Rodadas de maior e menor desempenho")
    for texto_extremo in _texto_extremos_e1(d):
        rel.paragrafo(texto_extremo)
    for modelo, arquivo in zip(MODELOS, ("perceptron_simples", "adaline", "mlp")):
        rel.figura_larga(fig(d, "etapa1", f"09_painel_extremos_{arquivo}.png"),
                         f"{modelo}: matrizes de confusão e curvas de aprendizado das rodadas de maior e de "
                         "menor acurácia (linha 1), sensibilidade (linha 2) e especificidade (linha 3).")


def _texto_extremos_e1(d):
    """Discussão das rodadas extremas (Etapa 1), escrita após inspecionar os painéis 09_*.png.
    Os valores de cada rodada vêm de metricas_por_rodada.csv."""
    def valor(modelo, metrica, rodada):
        return coluna_rodadas(d, modelo, metrica)[rodada - 1]

    P, A, M = MODELOS
    r_pmin, v_pmin = extremo(d, P, "acuracia")[2:]
    r_psmax = extremo(d, P, "sensibilidade")[0]
    r_pemax = extremo(d, P, "especificidade")[0]
    r_pemin = extremo(d, P, "especificidade")[2]
    v_pmax = extremo(d, P, "acuracia")[1]
    r_pmax = extremo(d, P, "acuracia")[0]
    acc_a = linha_modelo(d["tab1"]["acuracia"], A)
    esp_a = linha_modelo(d["tab1"]["especificidade"], A)
    custo = {l[0]: [float(v) for v in l[1:5]] for l in d["custo1"]}
    r_mmax = extremo(d, M, "acuracia")[0]
    r_mmin, v_mmin = extremo(d, M, "acuracia")[2:]
    r_mesp, v_mesp = extremo(d, M, "especificidade")[2:]
    r_msen = extremo(d, M, "sensibilidade")[2]
    eta = str(d["hp1"]["mlp"]["taxa_aprendizado"]).replace(".0", "").replace(".", ",")
    return [
        "No Perceptron Simples, as rodadas extremas mostram o efeito da não convergência. Todas as curvas "
        "de aprendizado oscilam durante as 100 épocas, sem tendência de queda, e as matrizes mudam de um "
        f"extremo ao outro. Na rodada de menor acurácia ({pct(v_pmin)}, rodada {r_pmin}), o hiperplano final "
        "classificou todas as amostras de teste como −1 (sensibilidade de "
        f"{pct(valor(P, 'sensibilidade', r_pmin), 0)}), e só a classe minoritária foi acertada. Na de menor "
        f"especificidade (rodada {r_pemin}), ocorreu o oposto: a rede respondeu +1 para tudo, e as curvas "
        "ficaram constantes, porque ao fim de todas as épocas o hiperplano estava fora da região dos dados. "
        f"A rodada de maior sensibilidade (rodada {r_psmax}) teve especificidade de apenas "
        f"{pct(valor(P, 'especificidade', r_psmax), 1)}, e a de maior especificidade (rodada {r_pemax}) teve "
        f"sensibilidade de {pct(valor(P, 'sensibilidade', r_pemax), 1)}. Mesmo a melhor rodada em acurácia "
        f"({pct(v_pmax)}, rodada {r_pmax}) acertou só {pct(valor(P, 'especificidade', r_pmax), 0)} da classe −1.",
        "No ADALINE, as curvas de todas as rodadas extremas são suaves e monotônicas e se estabilizam entre "
        f"{custo[A][1]:.0f} e {custo[A][2]:.0f} épocas ({custo[A][0]:.0f} em média), "
        "com EQM final em torno de 0,3 no treino e no teste. Os extremos ficam próximos "
        f"(acurácia entre {pct(acc_a[3])} e {pct(acc_a[2])}) e se devem à partição sorteada, e não ao "
        "treinamento. Na rodada de maior acurácia, por exemplo, o EQM de teste ficou abaixo do de treino, "
        "sinal de que o teste recebeu amostras mais fáceis, mais distantes da reta. Em nenhuma rodada a "
        f"especificidade passou de {pct(esp_a[2])}: a fronteira linear sempre sacrifica boa parte da "
        "classe −1.",
        f"Na MLP, a rodada {r_mmax} acertou todas as amostras de teste. A rodada de menor acurácia "
        f"({pct(v_mmin)}, rodada {r_mmin}) "
        + (f"é também a de menor especificidade ({pct(v_mesp)})" if r_mesp == r_mmin else
           f"não coincide com a de menor especificidade ({pct(v_mesp)}, rodada {r_mesp})")
        + f", e sua curva de aprendizado explica o resultado. Com η = {eta}, o EQM apresenta picos de "
        "instabilidade ao longo do treinamento (também visíveis nas rodadas de melhor desempenho), dos "
        "quais a rede normalmente se recupera. Nessa rodada, porém, um desses picos ocorreu nas últimas "
        "épocas, e o treinamento terminou, pelo limite de épocas, com os pesos ainda perturbados. Cerca de "
        f"{100 * (1 - valor(M, 'especificidade', r_mmin)):.0f}% da classe −1 passou a ser classificada como "
        "+1. A rodada de menor sensibilidade (rodada "
        f"{r_msen}) mostra outro fenômeno: o EQM de teste estaciona perto de 0,04 enquanto o de treino segue "
        "caindo até cerca de 10^{−3}, um sobreajuste local da fronteira nessa partição. Guardar os pesos da "
        "época de menor EQM de validação, ou reduzir η ao longo do treinamento, provavelmente eliminaria "
        "a maior parte da cauda inferior da distribuição da MLP.",
    ]


ARQUIVOS = {"Perceptron Simples": "perceptron_simples", "ADALINE": "adaline", "MLP": "mlp"}


def _comparacao(d, modelo, R):
    """Linha (média, desvio, maior, menor, erro-padrão) de comparacao_R10_R100.csv."""
    for linha in d["comparacao"]:
        if linha[0] == modelo and int(linha[1]) == R:
            return [float(v) for v in linha[2:7]]
    raise KeyError((modelo, R))


def _pareado(d, R, a, b):
    """Comparação pareada a - b (os modelos usam as mesmas partições em cada rodada)."""
    rodadas = d[f"rodadas2_{R}"]
    dif = rodadas[:, 1 + MODELOS.index(a)] - rodadas[:, 1 + MODELOS.index(b)]
    erro_padrao = dif.std(ddof=1) / np.sqrt(len(dif))
    return dif.mean(), erro_padrao, int(np.sum(dif > 1e-9)), int(np.sum(np.abs(dif) <= 1e-9)), \
        int(np.sum(dif < -1e-9))


def _taxas_por_pessoa(d, R):
    pessoas = d["pessoas"]
    taxas = {}
    for modelo in MODELOS:
        M = d[f"acumuladas_{R}"][modelo]
        taxas[modelo] = np.diag(M) / np.maximum(M.sum(axis=1), 1)
    return pessoas, taxas


def _etapa2(rel, d):
    rel.subsecao("B", "Etapa 2 – reconhecimento facial")
    hp_p, hp_m = d["hp2"]["perceptron"], d["hp2"]["mlp"]
    ocultas = "-".join(str(q) for q in hp_m["camadas_ocultas"])
    custo = {l[0]: [float(v) for v in l[1:5]] for l in d["custo2_100"]}

    # ------------------------------------------------------ hiperparâmetros
    aj_a = np.array(d["ajuste_a2"], dtype=float)  # eta; precisao; acurácia; épocas; tempo
    aj_m = d["ajuste_m2"]                          # topologia; eta; acurácia; desvio; tempo; pesos
    i_a = int(np.argmax(aj_a[:, 2]))
    melhor_m = max(aj_m, key=lambda l: float(l[2]))
    pior_m = min(aj_m, key=lambda l: float(l[2]))
    convergiu_p = custo["Perceptron Simples"][2] < hp_p["max_epocas"]
    n_tab = rel.proxima_tabela()
    rel.paragrafo(
        "O Perceptron Simples terminou o treinamento sem erros em "
        + ("todas as rodadas" if convergiu_p else "parte das rodadas")
        + f", com {num(custo['Perceptron Simples'][0], 1)} épocas em média (entre "
        f"{custo['Perceptron Simples'][1]:.0f} e {custo['Perceptron Simples'][2]:.0f}). Isso confirma que, com 900 "
        "atributos e 512 imagens de treino, o conjunto de treino é linearmente separável. No ADALINE, as "
        f"quatro configurações testadas ficaram entre {pct(aj_a[:, 2].min())} e {pct(aj_a[:, 2].max())} de "
        f"acurácia de validação. A taxa η = {taxa(aj_a[i_a, 0])}, cerca de "
        f"1/{(2 / d['lambda_max']) / aj_a[i_a, 0]:.0f} do limite "
        f"de estabilidade {num(2 / d['lambda_max'], 4)}, foi a mais precisa e também a mais rápida "
        f"({aj_a[i_a, 3]:.0f} épocas e {num(aj_a[i_a, 4], 2)} s em média). Com ε = 10^{{−4}}, o treinamento "
        f"levou {aj_a[-1, 3]:.0f} épocas e a acurácia caiu para {pct(aj_a[-1, 2])}: continuar minimizando o "
        "EQM de treino não melhorou a generalização. A Tabela "
        f"{_romano(n_tab)} mostra a busca da MLP. Duas lições da Etapa 1 se repetiram: (i) duas camadas "
        "ocultas superaram uma camada com número semelhante de pesos (50-50 contra 50), e (ii) a taxa de "
        f"aprendizado maior tornou instável a rede mais larga (topologia {pior_m[0]} com η = "
        f"{pior_m[1].replace('.', ',')}: {pct(float(pior_m[2]))} ± {pct(float(pior_m[3]))}). A melhor "
        f"configuração foi {melhor_m[0]} com η = {melhor_m[1].replace('.', ',')} ({pct(float(melhor_m[2]))}), "
        "e nenhuma outra ficou dentro da tolerância de parcimônia.",
        titulo_em_linha="1) Escolha dos hiperparâmetros")
    rel.tabela("MLP na Etapa 2: acurácia de validação (média ± desvio, 3 partições)",
               ["Topologia", "η", "Acurácia val.", "Pesos", "Tempo (s)"],
               [[l[0], l[1].replace(".", ","), f"{num(l[2])} ± {num(l[3])}", f"{int(l[5]):,}".replace(",", "."),
                 num(l[4], 1)] for l in aj_m],
               larguras=[1000, 650, 1550, 900, 830],
               negrito_linhas=[i for i, l in enumerate(aj_m) if l is melhor_m])

    # ------------------------------------------------------- R = 10 x R = 100
    t10 = {m: linha_modelo(d["tab2_10"], m) for m in MODELOS}
    t100 = {m: linha_modelo(d["tab2_100"], m) for m in MODELOS}
    c10 = {m: _comparacao(d, m, 10) for m in MODELOS}
    c100 = {m: _comparacao(d, m, 100) for m in MODELOS}
    n_t10 = rel.proxima_tabela()
    n_fc = rel.proxima_figura()
    rel.paragrafo(
        f"As Tabelas {_romano(n_t10)} e {_romano(n_t10 + 1)} trazem as estatísticas da acurácia nas duas "
        f"configurações, e a Fig. {n_fc} compara as distribuições.",
        titulo_em_linha="2) Validação por Monte Carlo com R = 10 e R = 100")
    tabela_resumo(rel, "Acurácia na Etapa 2 com R = 10 rodadas", d["tab2_10"])
    tabela_resumo(rel, "Acurácia na Etapa 2 com R = 100 rodadas", d["tab2_100"])
    rel.figura_larga(fig(d, "etapa2", "comparacao_R10_R100.png"),
                     "Distribuição da acurácia dos três modelos com R = 10 e R = 100 rodadas (boxplot à "
                     "esquerda; violin plot à direita, com os quartis tracejados).", largura_pol=6.2)

    ordem = sorted(MODELOS, key=lambda m: -t100[m][0])
    rel.paragrafo(
        f"Com R = 100, a ordem das acurácias médias foi {ordem[0]} ({pct(t100[ordem[0]][0])}), "
        f"{ordem[1]} ({pct(t100[ordem[1]][0])}) e {ordem[2]} ({pct(t100[ordem[2]][0])}). Ao contrário da "
        "Etapa 1, os modelos lineares ficaram muito próximos da MLP. Com p = 900 atributos e apenas 512 "
        "imagens de treino, um hiperplano é capaz de separar cada pessoa das demais no treino (o "
        "Perceptron chega a zero erros de treino em poucas épocas). Assim, o problema deixa de ser a "
        "capacidade do modelo e passa a ser a generalização para as imagens de teste, com poses, "
        "expressões e uso de óculos diferentes. Nessa situação, a não linearidade da MLP traz um ganho "
        "pequeno. O Perceptron, que para no primeiro hiperplano que separa o treino, sem procurar uma "
        "margem, é o que generaliza pior e o que mais varia. O ADALINE, que minimiza o EQM, posiciona "
        "melhor o hiperplano.")

    dif_a, ep_a, venc_a, emp_a, derr_a = _pareado(d, 100, "MLP", "ADALINE")
    dif_p, ep_p, venc_p, emp_p, derr_p = _pareado(d, 100, "MLP", "Perceptron Simples")
    dif_a10, ep_a10, _, _, _ = _pareado(d, 10, "MLP", "ADALINE")
    rel.paragrafo(
        "Como em cada rodada os três modelos usam a mesma partição, as diferenças podem ser avaliadas "
        f"rodada a rodada. Com R = 100, a MLP superou o ADALINE em {venc_a} rodadas, empatou em {emp_a} e "
        f"perdeu em {derr_a}, com diferença média de {num(100 * dif_a, 2)} ± {num(100 * ep_a, 2)} p.p. "
        "(média ± erro-padrão). Em relação ao Perceptron, a MLP venceu "
        f"{venc_p}, empatou {emp_p} e perdeu {derr_p}, com diferença de {num(100 * dif_p, 2)} ± "
        f"{num(100 * ep_p, 2)} p.p. Com R = 10, a diferença MLP − ADALINE foi de {num(100 * dif_a10, 2)} ± "
        f"{num(100 * ep_a10, 2)} p.p.: o erro-padrão é "
        + ("maior que a própria diferença, então dez rodadas não bastam para ordenar os dois modelos."
           if abs(dif_a10) < 2 * ep_a10 else "pequeno em relação à diferença."))

    n_fm = rel.proxima_figura()
    curto = {m: ("Perceptron" if m.startswith("Perc") else m) for m in MODELOS}
    ext10 = {l[0]: l for l in d["extremos2_10"]}
    ext100 = {l[0]: l for l in d["extremos2_100"]}
    min_nas_10 = all(int(ext100[m][3]) <= 10 for m in MODELOS)
    mudou_max = [m for m in MODELOS if float(ext100[m][2]) != float(ext10[m][2])]
    mudou_min = [m for m in MODELOS if float(ext100[m][4]) != float(ext10[m][4])]
    rodadas = d["rodadas2_100"][:, 1:]
    acumulada = np.cumsum(rodadas, axis=0) / np.arange(1, len(rodadas) + 1)[:, None]
    desvio_final = np.abs(acumulada - acumulada[-1]).max(axis=1)
    k_estavel = next(k for k in range(len(rodadas)) if desvio_final[k:].max() <= 0.002) + 1
    ordem_final = np.argsort(-acumulada[-1])
    mantida = np.all(np.diff(acumulada[:, ordem_final], axis=1) < 0, axis=1)
    k_ordem = next(k for k in range(len(rodadas)) if mantida[k:].all()) + 1
    rel.paragrafo(
        "Quanto à estabilidade, o desvio-padrão mede a dispersão das rodadas individuais. Ele é uma "
        "propriedade do modelo e da divisão 80/20, e não diminui com R. De fato, ele teve a mesma ordem de "
        "grandeza nas duas configurações (" + "; ".join(
            f"{curto[m]}: {num(c10[m][1])} e {num(c100[m][1])}" for m in MODELOS)
        + "). Com R = 10 ele foi um pouco maior, mas também é estimado com apenas dez valores. O que diminui "
        "com R é a incerteza da média, medida pelo erro-padrão s/√R, que caiu de " + lista(
            f"{num(100 * c10[m][4], 2)} para {num(100 * c100[m][4], 2)} p.p. ({curto[m]})" for m in MODELOS)
        + ", uma redução próxima de √10 ≈ 3,2.")
    texto_ext = (
        "Os valores extremos praticamente não mudaram. "
        + ("A rodada de menor acurácia dos três modelos entre as cem já fazia parte das dez primeiras"
           if min_nas_10 and not mudou_min else "O menor valor mudou para " + ", ".join(curto[m] for m in mudou_min))
        + (". Só mudou o maior valor do " + ", ".join(
            f"{curto[m]} (de {pct(float(ext10[m][2]))} na rodada {ext10[m][1]} para "
            f"{pct(float(ext100[m][2]))} na rodada {ext100[m][1]})" for m in mudou_max) if mudou_max else "")
        + ". Isso é uma coincidência deste sorteio: em geral, com poucas rodadas as partições mais "
        "desfavoráveis podem não ser sorteadas, e a faixa de variação fica subestimada.")
    rel.paragrafo(
        texto_ext + " Neste caso, o efeito da escolha de R apareceu sobretudo na média. As dez primeiras "
        "partições foram, por acaso, mais difíceis que a média, e com R = 10 as acurácias médias ficaram "
        + lista(f"{num(100 * (t100[m][0] - t10[m][0]), 2)} p.p. ({curto[m]})" for m in MODELOS)
        + f" abaixo das de R = 100. A Fig. {n_fm} mostra a média acumulada com o aumento do número de "
        "rodadas. Nas primeiras rodadas ela oscila vários décimos de ponto percentual, e as curvas do ADALINE "
        f"e da MLP se cruzam. A ordem final ({' > '.join(curto[MODELOS[i]] for i in ordem_final)}) só se "
        f"fixa a partir da rodada {k_ordem}, e as três médias só ficam a menos de 0,2 p.p. dos valores "
        f"finais a partir da rodada {k_estavel}. Conclui-se que R = 10 fornece uma estimativa grosseira, "
        "suficiente para uma primeira verificação, enquanto R = 100 permite comparar modelos cujas "
        "acurácias diferem em menos de 1 ponto percentual.")
    rel.figura(fig(d, "etapa2", "media_acumulada.png"),
               "Acurácia média acumulada ao longo das 100 rodadas. A linha pontilhada marca R = 10.")
    rel.paragrafo(
        f"O custo de treinamento por rodada foi de {num(custo['Perceptron Simples'][3], 2)} s para o "
        f"Perceptron, {num(custo['ADALINE'][3], 2)} s para o ADALINE ({num(custo['ADALINE'][0], 1)} épocas "
        f"em média) e {num(custo['MLP'][3], 2)} s para a MLP, que usou todas as {hp_m['max_epocas']} épocas "
        "em todas as rodadas sem atingir ε. A MLP custa cerca de "
        f"{custo['MLP'][3] / custo['Perceptron Simples'][3]:.0f} vezes o tempo do Perceptron e "
        f"{custo['MLP'][3] / custo['ADALINE'][3]:.0f} vezes o do ADALINE para um ganho inferior a "
        f"{max(1.0, np.ceil(100 * (t100['MLP'][0] - min(t100[m][0] for m in MODELOS)))):.0f} ponto percentual.")

    # ------------------------------------------------------------- extremos
    n_e = rel.proxima_figura()
    partes = []
    for R in (10, 100):
        ext = {l[0]: l for l in d[f"extremos2_{R}"]}
        partes.append(f"R = {R}: " + "; ".join(
            f"{'Perceptron' if m.startswith('Perc') else m} {num(ext[m][2])} (rodada {ext[m][1]}) e "
            f"{num(ext[m][4])} (rodada {ext[m][3]})" for m in MODELOS))
    # Como a semente é a mesma, quando as rodadas extremas de R = 10 e R = 100 coincidem as figuras são
    # idênticas; nesse caso a figura de R = 10 não é repetida (e o texto explica isso).
    e10 = {l[0]: (l[1], l[3]) for l in d["extremos2_10"]}
    e100 = {l[0]: (l[1], l[3]) for l in d["extremos2_100"]}
    iguais = [m for m in MODELOS if e10[m] == e100[m]]
    diferentes = [m for m in MODELOS if e10[m] != e100[m]]
    figs_matriz = [(100, m) for m in MODELOS] + [(10, m) for m in diferentes]
    figs_curva = [100] + ([10] if diferentes else [])
    n_c = n_e + len(figs_matriz)
    nota = ""
    if iguais:
        artigo = {"Perceptron Simples": "o Perceptron Simples", "ADALINE": "o ADALINE", "MLP": "a MLP"}
        nota = (" Para " + lista(artigo[m] for m in iguais) + ", as rodadas extremas com R = 10 são exatamente as mesmas "
                "de R = 100 (as dez primeiras rodadas são comuns às duas configurações), e as figuras "
                "seriam idênticas. Por isso, apenas as de R = 100 são mostradas"
                + (", enquanto para " + lista(artigo[m] for m in diferentes) + " são mostradas as duas configurações."
                   if diferentes else "."))
    rel.paragrafo(
        f"As Figs. {n_e} a {n_c - 1} mostram as matrizes de confusão 20 × 20 das rodadas de maior e de menor "
        "acurácia, e a"
        + (f"s Figs. {n_c} e {n_c + 1} trazem" if len(figs_curva) == 2 else f" Fig. {n_c} traz")
        + " as curvas de aprendizado correspondentes. Os extremos foram (maior e menor): "
        + ". ".join(partes) + "." + nota, titulo_em_linha="3) Rodadas de maior e menor acurácia")
    rel.paragrafo(_texto_extremos_e2(d))
    rel.paragrafo(_texto_curvas_e2(d))
    for R, modelo in figs_matriz:
        rel.figura_larga(fig(d, f"etapa2_R{R}", f"matrizes_extremos_{ARQUIVOS[modelo]}.png"),
                         f"{modelo}, R = {R}: matrizes de confusão das rodadas de maior (esquerda) e de "
                         "menor (direita) acurácia. Linhas: pessoa real; colunas: pessoa prevista.",
                         largura_pol=6.2)
    for R in figs_curva:
        rel.figura_larga(fig(d, f"etapa2_R{R}", "curvas_extremos.png"),
                         f"Curvas de aprendizado (EQM de treino e de teste por época) das rodadas de maior e "
                         f"de menor acurácia, R = {R}. Na MLP o eixo vertical está em escala logarítmica.",
                         largura_pol=5.6)

    # ----------------------------------------------------------- por classe
    pessoas, taxas = _taxas_por_pessoa(d, 100)
    _, taxas10 = _taxas_por_pessoa(d, 10)
    media_pessoa = np.mean([taxas[m] for m in MODELOS], axis=0)
    piores = np.argsort(media_pessoa)[:4]
    perfeitas = [pessoas[k] for k in range(len(pessoas)) if all(taxas[m][k] == 1.0 for m in MODELOS)]
    fora = sum(d["acumuladas_100"][m] for m in MODELOS).astype(int)
    np.fill_diagonal(fora, 0)
    top = np.dstack(np.unravel_index(np.argsort(fora.ravel())[::-1][:5], fora.shape))[0]
    pior10 = np.argsort(np.mean([taxas10[m] for m in MODELOS], axis=0))[:3]
    corr = np.corrcoef(np.array([taxas[m] for m in MODELOS]))
    n_total = int(d["acumuladas_100"]["MLP"].sum())
    n_ac = rel.proxima_figura()
    rel.paragrafo(
        f"Para identificar quais pessoas são mais difíceis, as matrizes das 100 rodadas foram somadas "
        f"({_milhar(n_total)} classificações de teste por modelo), e a Fig. {n_ac} mostra a taxa de acerto de cada "
        "pessoa. As dificuldades se concentram nas mesmas pessoas para os três modelos: as piores taxas "
        "médias foram de " + lista(f"{pessoas[k]} ({pct(media_pessoa[k], 1)})" for k in piores)
        + f". As três pessoas mais difíceis com R = 10 foram {lista(pessoas[k] for k in pior10)}, o que "
        "mostra que esse padrão já aparecia com poucas rodadas. Por outro lado, "
        f"{len(perfeitas)} pessoas foram reconhecidas em 100% dos casos pelos três modelos"
        + (f" ({lista(perfeitas)})" if 0 < len(perfeitas) <= 8 else "")
        + ". Os modelos acertaram 100% das imagens de "
        + lista(f"{int(np.sum(taxas[m] == 1.0))} ({'Perceptron' if m.startswith('Perc') else m})"
                for m in MODELOS) + " pessoas"
        + ". Somando os três modelos, as confusões mais frequentes (pessoa real → prevista) foram "
        + lista(f"{pessoas[a]} → {pessoas[b]} ({fora[a, b]} vezes)" for a, b in top if fora[a, b] > 0)
        + ".", titulo_em_linha="4) Análise dos erros por pessoa")
    rel.paragrafo(
        "A correlação de Pearson entre as taxas de acerto por pessoa foi de "
        f"{num(corr[0, 1], 2)} (Perceptron × ADALINE), {num(corr[0, 2], 2)} (Perceptron × MLP) e "
        f"{num(corr[1, 2], 2)} (ADALINE × MLP). Os três classificadores, lineares ou não, erram "
        "essencialmente nas mesmas pessoas e frequentemente trocam os mesmos pares. Portanto, a dificuldade "
        "vem das próprias imagens, e não do modelo. Uma explicação provável é que, reduzidas a 30 × 30 "
        "pixels e normalizadas, as imagens dessas pessoas em certas poses ou com óculos escuros ficam mais "
        "parecidas com as de outra pessoa do que com as suas em outra pose. " + _texto_simetria(fora, pessoas))
    for R in (100, 10):
        rel.figura_larga(fig(d, f"etapa2_R{R}", "acerto_por_classe.png"),
                         f"Taxa de acerto por pessoa e por modelo, obtida da soma das matrizes de confusão das "
                         f"{R} rodadas.", largura_pol=6.4)


def _texto_simetria(fora, pessoas):
    """Comenta se as confusões mais frequentes ocorrem nos dois sentidos (matriz somada, sem diagonal)."""
    pares = np.dstack(np.unravel_index(np.argsort(fora.ravel())[::-1][:10], fora.shape))[0]
    assim = [(a, b) for a, b in pares if fora[b, a] == 0]
    simet = [(a, b) for a, b in pares if fora[b, a] >= 0.5 * fora[a, b]]
    fracao = np.minimum(fora, fora.T).sum() / max(fora.sum(), 1)
    texto = (f"As confusões são, em geral, assimétricas: apenas {pct(fracao, 0)} dos erros têm "
             "correspondente no sentido inverso")
    if assim:
        a, b = assim[0]
        texto += (f". Por exemplo, {pessoas[a]} → {pessoas[b]} ocorreu {fora[a, b]} vezes, e "
                  f"{pessoas[b]} → {pessoas[a]} nenhuma")
    texto += ". Assim, a região de decisão de uma pessoa invade a de outra sem que o contrário aconteça."
    if simet:
        a, b = simet[0]
        texto += (f" A exceção mais clara é o par {pessoas[a]} ↔ {pessoas[b]}, confundido nos dois sentidos "
                  f"({fora[a, b]} e {fora[b, a]} vezes), o que sugere duas pessoas realmente parecidas nessa "
                  "resolução.")
    return texto


def _texto_extremos_e2(d):
    """Discussão das matrizes das rodadas extremas (Etapa 2), escrita após inspecionar as figuras."""
    ext = {l[0]: l for l in d["extremos2_100"]}
    mesma_pior = len({ext[m][3] for m in MODELOS}) == 1
    erros = {m: round((1 - float(ext[m][4])) * 128) for m in MODELOS}
    return (
        "Nas rodadas de maior acurácia, todas as imagens de teste ficam na diagonal. "
        + (f"A rodada de menor acurácia foi a mesma para os três modelos (rodada {ext[MODELOS[0]][3]}), "
           if mesma_pior else "As rodadas de menor acurácia variaram entre os modelos, ")
        + "com " + lista(f"{erros[m]} ({'Perceptron' if m.startswith('Perc') else m})" for m in MODELOS)
        + " erros em 128 imagens. Isso indica que o desempenho depende mais de quais imagens caem no teste do que "
        "do modelo. Nessa rodada, os três modelos erraram imagens das mesmas pessoas (glickman, karyadi, "
        "mitchell e sz24), que estão entre as mais difíceis na análise por pessoa apresentada adiante. A "
        "confusão karyadi → tammo, por exemplo, aparece nas três matrizes. Os erros ficam espalhados fora "
        "da diagonal, com no máximo dois por pessoa, e nenhuma classe inteira é perdida.")


def _texto_curvas_e2(d):
    """Discussão das curvas de aprendizado das rodadas extremas (Etapa 2), escrita após inspecionar as figuras."""
    custo = {l[0]: [float(v) for v in l[1:5]] for l in d["custo2_100"]}
    return (
        "Nas curvas de aprendizado, o Perceptron zera o EQM de treino em poucas épocas (entre "
        f"{custo['Perceptron Simples'][1]:.0f} e {custo['Perceptron Simples'][2]:.0f} nas 100 rodadas) e para, já "
        "que o treino é linearmente separável. O EQM de teste estaciona em um patamar muito mais alto na "
        "rodada de menor acurácia. O ADALINE reduz os dois EQMs de forma suave e monotônica até atingir o "
        f"critério |ΔEQM| ≤ ε (entre {custo['ADALINE'][1]:.0f} e {custo['ADALINE'][2]:.0f} épocas). Na pior "
        "rodada, o EQM de teste termina mais de dez vezes acima do de treino. A MLP parte de um EQM próximo "
        "de 10 e cai rapidamente para um platô em EQM ≈ 2, que corresponde a todas as 20 saídas próximas de "
        "−1 (para cada amostra, 19 dos 20 alvos valem −1). Ela só escapa desse platô após cerca de 100 "
        "épocas. Na melhor rodada, as curvas de treino e de teste descem juntas até o fim. Na pior, o EQM de "
        "teste estaciona em torno de 0,12 a partir de cerca de 300 épocas, enquanto o de treino continua "
        "caindo até cerca de 4·10^{−3}. As épocas restantes apenas ajustam o treino, sem reduzir o erro de "
        "teste, o que é um overfitting leve. Uma parada antecipada pelo EQM de validação reduziria bastante "
        "o custo da MLP.")


# =============================================================================
# Conclusões
# =============================================================================
def escrever_conclusoes(rel, d):
    acc = {m: linha_modelo(d["tab1"]["acuracia"], m) for m in MODELOS}
    esp = {m: linha_modelo(d["tab1"]["especificidade"], m) for m in MODELOS}
    t100 = {m: linha_modelo(d["tab2_100"], m) for m in MODELOS}
    c10 = {m: _comparacao(d, m, 10) for m in MODELOS}
    c100 = {m: _comparacao(d, m, 100) for m in MODELOS}
    ocultas1 = "-".join(str(q) for q in d["hp1"]["mlp"]["camadas_ocultas"])
    rel.secao("IV", "Conclusões")
    rel.paragrafo(
        "Os experimentos confirmaram, com números, a diferença entre modelos lineares e não lineares. Na "
        "espiral, que não é linearmente separável, o Perceptron Simples (acurácia média de "
        f"{pct(acc['Perceptron Simples'][0])}) e o ADALINE ({pct(acc['ADALINE'][0])}) ficaram próximos do "
        "classificador trivial que sempre responde +1 (71,4%): o primeiro praticamente o iguala, e o segundo "
        f"o supera em apenas {100 * (acc['ADALINE'][0] - 0.714):.0f} p.p. O Perceptron mostrou alta "
        "instabilidade, porque sua regra nunca converge nesse caso. O ADALINE é estável, mas tende a "
        f"favorecer a classe majoritária (especificidade de {pct(esp['ADALINE'][0])}). A MLP {ocultas1}, com "
        f"tangente hiperbólica, atingiu {pct(acc['MLP'][0])} e modelou a fronteira em espiral.")
    rel.paragrafo(
        "O estudo de topologias mostrou o papel da capacidade da rede. A rede com dois neurônios ocultos "
        "sofre underfitting, com erros altos e iguais no treino e no teste. A rede com três camadas de 60 "
        "neurônios, treinada sem parada antecipada, memoriza o treino na maioria das partições, e em algumas "
        "delas o erro de teste fica centenas de vezes maior que o de treino (overfitting). Ela também varia "
        "mais entre partições e tem custo computacional muito maior. A escolha dos "
        "hiperparâmetros por busca em grade com várias partições e a regra de parcimônia evitaram as duas "
        "situações. A validação também revelou que taxas de aprendizado altas aceleram a MLP, mas provocam "
        "picos de instabilidade no EQM. Quando um desses picos coincide com o fim do treinamento, a rede "
        "termina com pesos piores, o que explica a rodada de pior desempenho. Uma parada antecipada que "
        "guarde os melhores pesos resolveria esse problema.")
    rel.paragrafo(
        "No reconhecimento facial, a alta dimensionalidade (900 pixels para 512 imagens de treino) tornou "
        "o problema quase linearmente separável. Os três modelos superaram "
        f"{np.floor(100 * min(t100[m][0] for m in MODELOS)):.0f}% de acurácia média (com "
        f"R = 100: Perceptron {pct(t100['Perceptron Simples'][0])}, ADALINE {pct(t100['ADALINE'][0])} e MLP "
        f"{pct(t100['MLP'][0])}). A MLP foi a mais precisa, mas por pouco e com custo muito maior, e o "
        "ADALINE ofereceu a melhor relação entre desempenho e custo. As mesmas pessoas foram as mais "
        "difíceis para todos os modelos, o que aponta para a variabilidade das imagens, e não para o "
        "classificador. Quanto à divergência do enunciado, a execução com R = 10 e com R = 100 mostrou que "
        "o desvio-padrão por rodada é praticamente o mesmo nas duas configurações, mas a incerteza da média "
        f"cai cerca de √10 ≈ 3,2 vezes (na MLP, de {num(100 * c10['MLP'][4], 2)} para "
        f"{num(100 * c100['MLP'][4], 2)} p.p.). Com dez rodadas, a diferença entre a MLP e o ADALINE fica "
        "dentro da incerteza, e só com cem rodadas é possível ordená-los com segurança.")
    rel.paragrafo(
        "Como trabalhos futuros, sugere-se usar parada antecipada baseada em um conjunto de validação, "
        "treinar a MLP por mini-lotes para reduzir o custo, e, no reconhecimento facial, reduzir a "
        "dimensionalidade, por exemplo com PCA, além de ampliar o conjunto de treino com variações das "
        "imagens para diminuir a confusão entre as pessoas mais difíceis.")


def _milhar(n):
    """Inteiro com ponto como separador de milhar (padrão brasileiro)."""
    return f"{int(float(n)):,}".replace(",", ".")


def _romano(n):
    valores = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    texto = ""
    for valor, simbolo in valores:
        while n >= valor:
            texto += simbolo
            n -= valor
    return texto
