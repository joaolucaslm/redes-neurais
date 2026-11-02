# Trabalho Computacional AV2 – Redes Neurais Artificiais

Disciplina: Inteligência Artificial Computacional – UNIFOR  
Professor: Prof. Msc. Paulo Cirillo Souza Barbosa
Repositório: https://github.com/joaolucaslm/redes-neurais

Implementação, apenas com **numpy, matplotlib, seaborn e opencv-python**, dos modelos
**Perceptron Simples**, **ADALINE** e **Perceptron de Múltiplas Camadas (MLP)**, aplicados a:

- **Etapa 1** – classificação binária do conjunto `spiral_d.csv`;
- **Etapa 2** – reconhecimento facial de 20 pessoas (640 imagens da pasta `RecFac/`).

## Estrutura

```
modelos.py      classes PerceptronSimples, Adaline e MLP (fit, predict, EQM por época)
utils.py        leitura dos dados, normalização, bias, divisão treino/teste, one-hot,
                matriz de confusão, métricas, tabelas e gráficos
etapa1.py       experimentos da Etapa 1 (spiral_d.csv)
etapa2.py       experimentos da Etapa 2 (reconhecimento facial, R = 10 e R = 100)
requirements.txt
figuras/        figuras geradas (etapa1/, etapa2/, etapa2_R10/, etapa2_R100/)
resultados/     tabelas .csv/.txt e logs (mesma divisão de pastas)
relatorio/      relatório (.docx) e o script que o monta (fora do código avaliado)
```

## Como executar

Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python etapa1.py          # busca de hiperparâmetros + under/overfitting + 500 rodadas
python etapa2.py          # busca de hiperparâmetros + Monte Carlo com R = 10 e R = 100
```

Opções úteis:

| Comando | Efeito |
|---|---|
| `python etapa1.py --R 50` | menos rodadas (teste rápido) |
| `python etapa1.py --sem-ajuste` | pula a busca e usa os hiperparâmetros padrão do script |
| `python etapa2.py --R 10` | roda só uma configuração de R |
| `python etapa2.py --R 10 100 --sem-ajuste` | as duas configurações, sem a busca |

Tempo medido na execução final: Etapa 1 ≈ 20 min, Etapa 2 ≈ 13 min.
O andamento das rodadas é mostrado no terminal.

### Relatório

O relatório `relatorio/Relatorio_AV2.docx` é montado sobre o modelo `conference-template-a4.docx` a partir
das figuras e tabelas geradas acima (todos os números são lidos de `resultados/`). O script usa
`python-docx`, que fica fora do código avaliado:

```bash
pip install -r relatorio/requirements.txt
python relatorio/gerar_relatorio.py
```

## Observações

- **Pasta das imagens:** o código procura `RecFac` (ou `ReCFac`) na raiz e desce automaticamente
  quando há uma pasta aninhada (por exemplo `RecFac/RecFac/<pessoa>/`). O nome de cada subpasta é o
  rótulo da pessoa.
- **Reprodutibilidade:** semente fixa (42). Na Etapa 2 a mesma semente é usada para qualquer R, de modo
  que as 10 rodadas de R = 10 coincidem com as 10 primeiras de R = 100.
- **Divergência do enunciado (Etapa 2):** o item 5 pede R = 10 e o item 7 menciona R = 100. As duas
  configurações são executadas e reportadas, com saídas em pastas separadas.
- **Fora do escopo desta entrega:** Rede RBF e F1-score.
- **Formulação:** X tem dimensão (p+1)×N com o bias −1 na primeira linha, os rótulos são bipolares
  (+1/−1), o Perceptron e o ADALINE usam degrau bipolar e a MLP usa tangente hiperbólica. Todos os modelos
  param por número máximo de épocas.
- **Adaptações em relação aos slides** (justificadas nos comentários do código e no relatório):
  1. limite de épocas no Perceptron;
  2. MLP treinada por gradiente descendente em lote (vetorizado);
  3. pesos iniciais uniformes em ±1/√(nº de entradas);
  4. normalização min-max para [−1, 1] calculada só com o treino de cada rodada;
  5. desempate por argmax do potencial u no Perceptron multiclasse;
  6. BLAS com uma única thread (só desempenho);
  7. na topologia superdimensionada, ε = 0 e 15000 épocas, para que o overfitting fique visível.
