# Classificação de Uso do Solo

> Trabalho da disciplina de Sensoriamento Remoto (GEO05038) 2022/1. Código reorganizado com `uv`.

Benchmark de modelos tradicionais e profundos para classificação de uso do solo
no EuroSAT-RGB (Sentinel-2, 10 classes, 27 000 patches), com análise de
explicabilidade.

| Família     | Modelo        | Entrada                                                                 |
|-------------|---------------|-------------------------------------------------------------------------|
| Tradicional | Random Forest | features manuais (histogramas HSV, textura GLCM, momentos de Hu, VARI)  |
| Tradicional | XGBoost       | mesmas features manuais                                                 |
| CNN         | ResNet-18     | 224×224 RGB, pré-treinada no ImageNet                                   |

Todos os modelos compartilham as mesmas partições estratificadas de 5 folds.
As métricas são reportadas como média ± desvio com IC bootstrap a 95%; as
diferenças par-a-par entre modelos são avaliadas com bootstrap pareado e
McNemar (correção de Holm). O pipeline também calcula importância de features
/ SHAP para os modelos tradicionais e Grad-CAM / sensibilidade por oclusão
para a ResNet-18.

## Requisitos

[`uv`](https://docs.astral.sh/uv/) para gerenciamento de dependências.
Instale com:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Início rápido

```bash
uv sync                            # resolve + instala no .venv (apenas na 1ª vez)
bash scripts/run_all.sh            # estudo completo (~6–10 h em uma GPU)
# ou, para um smoke-test:
bash scripts/run_all.sh --fast     # < 10 min
```

`scripts/run_all.sh` chama cada etapa via `uv run`, então o venv do projeto é
usado automaticamente. As figuras (matrizes de confusão, F1 por classe,
calibração, curvas de treino, importância de features, SHAP, Grad-CAM,
oclusão) são gravadas em `results/figures/`; as métricas agregadas em
`results/metrics/`.

## Comandos úteis

```bash
uv run python scripts/06_evaluate_all.py   # apenas reagrega métricas
uv add some-package                        # adiciona dependência de runtime
uv lock --upgrade                          # atualiza o lockfile
```

## Estrutura do projeto

```
src/land_classification/   pacote (data, models, training, evaluation, explain)
scripts/                   pontos de entrada numerados (01_… → 09_…) + run_all.sh
configs/default.yaml       fonte única de verdade para hiperparâmetros
results/                   métricas, figuras, predições, checkpoints, explicações (no .gitignore)
```
