> Can hand-built features keep up with a neural network? A reproducible benchmark on
[EuroSAT-RGB](https://github.com/phelber/EuroSAT) (27,000 Sentinel-2 patches, 10 land-use classes),
with a look inside each model.

| Family      | Model         | Input                                                                    |
|-------------|---------------|--------------------------------------------------------------------------|
| Traditional | Random Forest | 87 hand-crafted features (HSV histograms, GLCM texture, Hu moments, VARI, ...) |
| Traditional | XGBoost       | the same 87 features                                                     |
| CNN         | ResNet-18     | 224×224 RGB image, pretrained on ImageNet                                |

All models share the same stratified 5-fold split. Differences are tested with a paired bootstrap and
McNemar's test (Holm correction). Explanations: feature importance and SHAP for the tree models,
Grad-CAM and occlusion sensitivity for the ResNet.

## Results

5-fold cross-validation, mean ± standard deviation across folds:

| Metric                     | Random Forest | XGBoost      | ResNet-18    |
|----------------------------|---------------|--------------|--------------|
| Accuracy                   | 86.93% ± 0.48 | 90.32% ± 0.44 | **98.96% ± 0.22** |
| Macro F1                   | 86.08% ± 0.54 | 89.75% ± 0.51 | **98.92% ± 0.22** |
| Calibration error (ECE) ↓  | 10.96% ± 0.54 | **1.98% ± 0.67** | 4.63% ± 0.09 |
| Training time per fold     | 4 s (CPU)     | 25 s (CPU)   | 10.4 min (GPU) |

Highway is the hardest class for the tree models (F1 63.1% and 73.2%) and easy for the ResNet (99.2%).
The full write-up, with figures, is on the project page.

## Reproduce

You only need [mise](https://mise.jdx.dev/). It installs the pinned Python 3.12 and uv; the pipeline downloads
the dataset itself (95 MB, checksum-verified).

```bash
git clone https://github.com/gstvschlz/land-classification
cd land-classification
mise trust             # allow the repo's mise.toml
mise install           # pins Python 3.12 and uv
mise run setup         # installs the locked dependencies
mise run smoke         # end-to-end check on a subset, a few minutes
mise run all           # the full study
```

The full study took about 80 minutes on a laptop with an RTX 5090 GPU (ResNet-18 is most of it). Without a GPU
expect many hours. Windows and Linux get CUDA 12.8 builds of PyTorch; macOS uses the standard PyPI build (untested here).

`smoke` and `all` write to `results/`, which overwrites the committed figures in `results/figures/`.
Restore them with `git checkout results/figures`.

### Tasks

| Task                            | What it does                                        |
|---------------------------------|-----------------------------------------------------|
| `mise run setup`                | Create `.venv` and install the locked dependencies  |
| `mise run download`             | Fetch EuroSAT, write the 5-fold split               |
| `mise run features`             | Compute the 87 hand-crafted features               |
| `mise run train-traditional`    | Random Forest and XGBoost (`-- --fast` for a quick run) |
| `mise run train-resnet`         | Fine-tune ResNet-18 (`-- --fast` for a quick run)   |
| `mise run explain-traditional`  | Feature importance and SHAP                         |
| `mise run explain-cnn`          | Grad-CAM and occlusion maps                         |
| `mise run evaluate`             | Aggregate metrics, significance tests, figures      |
| `mise run all`                  | All of the above, in order                          |
| `mise run smoke`                | The same stages with `--fast`                       |
| `mise run check`                | Sanity-check the feature extractor                  |
| `mise run site`                 | Preview the project page at <http://localhost:8000> |

Outputs: metrics in `results/metrics/`, figures in `results/figures/`, explanations in `results/explanations/`.

## Layout

```
src/land_classification/   package: data, models, training, evaluation, explain
scripts/                   numbered entry points (01_... to 09_...)
configs/default.yaml       every hyperparameter and the random seed
tests/check_features.py    feature-extractor sanity check
results/figures/           figures used by the project page
site/                      GitHub Pages source (deployed by .github/workflows/pages.yml)
```

## Notes

- The numbers above come from one run of `mise run all` with `configs/default.yaml`. GPU kernels are not
  bit-for-bit deterministic, so expect small differences in the last digit.
- The dataset is EuroSAT: Helber, Bischke, Dengel, Borth, *EuroSAT: A Novel Dataset and Deep Learning Benchmark
  for Land Use and Land Cover Classification*, IEEE JSTARS 12(7), 2019.

## License

MIT
