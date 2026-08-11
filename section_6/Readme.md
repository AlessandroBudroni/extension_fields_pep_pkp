# Section 6 experiments

## PEP-to-GIP reduction experiments

`PEP_to_GIP_reduction_experiment.py` performs experiments using
random self-orthogonal codes over fields of size `q = p^nu (nu > 1)`.
It measures:

- the probability that `sigma(G)*G^T` is invertible, where `sigma` is the `p`-power Frobenius
- the probability that solving GIP will also solve the underlying PEP, restricted to cases where `sigma(G)*G^T` is invertible

and produces plots for each of them against `p` or `nu`.

Run it with SageMath:

```sh
sage PEP_to_GIP_reduction_experiment.py \
  --p {field_characteristics} \
  --n {code_dimensions} \
  --nu {extension_degrees} \
  --nexp {number_of_experiments} \
  --j {number_of_threads} \
```

`field_characteristics`, `code_dimensions`, and `extension_degrees` must be specified as comma-separated integers without spaces. At least one of `field_characteristics` or `extension_degrees` should be a single integer; the one that is not will be automatically set as the x-axis for the graphs.

Each run will produce in the `data/` directory two plot previews for the aforementioned probabilities in `.png` format as well as corresponding data files in `.npy` and `.txt` formats.