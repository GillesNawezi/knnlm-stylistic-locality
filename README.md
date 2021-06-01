# Capturing Structural Locality in Non-parametric Language Models

This repository is a fork of the [knnlm](https://github.com/urvashik/knnlm) repository and the exact commit that this code is based on can be found [here](https://github.com/pytorch/fairseq/tree/6a5181509aa1fa7d260985157e77211753da544b). Please use the exact commit page to determine software requirements for using this code. This README will be updated once the code has been merged into Fairseq.
This repository is heavily based on [fairseq](https://github.com/pytorch/fairseq).


## Dependencies

Before starting, make sure you install Fairseq (after pulling the code, from the project directory) and [FAISS](https://github.com/facebookresearch/faiss/wiki):
```bash
pip install --editable .

pip install faiss
```

## Data Preparation

Java: https://zenodo.org/record/3628665
Wikitext: https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

Extract locality features: `locality_features`.

## Experiments
For Wikitext-103 experiments, follow `wikitext_knn_lm.sh`.

For Java experiments, follow `bigcode_knn_lm_dynamic.sh`.

### A Note about Hardware

If your hardware constraints make this too slow, you can run it without using full precision keys by adding two flags: `--no-load-keys` and `--knn-sim-func "do_not_recomp_l2"`. This uses the quantized versions of keys stored within the FAISS index. You can make things faster by reducing the value of the `probe` (the number of clusters FAISS checks for neighbors) at the cost of performance. You can also try reducing the number of neighbors `k`.
