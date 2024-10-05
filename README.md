# Installation

Set up a python environment and run
`pip install -r requirements.txt`

# Usage 

## Translation and intervention

`python3 main_rejection_experiment.py --model_name meta-llama/Llama-2-7b-hf`

Will save results in `out_iclr` folder. 

## Plotting

`python plot_rejection_experiment.py --model_name meta-llama/Llama-2-7b-hf`

* Plots the probability assigned to target word during translation from source word, for each pair of languages.
* Plots the probability assigned to target word during translation from source word, if we intervene and project out unembedding vectors for *the correct* word in the latent language.
* Plots the probability assigned to target word during translation from source word, if we intervene and project out unembedding vectors for *a random* word in the latent language, and the correct word in the target language.

# Acknowledgements

Starting point of this repo was [Nina Rimsky's](https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.ipynb) Llama-2 wrapper.

# Citation
```
@article{wendler2024llamas,
  title={Do Llamas Work in English? On the Latent Language of Multilingual Transformers},
  author={Wendler, Chris and Veselovsky, Veniamin and Monea, Giovanni and West, Robert},
  journal={arXiv preprint arXiv:2402.10588},
  year={2024}
}
```
