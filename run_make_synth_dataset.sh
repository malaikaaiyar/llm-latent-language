#!/bin/bash

python3 make_synth_dataset.py --model_name gemma-2b --save_dir data/synth_gemma_2b --batch_size 128
python3 make_synth_dataset.py --model_name meta-llama/Llama-2-7b-hf --save_dir data/synth_llama_2_7b_new --batch_size 128
