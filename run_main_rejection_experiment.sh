#!/bin/bash
echo "running..."
python3 main_rejection_experiment.py --model_name meta-llama/Llama-2-7b-hf
python3 main_rejection_experiment.py --model_name meta-llama/Llama-2-13b-hf
python3 main_rejection_experiment.py --model_name google/gemma-2-2b
python3 main_rejection_experiment.py --model_name google/gemma-2-9b


# Transformer lens doesn't support Mistral-v0.3, and x0.1 is too stupid for this task
#python3 main_rejection_experiment.py --model_name mistralai/Mistral-7B-v0.3 
