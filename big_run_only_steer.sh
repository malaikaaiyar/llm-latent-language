#!/bin/bash

# Manually adjusted logarithmically spaced values with more density from 0.5 to 5
steer_coeffs=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9 1.2 1.6 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

for coeff in "${steer_coeffs[@]}"; do
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/fr_en_zh --src_lang fr --latent_lang en --dest_lang zh --steer_scale_coeff $coeff
done

for coeff in "${steer_coeffs[@]}"; do
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/fr_en_zh --src_lang zh --latent_lang en --dest_lang fr --steer_scale_coeff $coeff
done