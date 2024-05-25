#!/bin/bash

# Manually adjusted logarithmically spaced values with more density from 0.5 to 5
steer_coeffs=(0.2 0.5 0.9 1.0 1.2 1.5 2.0 3.0 4.0 5.0)

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/fr_de_zh_${coeff_no_decimal}.log --src_lang fr --latent_lang de --dest_lang zh --steer_scale_coeff $coeff 
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/zh_de_fr_${coeff_no_decimal}.log --src_lang zh --latent_lang de --dest_lang fr --steer_scale_coeff $coeff
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/fr_enalt_zh_${coeff_no_decimal}.log --src_lang fr --latent_lang en --dest_lang zh --steer_scale_coeff $coeff --intervention_correct_latent_space False
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/hook_only_new_subspace/zh_enalt_fr_${coeff_no_decimal}.log --src_lang zh --latent_lang en --dest_lang fr --steer_scale_coeff $coeff --intervention_correct_latent_space False
done
