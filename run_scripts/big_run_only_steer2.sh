#!/bin/bash

# Manually adjusted logarithmically spaced values with more density from 0.5 to 5
steer_coeffs=(0.5 1.0 2.0 4.0 5.0 6.0 8.0 10 15)

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/fr_de_zh_${coeff_no_decimal}.log --src_lang fr --latent_lang de --dest_lang zh --steer_scale_coeff $coeff --intervention_correct_latent_space True
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/zh_de_fr_${coeff_no_decimal}.log --src_lang zh --latent_lang de --dest_lang fr --steer_scale_coeff $coeff --intervention_correct_latent_space True
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/fr_enalt_zh_${coeff_no_decimal}.log --src_lang fr --latent_lang en --dest_lang zh --steer_scale_coeff $coeff --intervention_correct_latent_space False
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/zh_enalt_fr_${coeff_no_decimal}.log --src_lang zh --latent_lang en --dest_lang fr --steer_scale_coeff $coeff --intervention_correct_latent_space False
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/fr_en_zh_${coeff_no_decimal}.log --src_lang fr --latent_lang en --dest_lang zh --steer_scale_coeff $coeff --intervention_correct_latent_space True
done

for coeff in "${steer_coeffs[@]}"; do
  coeff_no_decimal=${coeff//./}
  python3 llama_experiment_template.py --intervention_func hook_only_new_subspace --log_file out/steer_28_May/zh_en_fr_${coeff_no_decimal}.log --src_lang zh --latent_lang en --dest_lang fr --steer_scale_coeff $coeff --intervention_correct_latent_space True
done
