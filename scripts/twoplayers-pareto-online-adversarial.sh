eval "$(conda shell.bash hook)"
# Simulate instance
conda activate acm_sigmetrics #Activate environment acm_sigmetrics
B=50
T=5000
for exponent in 1.2; do
  # Generate traces
  python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 1.2 --adversarial_1 --shuffle
  python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 1.2 --adversarial_2
  # Run processes in parallel for alpha = 3, and for policies fair (OHF) and fairslotted (OSF).
  for param in 3.0; do
    python main.py --alpha ${param} --custom_weights 1-2-2 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0_adv_1.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_${exponent}_roll_0_adv_2.pk --cache_type fairslotted --output res/2players-online-adversarial-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname fairslotted_${param} --time_horizon $T &
    python main.py --umin_umax 0.1-1.0 --alpha ${param} --custom_weights 1-2-2 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0_adv_1.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_${exponent}_roll_0_adv_2.pk --cache_type fair --output res/2players-online-adversarial-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname fair_${param} --time_horizon $T &
  done
done
