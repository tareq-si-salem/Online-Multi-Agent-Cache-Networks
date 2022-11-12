eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics

B=5
T=50000
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T}
python main.py --custom_weights 1-2-3.5 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/2players-online-alphas-${B} --construct_utility_point_cloud --construct_pareto_front --external_disagreement_points 0.0-0.0 --record_offline_stats_only --time_horizon ${T}
python main.py --custom_weights 1-2-3.5 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk --cache_type lru --output res/2players-online-alphas-${B} --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname lru --time_horizon ${T} &
python main.py --custom_weights 1-2-3.5 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk --cache_type lfu --output res/2players-online-alphas-${B} --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname lfu --time_horizon ${T} &
for param in 0.0 1.0 2.0; do
  python main.py --alpha ${param} --custom_weights 1-2-3.5 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fairslotted --output res/2players-online-alphas-${B} --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname fairslotted_${param} --time_horizon ${T} &
  python main.py --umin_umax 0.1-1.0 --alpha ${param} --custom_weights 1-2-3.5 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/2players-online-alphas-${B} --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname fair_${param} --time_horizon ${T} &
done
