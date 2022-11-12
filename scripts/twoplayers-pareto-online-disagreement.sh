eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
for B in 1 50; do
  python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon 10000
  python main.py --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/2players-1-$B --construct_utility_point_cloud --construct_pareto_front --external_disagreement_points 0.0-0.0 --record_offline_stats_only
  python main.py --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type lru --output res/2players-1-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname lru --time_horizon 10000 &
  python main.py --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type lfu --output res/2players-1-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-0.0 --experiment_subname lfu --time_horizon 10000 &
  for param in 0.0 0.5 0.7 0.75; do
    python main.py --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fairslotted --output res/2players-1-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-${param} --experiment_subname fairslotted_${param} --time_horizon 10000 &
    python main.py --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/2players-1-$B --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --external_disagreement_points 0.0-${param} --experiment_subname fair_${param} --time_horizon 10000 &
  done
done
