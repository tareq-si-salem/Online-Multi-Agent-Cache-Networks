eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
B=1
for param in 1.2 1.0 0.8 0.6; do
  for alpha in 0.5 1.0 2.0 4.0; do
    python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon 10000 --distribution_roll 0 --zipfs_exponent $param
    python main.py --alpha $alpha --custom_weights 1-2-2 --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_10000_B_${B}_${B}_s_${param}_roll_0.pk --cache_type fair --output res/pareto_explore_exponents --construct_utility_point_cloud --construct_pareto_front --external_disagreement_points 0.0-0.0 --record_offline_stats_only --experiment_name $param+$alpha &
  done
done
