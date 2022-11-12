eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
B=100
for param in 2.5 2.75 3.0 3.25 3.5 3.75 4.0; do
  for alpha in 0.01 1.0 1.5 2.0; do
    python tracegenerator.py --batch_min_size 100 --batch_max_size 100 --time_horizon 10000 --distribution_roll 0 --zipfs_exponent 1.2
    python main.py --alpha $alpha --custom_weights 1-2-$param --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/pareto_explore_retrievalcosts --construct_utility_point_cloud --construct_pareto_front --external_disagreement_points 0.0-0.0 --record_offline_stats_only --experiment_name 1-2-$param+$alpha &
  done
done
