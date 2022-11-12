eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
B=1
for param in 0 0.25 0.5 0.6 0.7 0.75; do
  for alpha in 1.0; do
    python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon 10000 --distribution_roll 0 --zipfs_exponent 1.2
    python main.py --alpha $alpha --custom_weights 1-2-2 --traces traces/trace_catalog_20_T_10000_B_${B}_${B}_s_1.2_roll_0.pk --cache_type fair --output res/pareto_explore_disagreements --construct_utility_point_cloud --construct_pareto_front --external_disagreement_points 0.0-$param --record_offline_stats_only --experiment_name $param+$alpha &
  done
done
