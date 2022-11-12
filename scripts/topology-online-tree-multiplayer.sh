eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics

B=50
T=5000
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T}
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 0.6
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 0.8
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 1.0
for players in 2 3 4; do
  querynodes=$(expr 13 / $players)
  alpha=1.0
  python main.py --time_horizon ${T} --umin_umax 0.1-1.0 --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type fair --output res/2players-topology-tree-multiplayer-${players} --graph_type balanced_tree --graph_size 8 --graph_degree 3 --query_nodes $querynodes --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players ${players} --custom_weights 9.237721826711306-6.025995038712244-9.223277007331232-3.7925575799529074-2.3610020660719675-1.6219179992471262-4.828852288827125-2.346378180450507-1.3709833735205916-1.386865507333856-4.389977465389839-3.4149041254675643
  for param in 1.0 2.0 3.0; do
    python main.py --time_horizon ${T} --umin_umax 0.1-1.0 --alpha ${param} --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type fair --output res/2players-topology-tree-multiplayer-${players} --graph_type balanced_tree --graph_size 8 --graph_degree 3 --query_nodes $querynodes --min_capacity 1 --max_capacity 5 --cached_offline_results --experiment_subname fair_${param} --players ${players} --custom_weights 9.237721826711306-6.025995038712244-9.223277007331232-3.7925575799529074-2.3610020660719675-1.6219179992471262-4.828852288827125-2.346378180450507-1.3709833735205916-1.386865507333856-4.389977465389839-3.4149041254675643 &
  done
done
