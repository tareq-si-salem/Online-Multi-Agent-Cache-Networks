eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics

B=100
T=5000
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T}
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 0.8
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 0.6
alpha=3.0
python main.py --time_horizon ${T} --umin_umax 0.1-1.0 --alpha $alpha --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type fair --output res/2players-topology-geant --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players 3
python main.py --time_horizon ${T} --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type lru --output res/2players-topology-geant --graph_type geant --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --experiment_subname lru --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --players 3 &
python main.py --time_horizon ${T} --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type lfu --output res/2players-topology-geant --graph_type geant --construct_utility_point_cloud --construct_pareto_front --cached_offline_results --experiment_subname lfu --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --players 3 &
for param in 1.0 3.0; do
  python main.py --time_horizon ${T} --umin_umax 0.1-1.0 --alpha ${param} --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type fair --output res/2players-topology-geant --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --cached_offline_results --experiment_subname fair_${param} --players 3 &
  python main.py --time_horizon ${T} --umin_umax 0.1-1.0 --alpha ${param} --traces traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_1.2_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.8_roll_0.pk-traces/trace_catalog_20_T_${T}_B_${B}_${B}_s_0.6_roll_0.pk --cache_type fairslotted --output res/2players-topology-geant --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --cached_offline_results --experiment_subname fairslotted_${param} --players 3 &
done
