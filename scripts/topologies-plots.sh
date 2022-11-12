eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
T=5000
B=100
python tracegenerator.py --batch_min_size $B --batch_max_size $B --time_horizon ${T} --zipfs_exponent 1.2
python main.py --output res/topology-tree/ --graph_type balanced_tree --graph_size 8 --query_nodes 4 --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players 3 --record_offline_stats_only --custom_weights 9.237721826711306-6.025995038712244-9.223277007331232-3.7925575799529074-2.3610020660719675-1.6219179992471262-4.828852288827125-2.346378180450507-1.3709833735205916-1.386865507333856-4.389977465389839-3.4149041254675643
python main.py --output res/topology-abilene/ --graph_type abilene --query_nodes 2 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players 3
python main.py --output res/topology-grid_2d/ --graph_type grid_2d --graph_size 8 --query_nodes 2 --repo_nodes 1 --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players 3
python main.py --output res/topology-geant/ --graph_type geant --query_nodes 3 --repo_nodes 2 --min_capacity 1 --max_capacity 5 --record_offline_stats_only --players 3
