eval "$(conda shell.bash hook)"
conda activate acm_sigmetrics
#=======================================
python plotter.py --ylim 0.5-0.8 --xlim 0.5-0.8 --task 2dplot --output ./out_figs/two_players_b_1.pdf --input ./res/2players-1-1/ &
python plotter.py --ylim 0.5-0.8 --xlim 0.5-0.8 --task 2dplot --output ./out_figs/two_players_b_50.pdf --input ./res/2players-1-50/ &
python plotter.py --ylim 0.5-0.8 --xlim 0.5-0.8 --task 2dplot --output ./out_figs/two_players_b_50-legend.pdf --input ./res/2players-1-50/ --legend &
#=======================================
python plotter.py --ylim 0.6-0.85 --xlim 0.25-0.5 --task 2dplot --output ./out_figs/2players-online-alphas-1.pdf --input ./res/2players-online-alphas-1/ &
python plotter.py --ylim 0.45-0.6 --xlim 0.45-0.85 --task 2dplot --output ./out_figs/2players-online-exponents-1.pdf --input ./res/2players-online-exponents-1/ &
python plotter.py --ylim 0.6-0.85 --xlim 0.25-0.5 --task 2dplot --output ./out_figs/2players-online-alphas-5.pdf --input ./res/2players-online-alphas-5/ &
python plotter.py --ylim 0.6-0.85 --xlim 0.25-0.5 --task 2dplot --output ./out_figs/2players-online-alphas-50.pdf --input ./res/2players-online-alphas-50/ &
python plotter.py --ylim 0.45-0.6 --xlim 0.45-0.85 --task 2dplot --output ./out_figs/2players-online-exponents-5.pdf --input ./res/2players-online-exponents-5/ &
python plotter.py --ylim 0.45-0.6 --xlim 0.45-0.85 --task 2dplot --output ./out_figs/2players-online-exponents-50.pdf --input ./res/2players-online-exponents-50/ &
#=======================================
python plotter.py --task barplot-multiplayer --output ./out_figs/2players-topology-tree-multiplayer-x-alpha-1.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 1 --policies fair
python plotter.py --task barplot-multiplayer --output ./out_figs/2players-topology-tree-multiplayer-x-alpha-2.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 2 --policies fair
python plotter.py --task barplot-multiplayer --output ./out_figs/2players-topology-tree-multiplayer-x-alpha-3.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 3 --policies fair
python plotter.py --task barplot-multiplayer --output ./out_figs/2players-topology-tree-multiplayer-x-alpha-legend.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 3 --policies fair --legend
python plotter.py --task barplot-multiplayer-pof --output ./out_figs/2players-topology-tree-multiplayer-x-alpha-POF.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 3 --policies fair
python plotter.py --task barplot-multiplayer-temp --output ./out_figs/2players-topology-tree-multiplayer-x-player-2.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 2 --player 2 --policies fair &
python plotter.py --task barplot-multiplayer-temp --output ./out_figs/2players-topology-tree-multiplayer-x-player-3.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 2 --player 3 --policies fair &
python plotter.py --task barplot-multiplayer-temp --output ./out_figs/2players-topology-tree-multiplayer-x-player-4.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 2 --player 4 --policies fair &
python plotter.py --task barplot-multiplayer-temp --output ./out_figs/2players-topology-tree-multiplayer-x-player-4-legend.pdf --input ./res/2players-topology-tree-multiplayer-x/ --params 2-3-4 --alpha 2 --player 4 --legend --policies fair &
#=======================================
python plotter.py --task 2dplotnopolicies --output ./out_figs/pareto_explore_retrievalcosts.pdf --input ./res/pareto_explore_retrievalcosts/ --sort_direction 0 &
python plotter.py --task 2dplotnopolicies --output ./out_figs/pareto_explore_exponents.pdf --input ./res/pareto_explore_exponents/ --sort_direction 1 &
#=======================================
python plotter.py --task barplot --output ./out_figs/2players-topology-tree.pdf --input ./res/2players-topology-tree/ &
python plotter.py --task barplot --output ./out_figs/2players-topology-2d_grid.pdf --input ./res/2players-topology-2d_grid/ &
python plotter.py --task barplot --output ./out_figs/2players-topology-geant.pdf --input ./res/2players-topology-geant/ &
python plotter.py --task barplot --output ./out_figs/2players-topology-abilene.pdf --input ./res/2players-topology-abilene/ &
python plotter.py --task barplot --output ./out_figs/2players-topology-topologies.pdf --input ./res/2players-topology-x/ --policies fair --alpha 0
#=======================================
python plotter.py --ylim 0.45-0.6 --xlim 0.45-0.85 --task 2dplot --output ./out_figs/2players-online-adversarial-50.pdf --input ./res/2players-online-adversarial-50/ --policies fair-fairslotted --legend --alpha 2.0
#=======================================
python plotter.py --task topology_draw
