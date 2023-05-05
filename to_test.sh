./run_experiment.sh python3 train.py +experiment=camelyon16 +model=distance_aware_self_attention settings.output_size=32 name=1head
./run_experiment.sh python3 train.py +experiment=camelyon16 +model=distance_aware_self_attention settings.output_size=32 settings.hidden_dim=16 name=1head_hidden16
./run_experiment.sh python3 train.py +experiment=camelyon16 +model=distance_aware_self_attention settings.output_size=16 settings.self_attention.num_heads=2 model.classifier.feature_size=32 name=2heads
./run_experiment.sh python3 train.py +experiment=camelyon16 +model=distance_aware_self_attention settings.output_size=16 settings.self_attention.num_heads=2 model.classifier.feature_size=32 settings.hidden_dim=16 name=2heads_hidden16

python3 train.py +experiment=camelyon16 +model=distance_aware_self_attention_2layers settings.output_size=16 name=2layers device=1 name=2layers