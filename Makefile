processed_data: # run scr/features/build_features.py
	@echo Executing build_features.py file
	@python3 scr/features/build_features.py
	@echo data/processed directory was created !

processed_data_npy: # run scr/features/build_features.py
	@echo Executing build_features_np.py file
	@python3 scr/features/build_features_np.py
	@echo data/processed directory was created !