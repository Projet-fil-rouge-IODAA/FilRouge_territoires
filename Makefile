reset_data:
	@echo Cleaning data directory...
	@echo Removing data/processed directory...
	@rm -rf data/processed
	@echo Removing data/cropped directory...
	@rm -rf data/cropped

crop_data: # run scr/features/crop_images.py
	@echo Executing crop_images.py file
	@python3 scr/features/crop_images.py
	@echo data/cropped directory was created !

processed_data: # run scr/features/build_features.py
	@echo Executing build_features.py file
	@python3 scr/features/build_features.py
	@echo data/processed directory was created !

processed_data_npy: # run scr/features/build_features.py
	@echo Executing build_features_np.py file
	@python3 scr/features/build_features_np.py
	@echo data/processed directory was created !

create_data: reset_data crop_data processed_data
