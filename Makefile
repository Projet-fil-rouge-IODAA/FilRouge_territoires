processed_data: # run scr/features/build_features.py
	@echo "What type of pixel sample do you wish to use? (big, small)"
	@read type_of_pixel_sample
	@echo Executing build_features.py file
	@python3 scr/features/build_features.py $(type_of_pixel_sample)
	@echo data/processed directory was created !