

datasets = ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]
depths = [1,2,3,4,14,28,56]
step_sizes = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
norm_types = ["norm", "frobenius_norm"]
alphas = [0.5, 1.0, 1.5, 2.0, 2.5]



for dataset in datasets:
	f = open(f"run_{dataset}.sh", "w")
	for depth in depths:
		for step_size in step_sizes:
			for norm_type in norm_types:
				for alpha in alphas:
					f.write(f"python3 run_grand_ex.py --dataset {dataset} --depth {depth} --step_size {step_size} --discritize_type {norm_type} --norm_exp {alpha} --run_time A10_FPT_Cloud --truncate_norm\n")
	f.close()

