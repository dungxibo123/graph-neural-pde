

datasets = ["Cora", "Citeseer","Pubmed", "CoauthorCS"]
depths = [567,678,789]
step_sizes = [0.9122304]
norm_types = ["norm"]
alphas = [0.0173192*i for i in range(3,14)]
truncate_coeffs = [0.3]




for dataset in datasets:
  f = open(f"run_{dataset}.sh", "w")
  for alpha in alphas:
    for step_size in step_sizes:
      for norm_type in norm_types:
        for t_co in truncate_coeffs:
          for depth in depths:
            f.write(f"python3 run_grand_ex.py --dataset {dataset} --epoch 120 --depth {depth} --step_size {step_size} --discritize_type {norm_type} --norm_exp {alpha} --run_time A10_FPT_Cloud_second --truncate_coeff {t_co} --truncate_norm --post_group_name ex_10_more_deep\n")
  f.close()
