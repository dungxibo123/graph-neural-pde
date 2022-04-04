

datasets = ["Cora", "Citeseer"]
depths = [3,4]
step_sizes = [0.000001, 0.000005, 0.00001, 0.00005]
norm_types = ["norm", "frobenius_norm"]
alphas = [4.0, 3.5]
truncate_coeffs = [2.0, 2.5]




for dataset in datasets:
  f = open(f"run_{dataset}.sh", "w")
  for depth in depths:
    for step_size in step_sizes:
      for norm_type in norm_types:
        for alpha in alphas:
          for t_co in truncate_coeffs:
            f.write(f"python3 run_grand_ex.py --dataset {dataset} --depth {depth} --step_size {step_size} --discritize_type {norm_type} --norm_exp {alpha} --run_time A10_FPT_Cloud_second --truncate_coeff {t_co} --truncate_norm\n")
  f.close()
