

datasets = ["Cora", "Citeseer","Pubmed", "CoauthorCS"]
depths = [224,336]
step_sizes = [0.8921733]
norm_types = ["norm"]
alphas = [0.025 * i for i in range(1,10)]
truncate_coeffs = [0.04,0.05,0.07,0.1,0.12]




for dataset in datasets:
  f = open(f"run_{dataset}.sh", "w")
  for depth in depths:
    for step_size in step_sizes:
      for norm_type in norm_types:
        for alpha in alphas:
          for t_co in truncate_coeffs:
            f.write(f"python3 run_grand_ex.py --dataset {dataset} --depth {depth} --step_size {step_size} --discritize_type {norm_type} --norm_exp {alpha} --run_time A10_FPT_Cloud_second --truncate_coeff {t_co} --truncate_norm --post_group_name ex_08\n")
  f.close()
