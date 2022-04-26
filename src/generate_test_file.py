



f = open("run_test_file.sh", "w")
depths = [i for i in range(1,800,15)]
step_sizes = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,9e-1, 0.99, 1.0]
for ss in step_sizes:
    for depth in depths:
        f.write(f"python3 grand_discritized.py --depth {depth} --step_size {ss}\n")


