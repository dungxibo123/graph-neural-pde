import os


import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

start = time.time()
os.system(f"sh run_file_{args.dataset}.sh")
t = time.time() - start
f = open("log/run_time.log", "w")
f.write(f"Time for sets contains 96 runs with {args.dataset} dataset on A10 servers is: {t}")
f.close()
