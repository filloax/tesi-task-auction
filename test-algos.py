import sys
import os
import tempfile
import re
from threading import Thread

from compare_algos import do_test

min=5
max=20
step=3
runs_each=20

out_path = "compare_results.csv"

if len(sys.argv) > 1:
	min = int(sys.argv[1])
	if len(sys.argv) > 2:
		max = int(sys.argv[2])
		if len(sys.argv) > 3:
			step = int(sys.argv[3])
			if len(sys.argv) > 4:
				runs_each = int(sys.argv[4])

print(f"Testing with agent amounts from {min} to {max}")

class DupOut:
	def __init__(self, stdout, file):
		self.stdout = stdout
		self.file = file

	def write(self, *args):
		self.stdout.write(*args)
		self.file.write(*args)

	def flush(self):
		self.stdout.flush()
		if not self.file.closed:
			self.file.flush()



def print_result(*args):
	with open(out_path, "a") as outfile:
		print(*args, file=outfile)

if os.path.exists(out_path):
	os.remove(out_path)

print_result("agents num;avg CBAA opt diff(%);avg CBBA opt diff(%);avg CBAA time(ms);avg CBBA time(ms);avg CBAA iterations;avg CBBA iterations;CBAA conflicts;CBBA conflicts")

local_path = os.path.abspath(__file__)

for n_agents in range(min, max + 1, step):
	print(f"Testing with {n_agents} agents:")
	(tmp_fd, tmp_path) = tempfile.mkstemp(text=True)
	stdout_bak = sys.stdout

	done1 = False

	try:
		with open(tmp_path, "w", encoding='utf-8') as tmp_file:
			sys.stdout = DupOut(sys.stdout, tmp_file)
			do_test(n_agents, runs_each, verbose=False, print_iter_progress=True, prog_update_time=0.1, test_same_score=False)
			sys.stdout.flush()
			sys.stdout = stdout_bak
			done1 = True
	finally:
		if not done1:
			os.close(tmp_fd)
			os.remove(tmp_path)


	# Avg. Optimized	c * x: 156.52
	# Avg. CBAA	c * x: 152.84	diff: 2.35%	time: 9ms	iterations: 20.37
	# Avg. CBBA	c * x: 84.09	diff: 46.27%	time: 13ms	iterations: 43.77
	diff_cbaa_avg = None
	diff_cbba_avg = None
	time_cbaa_avg = None
	time_cbba_avg = None
	it_cbaa_avg = None
	it_cbba_avg = None
	cbaa_conflicts = 0
	cbba_conflicts = 0
	try:
		with open(tmp_path, "r", encoding='utf-8') as tmp_file:
			for line in tmp_file:
				if "Avg. CBAA" in line:
					segs = line.split(": ")
					diff_cbaa_avg = round(float(segs[2].split("%")[0].strip()) / 100, 4)
					time_cbaa_avg = segs[3].split("ms")[0].strip()
					it_cbaa_avg = segs[4].strip()
				elif "Avg. CBBA" in line:
					segs = line.split(": ")
					diff_cbba_avg = round(float(segs[2].split("%")[0].strip()) / 100, 4)
					time_cbba_avg = segs[3].split("ms")[0].strip()
					it_cbba_avg = segs[4].strip()
				elif "WARNING: CBAA had " in line:
					cbaa_conflicts = line.split(" ")[3]
				elif "WARNING: CBBA had " in line:
					cbba_conflicts = line.split(" ")[3]
	finally:
		os.close(tmp_fd)
		os.remove(tmp_path)

	res=f"{n_agents};{diff_cbaa_avg};{diff_cbba_avg};{time_cbaa_avg};{time_cbba_avg};{it_cbaa_avg};{it_cbba_avg};{cbaa_conflicts};{cbba_conflicts}"
	print_result(res.replace(".", ","))
	print("Done with results:")
	print(res)
