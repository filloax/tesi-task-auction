#!/bin/bash

echo "Testing with agent amounts from 3 to 25"

echo "agents num,avg CBAA opt diff(%),avg CBBA opt diff(%),avg CBAA time(ms),avg CBBA time(ms),avg CBAA iterations,avg CBBA iterations" > compare_results.csv

local_path=$(realpath .)

for n_agents in $(seq 3 2 25); do
	echo "Testing with $n_agents agents:"
	TMP=$(mktemp)
	# TMP="$local_path/tmp.txt"
	python3 "$local_path/compare_algos.py" -p -N $n_agents -r 30 $* > "$TMP" &
	compare_pid=$!
	echo "Started with pid: $compare_pid"
	tail -f -n +0 "$TMP" &
	tail_pid=$!
	wait $compare_pid
	kill $tail_pid

	# Avg. Optimized	c * x: 156.52
	# Avg. CBAA	c * x: 152.84	diff: 2.35%	time: 9ms	iterations: 20.37
	# Avg. CBBA	c * x: 84.09	diff: 46.27%	time: 13ms	iterations: 43.77
	diff_cbaa_avg=$(cat "$TMP" | grep "Avg. CBAA" | cut -d: -f3 | sed 's/%\ttime//' | sed 's/ //')
	diff_cbba_avg=$(cat "$TMP" | grep "Avg. CBBA" | cut -d: -f3 | sed 's/%\ttime//' | sed 's/ //')
	time_cbaa_avg=$(cat "$TMP" | grep "Avg. CBAA" | cut -d: -f4 | sed 's/ms\titerations//' | sed 's/ //')
	time_cbba_avg=$(cat "$TMP" | grep "Avg. CBBA" | cut -d: -f4 | sed 's/ms\titerations//' | sed 's/ //')
	it_cbaa_avg=$(cat "$TMP" | grep "Avg. CBAA" | cut -d: -f5 | sed 's/ //')
	it_cbba_avg=$(cat "$TMP" | grep "Avg. CBBA" | cut -d: -f5 | sed 's/ //')

	echo "$n_agents,$diff_cbaa_avg,$diff_cbba_avg,$time_cbaa_avg,$time_cbba_avg,$it_cbaa_avg,$it_cbba_avg" >> "$local_path/compare_results.csv"
	echo "Done with results:"
	echo "$n_agents,$diff_cbaa_avg,$diff_cbba_avg,$time_cbaa_avg,$time_cbba_avg,$it_cbaa_avg,$it_cbba_avg"
	rm "$TMP"
done
