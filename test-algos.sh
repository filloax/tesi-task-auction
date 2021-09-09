#!/bin/bash

min="${1:-5}"
max="${2:-20}"
step="${3:-3}"
runs_each="${4:-20}"

echo "Testing with agent amounts from $min to $max"

echo "agents num;avg CBAA opt diff(%);avg CBBA opt diff(%);avg CBAA time(ms);avg CBBA time(ms);avg CBAA iterations;avg CBBA iterations;CBAA conflicts;CBBA conflicts" > compare_results.csv

local_path=$(realpath .)

for n_agents in $(seq "$min" "$step" "$max"); do
	echo "Testing with $n_agents agents:"
	TMP=$(mktemp)
	# TMP="$local_path/tmp.txt"
	python3 "$local_path/compare_algos.py" -p -T 0.1 -N $n_agents -r "$runs_each" > "$TMP" &
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
	cbaa_conflicts=$(cat "$TMP" | grep "WARNING: CBAA had " | awk '{ print $4 }')
	test -z "$cbaa_conflicts" && cbaa_conflicts="0"
	cbba_conflicts=$(cat "$TMP" | grep "WARNING: CBBA had " | awk '{ print $4 }')
	test -z "$cbba_conflicts" && cbba_conflicts="0"

	res="$n_agents;$diff_cbaa_avg;$diff_cbba_avg;$time_cbaa_avg;$time_cbba_avg;$it_cbaa_avg;$it_cbba_avg;$cbaa_conflicts;$cbba_conflicts"
	echo "$res" | tr '.' ',' >> "$local_path/compare_results.csv"
	echo "Done with results:"
	echo "$res"
	rm "$TMP"
done
