#!/bin/bash

echo "Testing with agent amounts from 3 to 25"

echo "agents num,time(ms),average diff(%),exact results(%)" > test_results.csv

for n_agents in $(seq 3 2 25); do
	echo "Testing with $n_agents agents:"
	res=$(python3 taskAssignSeq.py $n_agents 100 | tail -n 4)
	time_taken=$(grep time <<< "$res" | cut -d: -f2 | tr -d '[:space]')
	avg_diff=$(grep "Average pct" <<< "$res" | cut -d: -f2 | tr -d '[:space]')
	pct_exact=$(grep "Pct of" <<< "$res" | cut -d: -f2 | tr -d '[:space]')
	echo "$n_agents,$time_taken,$avg_diff,$pct_exact" >> test_results.csv
	echo "$res"
done
