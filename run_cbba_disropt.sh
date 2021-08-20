#!/bin/bash

while getopts "n:t:L:v" opt; do
  case $opt in
    n) num="$OPTARG"
    ;;
    t) tasks="$OPTARG"
    ;;
    L) agent_tasks="$OPTARG"
    ;;
    v) verbose="-v "
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

tmp=$(mktemp)
python3 task_positions.py -n "$num" -t "$tasks"
mpirun -np "$num" python3 task_bundle_disropt.py -L "$agent_tasks" $verbose | tee $tmp

echo ""
echo ""
echo "Task matrix:"
cat $tmp | grep "Assigned tasks" -A 1 | grep "\["