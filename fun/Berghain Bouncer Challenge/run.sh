#!/bin/zsh
echo "Running probability"
for i in $(seq 1 50); do
  python probability.py \
  | awk -F': ' '/^Total rejected:/ {print $2}'
done

echo "Running day2"
for i in $(seq 1 50); do
  python day2.py \
  --scenario 1 \
  | awk -F': ' '/^Total rejected:/ {print $2}'
done