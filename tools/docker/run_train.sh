#!/bin/bash

attempt=0
while true; do
  if [[ -n $(ls /mnt) ]]; then
    break
  else
    if (( attempt > 20 )); then
      echo "Timeout. Fail to start container."
      exit 2
    fi
    attempt=$((attempt+1))
    sleep 3
  fi
done

cd /mnt
python $1
