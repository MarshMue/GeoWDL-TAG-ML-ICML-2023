# clear queue
ts -C

# setup ts to limit to a fixed number of processes
ts -S 20

for trial in {0..9}
do
  echo "Trial $trial"
  for n_atoms in 10 15 20 25 30 35 40 45 50
  do
    # need to do this to do have no locality
    echo "n_atoms $n_atoms, locality ninf"
    taskset --cpu-list 36-55 ts python atomicTestRun.py --n_atoms $n_atoms --locality "ninf" --trial $trial
    for locality in -4 -3 -2 -1 1 2 3 4
    do
      echo "n_atoms $n_atoms, locality $locality"
      taskset --cpu-list 36-55 ts python atomicTestRun.py --n_atoms $n_atoms --locality $locality --trial $trial
    done
  done
done