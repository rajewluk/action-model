#/bin/bash

echo "CAMPAIGN PID: $$"
SEED=40
LAST_SEED=80

until [ $SEED -ge $LAST_SEED ]
do
  ((SEED=SEED+1))
  sed -ir "s/^[#]*\s*seed-base =.*/seed-base = $SEED/" config.cfg

  mkdir -p campaign_results
  cd campaign_results
  RES_DIR="seed_"$SEED
  if [ ! -d $RES_DIR ]; then
    mkdir -p $RES_DIR;
    echo "RUN SEED:"`sed -rn 's/^seed-base =([^\n]+)$/\1/p' ../config.cfg`
    cd ..
    cp plan_base/* ./
    ./run_all.sh
    until [ $? -ne 0 ]
    do
      sleep 15
      echo "LEFT: "`pgrep -l -f run.py | wc -l`" - "`date +"%T"`
      pgrep -l -f run.py > /dev/null 2>&1
    done
    cd campaign_results
    mv ../results/* $RES_DIR/
  else
    echo "SKIP SEED: "$SEED
  fi
  cd ..
done

exit 0




#python3 times.py
