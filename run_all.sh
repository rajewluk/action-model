#!/bin/bash

nohup python3 run.py plan-fix-fix-no_cl.cfg &
nohup python3 run.py plan-fix-fix-cl.cfg &
nohup python3 run.py plan-fix-fix-joint.cfg &

nohup python3 run.py plan-fix-rnd-no_cl.cfg &
nohup python3 run.py plan-fix-rnd-cl.cfg &
nohup python3 run.py plan-fix-rnd-joint.cfg &

exit 0
