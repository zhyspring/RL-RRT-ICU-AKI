# RL-RRT-ICU-AKI
the reinforcement learning in RRT operation optimation of ICU-AKI patients (in MIMIC and eICU)

## file control

### Pre content
the concepts of MIMIC database and the creation of MIMIC/eICU github project.

### the data process
firstly, we should find the AKI (who required AKI after going into the ICU) and AKI level by the KDIGO criterion.

#### v1_aki_level.sql
get the aki_flag and aki_level for each stay_id & charttime

get the max_aki_level during the ICU time for each stay_id

#### v3_action_status_seq
get the time and hr before ICU 24h and During ICU. With the Time granularity of 6h.

