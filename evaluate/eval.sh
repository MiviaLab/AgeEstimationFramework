#!/bin/bash
FROM=vggface2
TO=adience_shift
VARIANT=_DROPOUT_NOVAL

for f in ../fine_tuned_${TO}_from_${FROM}${VARIANT}/* ;
do
    echo $(basename $f) | awk 'BEGIN { FS = "_" } ; { print $4 }' | cut -c4-
    python3 eval_age_adience.py --path $f --out_path results_${FROM}_to_${FROM}${VARIANT}/
done
