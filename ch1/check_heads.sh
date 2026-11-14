#!/bin/bash
# Which corrector heads did a finished DIII-D run use?   check_heads.sh <run dir> <profile> <HEADS of the profile>
# Prints HEADS_CHECK OK only if the run recorded the profile's HEADS and wrote (v22c10) / did not write (v21) the four v2.2 head-group
# predictions. A file from a run that prints HEADS_CHECK FAILED was produced by the other recipe and must not be submitted.
RUN=${1:?run dir}; PROF=${2:?profile}; EXP=${3:?HEADS}
used=$(cat "$RUN/heads_used.txt" 2>/dev/null || echo none)
n=$(ls "$RUN"/cc/hp_c10[abcd].npy 2>/dev/null | wc -l | tr -d ' ')
want=0; [ "$EXP" = v22c10 ] && want=4
if [ "$used" = "$EXP" ] && [ "$n" = "$want" ]; then echo "HEADS_CHECK OK: profile $PROF, HEADS=$used, v2.2 head groups predicted: $n"
else echo "HEADS_CHECK FAILED: profile $PROF expects HEADS=$EXP ($want v2.2 head groups); the run recorded HEADS=$used and wrote $n - DO NOT SUBMIT THIS FILE"; exit 1; fi
