#!/bin/bash
source myvenv/bin/activate
./jobs/grad_o_inf.sh
./jobs/o_inf.sh "red"
./jobs/o_inf.sh "syn"
./jobs/o_inf.sh "mix"