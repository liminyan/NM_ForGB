#!/bin/sh
#SBATCH -J drv_31_1

srun -n 20 /ddnstor/xuewei/3rdParty/anaconda3/bin/python3.7 -W ignore main.py v 31 1
