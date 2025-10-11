In order to run the various experiments from TASK 2 of our report please follow these instructions:

In a command line interface run:
python main.py --experiment 1

--experiment : can be 1 or 2. 1 is a one-time experiment (figures 2 and 4 in report), 2 is the learning rate sensiytivity analysis.

The following command runs one experiment of twenty 100-epoch runs. By default the learning rate is 0.0025 and the initial weight multiplicator is 0.01. These two arguments can be freely chosen from the command line, for instance:
python main.py --experiment 1 --lr 0.01 --weight_mult 1

--lr : learning rate
--weight_mult : initial weight multiplicator

Finally, this command runs the sensitivity analysis on the learning rate:
python main.py --experiment 2