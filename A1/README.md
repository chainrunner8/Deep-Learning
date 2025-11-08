In order to run the various experiments from TASK 2 of our report please follow these instructions:

In a command line interface run:
python task2.py --experiment (one of 1, 2, 3, 4)

--experiment 1 : runs the learning rate search.
--experiment 2 : runs the 24-class task.
--experiment 3 : runs the 720-class task.
--experiment 4 : runs the regression task.

At the moment of writing this, all of these commands were tested successfully. If you run into any problem replicating our experiments, please contact us.

Please note that while requirements.txt mentions torch==2.10.0.dev20251031+cu128, earlier torch builds should also work just fine.
The GPU of the author of task 2 simply required the nightly build.