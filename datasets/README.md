# Dataset Usage


In this folder, we put all the datasets well organized in folders.
The __first level__ denotes which is the source (i.e., darias, rl_bench, or human_dataset) while the second level indicates
the task involved.

Then, in each folder, there is a collection of files with

 - ```trajectory_n.npz``` which indicates the trajectory n. The file must be loadable by ```NamedTrajectory```
 - ```context_n.npy``` which indicates the context n (if the task has a context). The file must contain a single ```numpy``` array.
 
 
It is conveniente to have also the ```RLBench``` task saved as the simulation is very time-consuming. 
 