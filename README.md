# QAOANeural
## My first attempt with neural networks via Numpy to predict QAOA quantum circuits.
The goal of this codebase is to test how neural networks predict parameters for QAOA quantum circuits.  
  
## Outline of steps taken:  
1. Generated numerous QAOA instances using the generate_ising.py functions with various parameters.  
2. Optimise the QAOA parameters via traditional methods to generate known outputs for neural network.  
3. The numpified inputs and outputs are saved under the folder 'datasets' in csv format.  
4. Trained some simple neural networks on the datasets, saved in "trained_networks" folder.
5. Analysed datasets as well as trained neural networks in Jupyter notebooks for visualisation.  
  
### Other things to note: 
- The Jupyter notebooks are used to mainly visualize the results. Most of the technical stuff are in the python scripts.  
- I use HPC to complete independent batch jobs where possible, most notably in the training of the many network configurations in Step 4.  
- This was done by calling `python train_neural_network.py $args` on bash/sbatch scripts, where `$args` define the many different network configurations.  
- Steps 1 and 2 was also run in batches on HPC for the samples in the training and validation datasets. 