# QAOANeural
## My first attempt with neural networks via Numpy to predict QAOA quantum circuits.

The goal of this codebase is to test how neural networks predict parameters for QAOA quantum circuits.  
Outline of steps taken:  
- Generated QAOA instances using the generate_ising.py functions with various parameters.  
- Optimise the QAOA parameters via traditional methods to generate known outputs for neural network.  
- The numpified inputs and outputs are saved under the folder 'datasets' in csv format.  
- Trained some simple neural networks on the datasets, saved in "trained_networks" folder.
- Analysed datasets as well as trained neural networks in Jupyter notebooks for visualisation.  
  
### Other things to note: 
- I use HPC to complete independent batch training of multiple network configurations.  
- This was done by calling `python train_neural_network.py $args` on bash/sbatch scripts, where $args define the many different network configurations.  
- Same thing for generating QAOA instances to find the optimal output angles in batches for the many samples in the training and validation datasets. 