# QAOANeural
## My first attempt with neural networks via Numpy to predict QAOA quantum circuits.

The goal of this codebase is to test how neural networks predict parameters for QAOA quantum circuits. 
Outline of steps:  
- I generated QAOA instances using the generate_ising.py functions with various parameters.  
- Optimise the QAOA parameters via traditional methods to generate known outputs for neural network.  
- The numpified inputs and outputs are saved under the folder 'datasets' in csv format.  
- Trained some simple neural networks on the datasets, saved in "trained_networks" folder.
- Analysed datasets as well as trained neural networks in Jupyter notebooks for visualisation.
