# QAOANeural
## My first attempt with neural networks via Numpy to define QAOA quantum circuits.

The goal of this codebase is to test how neural networks predict QAOA quantum circuits.  
I generated QAOA instances using the generate_ising.py functions.  
I created QAOA circuits for these instances and pre-optimise the QAOA parameters.  
The raw datasets are saved in json format of labeled QAOA instances.  
The numpified inputs and outputs are saved under the folder 'datasets' in csv format.  
The neural network inputs are the Hamiltonian coefficients defining the QAOA circuit.  
The network outputs the predicted QAOA parameters for the instance Hamiltonian.  
