# QAOANeural  
## A first attempt with neural networks via Numpy to predict QAOA quantum circuits.  
Two main goals for this codebase:  
1) Test how neural networks could predict parameters for QAOA quantum circuits.  
2) To code a neural network from scratch to understand their workings better.  
  
By using Git repos, the hope is that the results are easily reproducible.  
  
## Outline of steps taken to produce results:  
1. Generated numerous QAOA instances using the generate_ising.py functions with various parameters.  
2. Optimised QAOA parameters via traditional methods `scipy.optimize.minimize` to generate input-output pairs.  
3. Trained neural network with simple architectures using the generated inputs and outputs, saved in "trained_networks" folder.  
4. Visualized and analysed datasets, and analysed performance of trained networks.  
  
### Other things to note:  
- Jupyter notebooks are used for visualization (Step 4). Most of the technical stuff (neural network training, sorting datasets) are in python scripts.  
- I used HPC to complete independent jobs in job arrays where possible, most notably in the training of the many network configurations in Step 3.
- This was done by calling `python train_neural_network.py $args` on bash/sbatch scripts, where `$args` define the many different network configurations.  
- Steps 1 and 2 were also run in batches on HPC for the training and validation samples for quickly generating many input-output pairs.  
- Results should still be reproducible without HPC, especially if training only a few, similarly small neural networks.
- As a complete neural network beginner, I may or may not be using the technical terminology correctly nor consistently.  
