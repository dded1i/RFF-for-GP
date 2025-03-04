# RFF-for-GP  

## About The Project  
A Python implementation of Gaussian Processes (GP) with Random Fourier Features (RFF) for scalable approximation.  

## Features  
- Implements Gaussian Process (GP) regression.  
- Uses Random Fourier Features (RFF) for computational efficiency.  
- Supports different kernel functions.  
## Expected Results
## Usage
The functions to test and plot results are in GP_test (2).py.
Variable selection implementation is in GP_no_scheme.py.
No modifications are needed to run the simulations.


For questions or suggestions, contact me at dd68@rice.edu 
## Installation  
Ensure **Python 3.7+** is installed with the required packages:  
```bash
pip install numpy scipy scikit-learn matplotlib
```

## Example  
```python
def func(x: np.ndarray, eps=0):
    return x[:, 1] * x[:, 3] * np.sqrt(x[:, 5]) + x[:, 10] + 0.5 * np.exp(x[:, 11]) + eps
```
This is the surface response we're testing on. I'm using the ones from the papers, but you can tweak it.

n = 100
p = 20
n_f = 100

This section is input data, which also aligns with the simulation. You can modify n (covariance matrix dimension), but I was more interested in the number of iterations.


```python
model = gp.GPM_rand_features(x, y, 0.25, 2, 0.1, 500, 100)
```

This also contains parameters according to the paper.

```python
models = model.mcmc_iterate_verbose(5_000, 100)
```

Finally, here in the first parameter, you can decide on the number of iterations (1 coordinate) and the step in which the iterations are explained. This is where you play around!
## Performance Metrics
