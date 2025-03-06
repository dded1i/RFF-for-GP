# RFF-for-GP  

## About The Project  
A Python implementation of Gaussian Processes (GP) with Random Fourier Features (RFF) for scalable approximation.  

## Features  
- Implements Gaussian Process (GP) regression.  
- Uses Random Fourier Features (RFF) for computational efficiency.  
- Supports different kernel functions.

## Installation  
Ensure **Python 3.7+** is installed with the required packages:  
```bash
pip install numpy scipy scikit-learn matplotlib
```

## Usage
To test the variable selection method with the RFF kernel, run `GP_test(2).py`.  
Ensure that this file is in the same folder as `GP_no_scheme.py`.  
No modifications to the files are needed to run the simulations. 
This will generate:
- a printed Mean Error value
- runtime in seconds
- boxplots illustration of the association of individual covariates (x’s) with the response

## Testing Walkthrough 
The `GP_test(2).py` script follows these steps:
- import the required libraries
  ```
  import numpy as np
  import GP_no_scheme as gp
  import matplotlib.pyplot as plt
  ```
  
- set random seed `np.random.seed(42)` to make the results reproducible
- define the response function:
  ```
  def func(x: np.ndarray, eps=0):
  return x[:, 0] + x[:, 1] + np.sin(3 * x[:, 2]) + np.sin(5 * x[:, 3]) + eps
  ```
  - `eps` is normally distributed noise term.
  - The function corresponds to equation (6.2) from the referenced paper for results comparison, but you can modify it as needed.

## Expected Results
To test the variable selection method with the RFF kernel, run GP_test(2).py. Make sure the file is in the same folder as GP_no_scheme.py.
The expected results include:
- **Mean Error:**
    -  Mean error = 0.055775034318735026




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
