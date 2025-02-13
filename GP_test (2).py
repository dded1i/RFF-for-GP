import numpy as np
import GP_no_scheme as gp
import matplotlib.pyplot as plt


#surface response we test on
# Let's test (6.2 in the paper):
# y = x1 + x2 + sin(3 * x3) + sin(5 * x4) + eps

# def func(x: np.ndarray, eps=0):
    # return x[:, 0] + x[:, 1] + np.sin(3 * x[:, 2]) + np.sin(5 * x[:, 3]) + eps

def func(x: np.ndarray, eps=0):
    return x[:, 1] * x[:, 3] * np.sqrt(x[:, 5]) + x[:, 10] + 0.5 * np.exp(x[:, 11]) + eps

#parametrization, same as in the simulation studies we have replicated
n = 100
p = 20
n_f = 100

x = np.random.uniform(size=(n, p))
eps = np.random.normal(scale=0.05, size=n)

x_f = np.random.uniform(size=(n_f, p))

y = func(x, eps)
y_f = func(x_f)

#commented out GPM3 refers to adaptive MCMC scheme without the random features
#model = gp.GPM3(x, y, 0.25, 2, 0.1, 500)
model = gp.GPM_rand_features(x, y, 0.25, 2, 0.1, 500, 100)
# model.m = 20
models = model.mcmc_iterate_verbose(5_000, 100)
print(models[-1].alpha)

all_ros = np.row_stack([m.ro[:20] for m in models])
likelihoods = [m.saved_log_likelihood for m in models]

best = models[np.argmax(likelihoods)]
y_predict = best.predict(x_f)

print(f"Mean error = {np.sqrt(np.mean(np.square(y_predict - y_f)))}")

# Plot the results
plt.figure(dpi=100)
plt.figure(figsize=(9, 6))  # Set figure size for the box plot
plt.boxplot(all_ros, flierprops={'marker': 'o', 'markersize': 1})
plt.show()
