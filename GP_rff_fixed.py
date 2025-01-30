import numpy as np
import copy
import scipy.stats as stat
import scipy.special as spec
import time


class GPM1:
    """
    The univariate Gausian Process Model (page 37 of the thesis):

    y | Theth_gamma, r ~ N_n(0, 1/r * I_n + C(Theth_gamma))

    ro_k | gamma_k ~ gamma_k * U(0, 1) + (1 - gamma_k) * delt_1(*)

    gamma_k ~ Bern(alph_k)

    lambda_a ~ G(1, 1)

    lambda_z ~ G(1, 1)

    r ~ G(a_r, b_r)

    C = 1 / lambda_a * J_n + 1 / lambda_z * exp(-G)
    """
    # Parameters

    n: np.int32
    p: np.int32
    a_r: np.float64
    b_r: np.float64

    gamma: np.ndarray  # p-dimensional binary vector
    ro: np.ndarray  # p-dimensional vector
    alpha: np.ndarray  # p-dimensional vector

    x: np.ndarray  # n x p - dimensional matrix
    y: np.ndarray  # p-dimensional vector

    lambda_a: np.float64
    lambda_z: np.float64
    r: np.float64

    saved_log_likelihood: np.float64 | None = None  # a little optimization

    a = 3  # a from L^a (definetely helps)

    def count_chosen(self) -> np.int32:
        return np.sum(self.gamma)

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alpha_scalar: np.float64,
            a_r: np.float64,
            b_r: np.float64,
    ):
        super().__init__()
        self.n, self.p = x.shape
        self.alpha = np.full(self.p, alpha_scalar)
        self.gamma = (np.random.uniform(size=self.p) < self.alpha).astype(np.int8)
        self.ro = np.ones(self.p)
        self.ro[self.gamma == 1] = np.random.uniform(size=np.sum(self.gamma))

        self.a_r = a_r
        self.b_r = b_r
        self.x = x
        self.y = y

        self.lambda_a = np.random.gamma(1, 1)
        self.lambda_z = np.random.gamma(1, 1)
        self.r = np.random.gamma(a_r, b_r)

    def update_ro(self, ro_idx: np.int32, ro: np.float64) -> None:
        self.ro[ro_idx] = ro

    # Computations of the covariance matrix

    def calc_C(self) -> np.ndarray:
        """Calculates matrix C which is a part of the covariate matrix"""
        x_expanded = np.repeat(np.expand_dims(self.x, 1), self.n, 1)
        x_expanded_T = np.transpose(x_expanded, (1, 0, 2))
        ro_expanded = np.expand_dims(self.ro, (0, 1))
        result = np.prod(ro_expanded ** np.square(x_expanded - x_expanded_T), 2)
        return 1 / self.lambda_a + result / self.lambda_z

    def calc_Cov(self) -> np.ndarray:
        """Calculates covariation matrix for the multivariate normal distribution"""
        return np.eye(self.n) / self.r + self.calc_C()

    def calc_log_likelihood(self) -> np.float64:
        """Calculates log_likelihood for the chosen parameters
        (multivariate normal distribution)"""
        Cov = self.calc_Cov()
        const_term = self.n * np.log(2 * np.pi)
        det_term = np.linalg.slogdet(Cov)[1]
        inv_term = np.dot(self.y, np.linalg.solve(Cov, self.y))
        return -(const_term + det_term + inv_term) / 2

    # Markov Chain Monte-Carlo (Scheme 1)

    # Moves

    def mcmc_move_add(self):
        """Tries 'Add' move and returns the new Markov Chain (or self)."""
        try:
            pos = np.random.choice(np.where(self.gamma == 0)[0])
        except:
            return self  # nothing to add - skip the move

        next = copy.deepcopy(self)
        next.gamma[pos] = 1
        # next.ro[pos] = np.random.uniform()
        next.update_ro(pos, np.random.uniform())

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = (1 - self.alpha[pos]) / self.alpha[pos]
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    def mcmc_move_delete(self):
        """Tries 'Delete' move and returns the new Markov Chain (or self)."""
        try:
            pos = np.random.choice(np.where(self.gamma == 1)[0])
        except:
            return self  # nothing to delete - skip the move

        next = copy.deepcopy(self)
        next.gamma[pos] = 0
        next.update_ro(pos, 1)

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = self.alpha[pos] / (1 - self.alpha[pos])
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    def mcmc_move_swap(self):
        """Tries 'Swap' move and returns the new Markov Chain (or self)."""
        try:
            pos_add = np.random.choice(np.where(self.gamma == 0)[0])
            pos_delete = np.random.choice(np.where(self.gamma == 1)[0])
        except:
            return self  # nothing to add or to delete - skip the move

        next = copy.deepcopy(self)
        next.gamma[pos_add] = 1
        next.update_ro(pos_add, np.random.uniform())
        next.gamma[pos_delete] = 0
        next.update_ro(pos_delete, 1)

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = (1 - self.alpha[pos_add]) / self.alpha[pos_add] * self.alpha[pos_delete] / (
                    1 - self.alpha[pos_delete])
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    def mcmc_move_keep(self):
        """Tries 'Keep' move and returns the new Markov Chain (or self)."""
        next = copy.deepcopy(self)
        for pos in np.where(self.gamma == 1)[0]:
            next.update_ro(pos, np.random.uniform())

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = 1
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    # Updates

    def mcmc1_update_gamma_and_ro(self):
        move = np.random.randint(3)
        if move == 1:  # Add
            next = self.mcmc_move_add()
        elif move == 2:  # Delete
            next = self.mcmc_move_delete()
        else:  # Swap
            next = self.mcmc_move_swap()
        return next.mcmc_move_keep()

    def mcmc_update_lambda_a(self):
        next = copy.deepcopy(self)
        next.lambda_a = np.random.gamma(1, self.lambda_a)

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = stat.gamma.pdf(self.lambda_a, next.lambda_a) / stat.gamma.pdf(next.lambda_a, self.lambda_a)
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    def mcmc_update_lambda_z(self):
        next = copy.deepcopy(self)
        next.lambda_z = np.random.gamma(1, self.lambda_z)

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = stat.gamma.pdf(self.lambda_z, next.lambda_z) / stat.gamma.pdf(next.lambda_z, self.lambda_z)
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    def mcmc_update_r(self):
        next = copy.deepcopy(self)
        next.r = np.random.gamma(1, self.r)

        next.saved_log_likelihood = next.calc_log_likelihood()
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        q_div_q1 = stat.gamma.pdf(self.r, next.r) / stat.gamma.pdf(next.r, self.r)
        accept_prob = np.exp(self.a * (next.saved_log_likelihood - self.saved_log_likelihood)) * q_div_q1

        return next if np.random.uniform() < accept_prob else self

    # MCMC Scheme 2

    def mcmc2_update_gamma_and_ro(self):
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        model = self
        for pos in range(self.p):
            # Between Models
            next = copy.deepcopy(model)
            if next.gamma[pos] == 0:
                next.gamma[pos] = 1
                next.update_ro(pos, np.random.uniform())
            else:
                next.gamma[pos] = 0
                next.update_ro(pos, 1)
            next.saved_log_likelihood = next.calc_log_likelihood()

            accept_prob = np.exp(self.a * (next.saved_log_likelihood - model.saved_log_likelihood))
            if np.random.uniform() < accept_prob:
                model = next

            # Within Model
            if next.gamma[pos] == 1:
                next = copy.deepcopy(model)
                next.update_ro(pos, np.random.uniform())
                next.saved_log_likelihood = next.calc_log_likelihood()

                accept_prob = np.exp(self.a * (next.saved_log_likelihood - model.saved_log_likelihood))
                if np.random.uniform() < accept_prob:
                    model = next

        return model

    # Iteration

    def mcmc_iteration(self, scheme=1):
        return (
            self.mcmc1_update_gamma_and_ro() if scheme == 1 else
            self.mcmc2_update_gamma_and_ro() if scheme == 2 else
            None
        ).mcmc_update_lambda_a().mcmc_update_lambda_z().mcmc_update_r()

    def mcmc_iterate_verbose(self, iterations: int, verbose_step: int = 1, scheme=1) -> list:
        all_chains = []
        model = self
        start = time.time()
        for i in range(iterations):
            next = model.mcmc_iteration(scheme)
            all_chains.append(model)
            if i % verbose_step == 0:
                print(
                    f"\nITER {i}: \t{model.saved_log_likelihood}\n{model.lambda_a} {model.lambda_z} {model.r}\n\n{model.ro}\n")
            model = next
        all_chains.append(model)
        print(
            f"\nITER {iterations}: \t{model.saved_log_likelihood}\n{model.lambda_a} {model.lambda_z} {model.r}\n\n{model.ro}\nEND!")
        print(f"Time: {time.time() - start} s")
        return all_chains

    def mcmc_iterate_optimized(self, iterations: int, verbose_step: int = 1, scheme=1) -> tuple:
        log_likelihoods = []
        best_model = None
        ros = []

        model = self
        start = time.time()
        for i in range(iterations):
            next = model.mcmc_iteration(scheme)
            log_likelihoods.append(model.saved_log_likelihood)
            ros.append(model.ro)
            if best_model is None or best_model.saved_log_likelihood < model.saved_log_likelihood:
                best_model = model
            if i % verbose_step == 0:
                print(
                    f"\nITER {i}: \t{model.saved_log_likelihood}\n{model.lambda_a} {model.lambda_z} {model.r}\n\n{model.ro}\n")
            model = next

        log_likelihoods.append(model.saved_log_likelihood)
        ros.append(model.ro)
        if best_model is None or best_model.saved_log_likelihood < model.saved_log_likelihood:
            best_model = model
        print(
            f"\nITER {iterations}: \t{model.saved_log_likelihood}\n{model.lambda_a} {model.lambda_z} {model.r}\n\n{model.ro}\nEND!")
        print(f"Time: {time.time() - start} s")

        return best_model, log_likelihoods, ros

    # Prediction

    def predict(self, x_f: np.ndarray) -> np.ndarray:
        n1 = x_f.shape[0]

        model = copy.copy(self)
        model.n = self.n + n1
        model.x = np.row_stack([model.x, x_f])

        Cov = model.calc_Cov()
        return Cov[self.n:, :self.n] @ np.linalg.solve(Cov[:self.n, :self.n], self.y)


class GPM2(GPM1):
    """
    The univariate Gausian Process Model (optimized)
    """
    # Parameters

    # m: np.int32 | None = None
    saved_dist_squares: np.ndarray | None = None
    saved_G: np.ndarray | None = None

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alpha_scalar: np.float64,
            a_r: np.float64,
            b_r: np.float64,
    ):
        super().__init__(x, y, alpha_scalar, a_r, b_r)

    # Computations

    def calc_G(self) -> np.ndarray:
        """Calculates matrix G which is a part of the covariate matrix"""
        if self.saved_dist_squares is None:
            x_expanded = np.repeat(np.expand_dims(self.x, 1), self.n, 1)
            x_expanded_T = np.transpose(x_expanded, (1, 0, 2))
            self.saved_dist_squares = np.square(x_expanded - x_expanded_T)
            self.saved_G = None

        if self.saved_G is None:
            ro_expanded = np.expand_dims(np.log(self.ro), (0, 1))
            self.saved_G = np.sum(self.saved_dist_squares * ro_expanded, 2)

        return self.saved_G

    def calc_C(self) -> np.ndarray:
        """Calculates matrix C which is a part of the covariate matrix"""
        return 1 / self.lambda_a + np.exp(self.calc_G()) / self.lambda_z

    def update_ro(self, ro_idx: np.int32, ro: np.float64) -> None:
        if self.saved_G is None or self.saved_dist_squares is None:
            self.ro[ro_idx] = ro
        else:
            old_ro = self.ro[ro_idx]
            self.saved_G += self.saved_dist_squares[..., ro_idx] * np.log(ro / old_ro)
            self.ro[ro_idx] = ro

    # Prediction

    def predict(self, x_f: np.ndarray) -> np.ndarray:  # TO DO
        n1 = x_f.shape[0]

        model = copy.copy(self)
        model.n = self.n + n1
        model.x = np.row_stack([model.x, x_f])
        model.saved_dist_squares = None

        Cov = model.calc_Cov()
        return Cov[self.n:, :self.n] @ np.linalg.solve(Cov[:self.n, :self.n], self.y)


class GPM3(GPM2):
    """
    The univariate Gausian Process Model (optimized)
    Supports the Adaptive form of the Scheme 2
    """

    # Parameters

    t0: np.int32
    t: np.int32 = 0

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alpha_scalar: np.float64,
            a_r: np.float64,
            b_r: np.float64,

            t0: np.int32,
            eps: np.float64,
    ):
        super().__init__(x, y, alpha_scalar, a_r, b_r)
        self.t0 = t0

    def mcmc3_adaptive_update_gamma_and_ro(self):
        if self.saved_log_likelihood is None:
            self.saved_log_likelihood = self.calc_log_likelihood()

        model = self
        for pos in range(self.p):
            # Between Models
            next = copy.deepcopy(model)
            if next.gamma[pos] == 0 and np.random.uniform() < self.alpha[pos]:
                next.gamma[pos] = 1
                next.update_ro(pos, np.random.uniform())

                next.saved_log_likelihood = next.calc_log_likelihood()
                accept_prob = np.exp(self.a * (next.saved_log_likelihood - model.saved_log_likelihood))
                if np.random.uniform() < accept_prob:
                    model = next

            elif next.gamma[pos] == 1 and np.random.uniform() >= self.alpha[pos]:
                next.gamma[pos] = 0
                next.update_ro(pos, 1)

                next.saved_log_likelihood = next.calc_log_likelihood()
                accept_prob = np.exp(self.a * (next.saved_log_likelihood - model.saved_log_likelihood))
                if np.random.uniform() < accept_prob:
                    model = next

            # Within Model
            if next.gamma[pos] == 1:
                next = copy.deepcopy(model)
                next.update_ro(pos, np.random.uniform())
                next.saved_log_likelihood = next.calc_log_likelihood()

                accept_prob = np.exp(self.a * (next.saved_log_likelihood - model.saved_log_likelihood))
                if np.random.uniform() < accept_prob:
                    model = next

        return model

    def mcmc3_update_gamma_and_ro(self):
        if self.t <= self.t0:
            model = self.mcmc2_update_gamma_and_ro()
        else:
            model = self.mcmc3_adaptive_update_gamma_and_ro()
            model.alpha = (model.alpha * model.t + model.gamma) / (model.t + 1)
        model.t += 1
        return model

    def mcmc_iteration(self, scheme=1):
        if scheme != 3:
            return super().mcmc_iteration(scheme)
        else:
            return self.mcmc3_update_gamma_and_ro().mcmc_update_lambda_a().mcmc_update_lambda_z().mcmc_update_r()


# Introduce random features

class GPM_rand_features(GPM3):
    """
    The univariate Gausian Process Model with random features:

    z(x) = sqrt(2/D) * [ cos(omega_i.T @ x + b_i) ].T
    (z = sqrt(2 / D) * cos(x @ omega + b))

    omega_i ~ N(0, -2log(ro)) [shape: (p, D)]

    b_i ~ U[0, 2 * pi] [shape: (1, D)]

    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alpha_scalar: np.float64,
            a_r: np.float64,
            b_r: np.float64,

            t0: np.int32,
            D: np.int32,
    ):
        super().__init__(x, y, alpha_scalar, a_r, b_r, t0, None)
        self.D = D
        self.saved_dist_squares = None
        self.saved_G = None
        self.lambda_a = None
        self.lambda_z = None

    def calc_C(self) -> np.ndarray:
        """Calculates matrix C which is a part of the covariate matrix"""

        omega = np.row_stack([
            np.random.normal(
                scale=np.sqrt(-2 * np.log(ro)), size=self.D
            ) if ro != 1 else np.zeros(self.D)
            for ro in self.ro])
        b = np.random.uniform(0, 2 * np.pi, (1, self.D))
        z = np.cos(self.x @ omega + b) * np.sqrt(2 / self.D)

        expG = z @ z.T
        return expG

    def mcmc_update_lambda_a(self):
        return self

    def mcmc_update_lambda_z(self):
        return self

    def update_ro(self, ro_idx: np.int32, ro: np.float64) -> None:
        self.ro[ro_idx] = ro

    def mcmc_iteration(self, scheme=1):
        self.saved_log_likelihood = self.calc_log_likelihood()
        return super().mcmc_iteration(scheme)