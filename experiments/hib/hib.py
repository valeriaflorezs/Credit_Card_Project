import numpy as np
from scipy.optimize import minimize
from combos import logit, thr_fast, f1_at
def hib_scores(O, y, T, lam=0.05, tau=0.3):
    """HIB: pesos softmax sobre logits que maximizan un F1 suave (regla propia, protocolo congelado)."""
    Z, Zt = logit(O), logit(T); m = Z.shape[1]; t0 = thr_fast(y, Z.mean(1))
    def f(p):
        w = np.exp(p[:m]-p[:m].max()); w /= w.sum(); q = 1/(1+np.exp(-(Z @ w-p[m])/tau))
        return -(2*(q*y).sum()/(q.sum()+y.sum())) + lam*m*((w-1/m)**2).sum()
    r = minimize(f, np.r_[np.zeros(m), t0], method="L-BFGS-B", options=dict(maxiter=60)); w = np.exp(r.x[:m]-r.x[:m].max()); w /= w.sum()
    return Z @ w, Zt @ w, w
