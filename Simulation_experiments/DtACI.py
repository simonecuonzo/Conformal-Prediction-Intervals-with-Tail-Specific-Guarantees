# funzioni

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gamma
from scipy.stats import t, uniform

import pandas as pd
from scipy.stats import t as student_t

from math import sqrt, log

from types import SimpleNamespace
from scipy.optimize import minimize


import statsmodels.api as sm

from statsmodels.tsa.ar_model import AutoReg
import os

from scipy.special import logsumexp

class SkewStudent(object):

    """Skewed Student distribution class.

    Attributes
    ----------
    eta : float
        Degrees of freedom. :math:`2 < \eta < \infty`
    lam : float
        Skewness. :math:`-1 < \lambda < 1`

    Methods
    -------
    pdf
        Probability density function (PDF)
    cdf
        Cumulative density function (CDF)
    ppf
        Inverse cumulative density function (ICDF)
    rvs
        Random variates with mean zero and unit variance

    """

    def __init__(self, eta=10., lam=-.1):
        """Initialize the class.

        Parameters
        ----------
        eta : float
            Degrees of freedom. :math:`2 < \eta < \infty`
        lam : float
            Skewness. :math:`-1 < \lambda < 1`

        """
        self.eta = eta
        self.lam = lam

    def __const_a(self):
        """Compute a constant.

        Returns
        -------
        a : float

        """
        return 4*self.lam*self.__const_c()*(self.eta-2)/(self.eta-1)

    def __const_b(self):
        """Compute b constant.

        Returns
        -------
        b : float

        """
        return (1 + 3*self.lam**2 - self.__const_a()**2)**.5

    def __const_c(self):
        """Compute c constant.

        Returns
        -------
        c : float

        """
        return gamma((self.eta+1)/2) \
            / ((np.pi*(self.eta-2))**.5*gamma(self.eta/2))

    def pdf(self, arg):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        array
            PDF values. Same shape as the input.

        """
        c = self.__const_c()
        a = self.__const_a()
        b = self.__const_b()

        return b*c*(1 + 1/(self.eta-2) \
            *((b*arg+a)/(1+np.sign(arg+a/b)*self.lam))**2)**(-(self.eta+1)/2)

    def loglikelihood(self, param, arg):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        array
            PDF values. Same shape as the input.

        """
        self.eta, self.lam = param

        return -np.log(self.pdf(arg)).sum()

    def cdf(self, arg):
        """Cumulative density function (CDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        Returns
        -------
        array
            CDF values. Same shape as the input.

        """
        a = self.__const_a()
        b = self.__const_b()

        y = (b*arg+a)/(1+np.sign(arg+a/b)*self.lam) * (1-2/self.eta)**(-.5)
        cond = arg < -a/b

        return cond * (1-self.lam) * t.cdf(y, self.eta) \
            + ~cond * (-self.lam + (1+self.lam) * t.cdf(y, self.eta))

    def ppf(self, arg):
        """Inverse cumulative density function (ICDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        Returns
        -------
        array
            ICDF values. Same shape as the input.

        """
        arg = np.atleast_1d(arg)

        a = self.__const_a()
        b = self.__const_b()

        cond = arg < (1-self.lam)/2

        ppf1 = t.ppf(arg / (1-self.lam), self.eta)
        ppf2 = t.ppf(.5 + (arg - (1-self.lam)/2) / (1+self.lam), self.eta)
        ppf = -999.99*np.ones_like(arg)
        ppf = np.nan_to_num(ppf1) * cond \
            + np.nan_to_num(ppf2) * np.logical_not(cond)
        ppf = (ppf * (1+np.sign(arg-(1-self.lam)/2)*self.lam) \
            * (1-2/self.eta)**.5 - a)/b

        if ppf.shape == (1, ):
            return float(ppf)
        else:
            return ppf

    def rvs(self, size=1):
        """Random variates with mean zero and unit variance.

        Parameters
        ----------
        size : int or tuple
            Size of output array

        Returns
        -------
        array
            Array of random variates

        """
        return self.ppf(uniform.rvs(size=size))

    def plot_pdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.pdf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.pdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_cdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.cdf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.cdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_ppf(self, arg=np.linspace(.01, .99, 100)):
        """Plot inverse cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.ppf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.ppf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_rvspdf(self, arg=np.linspace(-2, 2, 100), size=1000):
        """Plot kernel density estimate of a random sample.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        """
        rvs = self.rvs(size=size)
        xrange = [arg.min(), arg.max()]
        sns.kdeplot(rvs, clip=xrange, label='kernel')
        plt.plot(arg, self.pdf(arg), label='true pdf')
        plt.xlim(xrange)
        plt.legend()
        plt.show()




# ------------------------------
# 1) IID Normal
# ------------------------------
def sim_normal_iid(T=500, mu=0.0, sigma=0.02, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.normal(mu, sigma, T)

# ------------------------------
# 2) AR(1) Normal
# ------------------------------
def sim_normal_ar1(T=500, mu=0.0, sigma=0.02, phi=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    eps = np.random.normal(0.0, sigma, T)
    y = np.zeros(T)
    for i in range(1, T):
        y[i] = mu + phi*y[i-1] + eps[i]
    return y

# ------------------------------
# 3) IID Student-t
# ------------------------------
def sim_t_iid(T=500, df=3, scale=0.02, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return student_t.rvs(df=df, loc=0, scale=scale, size=T)


# ------------------------------
# 4) AR(1) Student-t
# ------------------------------
def sim_t_ar1(T=500, df=3, scale=0.02, phi=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    eps = student_t.rvs(df=df, loc=0, scale=scale, size=T)
    y = np.zeros(T)
    for i in range(1, T):
        y[i] = phi*y[i-1] + eps[i]
    return y



# ------------------------------
# 5) IID Skewed Student-t
# ------------------------------
def sim_skewt_iid(T, df=5, lam=0.6, scale=0.02, random_state=None):
    """
    Simula una serie i.i.d. dalla Skewed Student-t (Hansen 1994).

    Parameters
    ----------
    T : int
        Lunghezza della serie.
    df : float
        Gradi di libertà (eta).
    lam : float
        Parametro di asimmetria (lambda).
    scale : float
        Scala (deviazione standard target).
    random_state : int or None
        Seed per la riproducibilità.

    Returns
    -------
    np.ndarray
        Serie di lunghezza T (i.i.d. skew-t).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Istanzia la distribuzione skew-t
    sts = SkewStudent(eta=df, lam=lam)

    # Genera T campioni i.i.d. (media 0, varianza 1)
    data = sts.rvs(size=T)

    # Applica la scala desiderata
    return scale * np.array(data)


# ------------------------------
# 6) AR(1) Skewed Student-t
# ------------------------------

def sim_skewt_ar1(T, df=5, lam=0.6, phi=0.3, scale=0.02, random_state=None):
    """
    Simula una serie Skewed Student-t con struttura AR(1):
        x_t = phi * x_{t-1} + scale * epsilon_t,
        epsilon_t ~ Skew-t(eta=df, lam=lam, var=1)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Distribuzione skew-t
    sts = SkewStudent(eta=df, lam=lam)

    # Shock i.i.d. (varianza unitaria)
    eps = np.array(sts.rvs(size=T))
    eps =scale *eps
    # Serie AR(1)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t-1] +  eps[t]

    return x



# --------- quantile (pinball) loss ρ_τ(u) ----------
def _rho_tau(u, tau):
    return (tau - (u < 0).astype(float)) * u

'''
# --------- Huberized (Moreau-smoothed) pinball loss ρ_{τ,γ}(u) ----------
def _rho_tau(u, tau, gamma=1e-3):
    """
    Huberized pinball (Moreau envelope) per il residuale u = y - q:
        ρ_{τ,γ}(u) =
            τ u - 0.5 τ^2 γ,                 if u ≥ τγ
            0.5 u^2 / γ,                      if -(1-τ)γ < u < τγ
           -(1-τ) u - 0.5 (1-τ)^2 γ,          if u ≤ -(1-τ)γ
    dove γ > 0 controlla l'ampiezza della regione quadratica.
    Per γ → 0 si torna alla pinball loss standard.
    """
    import numpy as np

    u = np.asarray(u, dtype=float)
    loss = np.empty_like(u)

    a = tau * gamma               # soglia alta
    b = (1.0 - tau) * gamma       # soglia bassa (positiva)

    mask_high = (u >= a)
    mask_low  = (u <= -b)
    mask_mid  = (~mask_high) & (~mask_low)

    loss[mask_high] = tau * u[mask_high] - 0.5 * (tau ** 2) * gamma
    loss[mask_mid]  = 0.5 * (u[mask_mid] ** 2) / gamma
    loss[mask_low]  = -(1.0 - tau) * u[mask_low] - 0.5 * ((1.0 - tau) ** 2) * gamma

    return loss
'''

# --------- un passo di ricorrenza CAViaR ----------
def _caviar_next(beta, var_prev, r_prev, tau, model, G=10.0):
    model = model.upper()
    if model == "AD":          # Adaptive
        b1 = float(beta[0])
        return var_prev + b1 * (1.0 / (1.0 + np.exp(G * (r_prev - var_prev))) - tau)
    elif model == "SAV":       # Symmetric Absolute Value
        b0, b1, b2 = [float(x) for x in beta]
        return b0 + b1 * var_prev + b2 * abs(r_prev)
    elif model == "AS":        # Asymmetric Slope
        b0, b1, b2, b3 = [float(x) for x in beta]
        pos = 1.0 if r_prev > 0 else 0.0
        neg = 1.0 if r_prev < 0 else 0.0
        return b0 + b1 * var_prev + (b2*pos + b3*neg) * abs(r_prev)
    elif model == "IG":        # Indirect GARCH
        b0, b1, b2 = [float(x) for x in beta]
        inner = abs(b0 + b1*(var_prev**2) + b2*(r_prev**2))
        return -np.sqrt(inner)


# --------- genera il path VaR_t dato beta (stima o forecast) ----------
def caviar_path(beta, series, tau, model = "AS", G=10.0, var0=None, init_window=300):
    r = np.asarray(series, dtype=float)
    n = len(r)
    VaR = np.empty(n, dtype=float)

    if var0 is None:
        m0 = max(10, min(init_window, n))   # seed robusto
        var0 = np.quantile(r[:m0], tau)
    VaR[0] = var0

    for t in range(1, n):
        VaR[t] = _caviar_next(beta, VaR[t-1], r[t-1], tau, model, G=G)
    return VaR

# --------- loss totale per stima CAViaR su finestra ----------
def _caviar_loss(beta, series, tau, model, G=10.0):
    VaR = caviar_path(beta, series, tau, model=model, G=G)
    u = np.asarray(series, float) - VaR
    return np.sum(_rho_tau(u, tau))

# --- helper: loss per AD (scalare) ---
def _caviar_loss_AD(b1, series, tau, G):
    # b1 è scalare, ma _caviar_loss si aspetta un vettore di parametri
    return _caviar_loss(np.array([float(b1)]), series, tau, model="AD", G=G)
from scipy.optimize import minimize, minimize_scalar

# --------- stima parametri (derivative-free, robusta) ----------
def est_caviar(train_series, tau, model="AS", G=10.0):
    """
    Stima CAViaR con:
      - AD  -> minimize_scalar(method='Brent') in [-10, 10]
      - altri (SAV/AS/IG) -> minimize(method='Nelder-Mead') unconstrained
    """
    model = str(model).upper()
    r = np.asarray(train_series, dtype=float)
    if G is None:
        G = 10.0  # default come da richiesta

    if model == "AD":
        # Ottimizzazione univariata (Brent) con bounds [-10, 10]
        res = minimize_scalar(
            _caviar_loss_AD,
            bounds=(-10.0, 10.0),
            method="bounded",      # uso 'bounded' perché Brent puro non accetta bounds espliciti in SciPy
            args=(r, tau, G),
            options=dict(maxiter=5000, xatol=1e-6)
        )
        params = np.array([res.x])
        success = res.success

    else:
        # Inizializzazioni standard per gli altri modelli
        if model == "SAV":
            x0 = np.array([0.0, 0.9, 0.1])
        elif model == "AS":
            x0 = np.array([0.0, 0.9, 0.1, 0.1])
        elif model == "IG":
            x0 = np.array([0.1, 0.8, 0.1])


        # Nelder–Mead senza vincoli (low=-Inf, up=Inf)
        res = minimize(
            _caviar_loss,
            x0=x0,
            args=(r, tau, model, G),
            method="Nelder-Mead",
            options=dict(maxiter=5000, xatol=1e-6, fatol=1e-6)
        )
        params = res.x
        success = res.success

    return SimpleNamespace(params=params, tau=tau, model=model, G=G, success=success)

# --------- stima tripla (low/med/high) per un braccio ----------
def fit_caviar_triplet(series, alpha, model="AS", G=10.0):
    tau_low, tau_med, tau_high = alpha/2.0, 0.5, 1.0 - alpha/2.0
    fit_low  = est_caviar(series, tau_low,  model=model, G=G)
    #fit_med  = est_caviar(series, tau_med,  model=model, G=G)
    fit_high = est_caviar(series, tau_high, model=model, G=G)
    return dict(low=fit_low, high=fit_high)#dict(low=fit_low, med=fit_med, high=fit_high)





def sim_t_iid_locscale(T=500, mu=0.0, sd=1.0, df=3, random_state=None):
    """
    IID Student-t con media ≈ mu e deviazione standard ≈ sd.
    """
    # Genero una serie Student-t con scala 1 (loc=0, scale=1)
    z = sim_t_iid(T=T, df=df, scale=1.0, random_state=random_state)
    # Location-scale transform
    return mu + sd * z


def sim_t_ar1_locscale(T=500, mu=0.0, sd=1.0, df=3, phi=0.2, random_state=None):
    """
    AR(1) Student-t con media ≈ mu e deviazione standard ≈ sd.
    """
    # Genero AR(1) Student-t con scala 1
    z = sim_t_ar1(T=T, df=df, scale=1.0, phi=phi, random_state=random_state)
    # Location-scale transform
    return mu + sd * z



def sim_skewt_iid_locscale(T=500, mu=0.0, sd=1.0, df=5, lam=0.6, random_state=None):
    """
    IID Skewed Student-t con media ≈ mu e deviazione standard ≈ sd.
    """
    # Genero skew-t iid con scala 1
    z = sim_skewt_iid(T=T, df=df, lam=lam, scale=1.0, random_state=random_state)
    # Location-scale transform
    return mu + sd * z



def sim_skewt_ar1_locscale(T=500, mu=0.0, sd=1.0, df=5, lam=0.6, phi=0.3, random_state=None):
    """
    AR(1) Skewed Student-t con media ≈ mu e deviazione standard ≈ sd.
    """
    # Genero AR(1) skew-t con scala 1
    z = sim_skewt_ar1(T=T, df=df, lam=lam, phi=phi, scale=1.0, random_state=random_state)
    # Location-scale transform
    return mu + sd * z



# OPEN QUANTILE

def compute_cp_coverage(trading_df):

    df = trading_df.copy()

    # =========================================================
    # OUTER BAND: L, U
    # =========================================================
    valid_outer = df["L"].notna() & df["U"].notna() & df["spread"].notna()

    # ---------- FULL ----------
    mask_full_outer = valid_outer
    spread_full_o = df.loc[mask_full_outer, "spread"]
    L_full_o      = df.loc[mask_full_outer, "L"]
    U_full_o      = df.loc[mask_full_outer, "U"]

    n_outer_full = mask_full_outer.sum()

    # two-sided coverage: L <= spread <= U
    covered_full_o = ((spread_full_o >= L_full_o) & (spread_full_o <= U_full_o)).sum()
    cov_outer_full = covered_full_o / n_outer_full if n_outer_full > 0 else np.nan

    # one-sided coverage lower: spread >= L
    covered_full_o_lower = (spread_full_o >= L_full_o).sum()
    cov_outer_full_lower = covered_full_o_lower / n_outer_full if n_outer_full > 0 else np.nan

    # one-sided coverage upper: spread <= U
    covered_full_o_upper = (spread_full_o <= U_full_o).sum()
    cov_outer_full_upper = covered_full_o_upper / n_outer_full if n_outer_full > 0 else np.nan


    # interval width outer FULL
    if n_outer_full > 0:
        #width_full_o = (U_full_o - L_full_o)
        #outer_width_full_mean   = float(width_full_o.mean())
        #outer_width_full_median = float(width_full_o.median())

        #### TOLGO VALORI IMMENSI PER LA STABLITà DEL CALCOLO DELLA MEDIA

        width_full_o = (U_full_o - L_full_o)

        WIDTH_CAP = 1e2#1e3
        width_full_o_capped = width_full_o[width_full_o <= WIDTH_CAP]

        ################# PRINTA NUMERO DI QUANTE OSSERVAZIONI HAI SCARTATO 
        n_total   = len(width_full_o)
        n_kept    = len(width_full_o_capped)
        n_removed = n_total - n_kept

        #print(
        #    f"[width cap] removed {n_removed}/{n_total} "
        #    f"({100 * n_removed / n_total:.2f}%) intervals "
        #    f"with width > {WIDTH_CAP:.1e}"
        #)
        ######################
        
        outer_width_full_mean = (
            float(width_full_o_capped.mean())
            if len(width_full_o_capped) > 0
            else np.nan
        )

        outer_width_full_median = float(width_full_o.median())

    else:
        outer_width_full_mean   = np.nan
        outer_width_full_median = np.nan


    return {
        # ================= OUTER two-sided =================
        "cov_outer_full": cov_outer_full,
        "n_outer_full": n_outer_full,


        # OUTER one-sided coverage
        "cov_outer_full_lower": cov_outer_full_lower,
        "cov_outer_full_upper": cov_outer_full_upper,


        # OUTER interval width stats
        "outer_width_full_mean":   outer_width_full_mean,
        "outer_width_full_median": outer_width_full_median,
    }



##################################################################################################################################


def rolling_ar_forecast(
    y_window,
    p=1,
    alpha=0.10
):
    """
    Fit AR(p) on y_window and produce:
      - point forecast m_t
      - parametric prediction interval (L_hat, U_hat)
      - predictive std sigma_pred (implicit)

    Returns:
      m_t, L_hat, U_hat, sigma_pred
    """

    y = np.asarray(y_window, float)

    if len(y) <= p + 2:
        # fallback
        m_t = float(np.mean(y))
        sigma = float(np.std(y, ddof=1)) if len(y) > 1 else 1e-6
        z = 1.96  # approx
        return m_t, m_t - z * sigma, m_t + z * sigma, sigma

    model = AutoReg(y, lags=p, trend="c", old_names=False)
    res = model.fit()

    # one-step-ahead forecast
    pred = res.get_prediction(start=len(y), end=len(y))
    m_t = float(pred.predicted_mean[0])
    


    # QUESTI SONO I REALI PREDICTION INTERVALS CHE USO NEL NAIVE METHOD

    # prediction interval (parametric)
    ci = pred.conf_int(alpha=alpha)
    L_hat = float(ci[0, 0])
    U_hat = float(ci[0, 1])



    # QUESTI INTERVALLI LI CALCOLO SOLAMENTE PER CALCOLARE LA VARIANZA
    # prediction interval (parametric)
    ci_var = pred.conf_int(alpha=0.1)
    L_hat_var = float(ci_var[0, 0])
    U_hat_var = float(ci_var[0, 1])

    # predictive std (approx)
    sigma_pred = float((U_hat_var - L_hat_var) / (2 * 1.645))  # rough, ok for scaling



    return m_t, L_hat, U_hat, sigma_pred



##################################################################################################################################



    
    
def init_faci_state(gammas, alpha_init):
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)

    return {
        "gammas": gammas,
        "expert_alphas": np.full(k, alpha_init),
        "expert_ws": np.ones(k),   # <-- pesi positivi iniziali
        "k": k
    }
def sigma_faci(I):
    """
    Mixing parameter sigma used in FACI.

    sigma = 1 / (2I)
    """
    return 1.0 / (2.0 * I)


import numpy as np

def eta_faci(alpha_target, I, k):
    """
    Learning rate eta for FACI (alpha-based),
    derived from Gibbs & Candès (2022).

    Parameters
    ----------
    alpha_target : float
        Target miscoverage level (e.g. 0.1).
    I : int
        Horizon parameter (theoretical window).
    k : int
        Number of experts (len(gammas)).

    Returns
    -------
    eta : float
        FACI learning rate.
    """
    alpha = alpha_target               # miscoverage
    coverage = 1.0 - alpha

    #denom = (
    #    (coverage**2) * (alpha**3)
    #    + (coverage**3) * (alpha**2)
    #) / 3.0
    denom = (coverage**2) * (alpha**2) 

    eta = np.sqrt(3.0 / I) * np.sqrt((np.log(I * k) + 2.0) / denom)
    return eta


def pinball_loss(beta, theta, alpha_target):
    """
    Pinball loss used in FACI (alpha-based formulation).

    ℓ(beta, theta) = alpha*(beta - theta) - min(0, beta - theta)

    Parameters
    ----------
    beta : float
        Rank of current score.
    theta : array-like
        Expert alphas.
    alpha_target : float
        Target miscoverage level.

    Returns
    -------
    loss : ndarray
        Pinball loss per expert.
    """
    u = beta - theta
    return alpha_target * u - np.minimum(u, 0.0)



def rolling_qr_faci(
    y_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=1/1000,
    lookback=240,
    caviar_model="SAV",
    G=10.0,
):
    """
    Rolling CQR + FACI (alpha-based), DtACI-faithful version.
    Weight update follows exactly the DtACI GitHub implementation.
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Base CQR predictor (empirical quantiles)
    # ======================================================
    window_data = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])

    tau_low = alpha_target / 2.0
    tau_high = 1.0 - alpha_target / 2.0


    # q_low_t = np.quantile(window_data, tau_low)
    # q_high_t = np.quantile(window_data, tau_high)
    
    
    _, q_low_t, q_high_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha=alpha_target # QUI alpha=alpha_target LO USO...
    )
    # ======================================================
    # Expert probabilities + aggregated alpha
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = np.sum(expert_probs * expert_alphas)

    # ======================================================
    # Conformal calibration
    # ======================================================
    nc = len(scores_history)

    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(scores_history)[qn_idx])

    # ======================================================
    # Predictive interval
    # ======================================================
    L_t = float(q_low_t - qn)
    U_t = float(q_high_t + qn)

    # ======================================================
    # Current score
    # ======================================================
    score_t = max(q_low_t - y_t, y_t - q_high_t)
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update (skip at beginning)
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(scores_history[:-1])
        beta_t = np.mean(past_scores >= score_t)

        # ----- pinball losses (DtACI style)
        expert_losses = pinball_loss(
            beta_t,
            expert_alphas,
            alpha_target
        )

        # ----- update expert alphas (local ACI)
        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)

        # ----- weight update (EXACT DtACI)
        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)              # stabilization
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )



    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws

    # ======================================================
    # Extreme Score Index
    # ======================================================
    den = max(abs(L_t), 1e-9)
    ESI_t = float(U_t / den)

    return float(L_t), float(U_t), float(ESI_t), faci_state, scores_history





def cp_faci(
    spread,
    lookback_aci=240,
    alpha_target=0.1,
    caviar_model="SAV",
    G=10,
    I=100,
    gammas=None,
    warm_up_0=1,
    warm_up=20
):
    spread = np.asarray(spread, float)
    T = len(spread)


    # -----------------------------------------
    # 1) Gammas e alpha_init
    # -----------------------------------------
    if gammas is None:
        gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)

    alpha_init = alpha_target

    # -----------------------------------------
    # 2) FACI theoretical hyperparameters
    # -----------------------------------------
    k = len(gammas)
    sigma = sigma_faci(I)
    eta = eta_faci(alpha_target=alpha_target, I=I, k=k)

    # -----------------------------------------
    # 3) Init FACI state
    # -----------------------------------------
    faci_state_out = init_faci_state(
        gammas=gammas,
        alpha_init=alpha_init
    )
    scores_outer = []

    # -----------------------------------------
    # 4) Allocazioni
    # -----------------------------------------
    L_out_arr = np.full(T, np.nan)
    U_out_arr = np.full(T, np.nan)
    alpha_bar_arr = np.full(T, np.nan)

    # =====================================================
    # 5A) PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        L_t, U_t, ESI_t, faci_state_out, scores_outer = rolling_qr_faci(
            y_obs=y_obs,
            scores_history=scores_outer,
            faci_state=faci_state_out,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            caviar_model=caviar_model,
            G=G
        )


    # =====================================================
    # 5B) MAIN LOOP (STORE RESULTS)
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        L_t, U_t, ESI_t, faci_state_out, scores_outer = rolling_qr_faci(
            y_obs=y_obs,
            scores_history=scores_outer,
            faci_state=faci_state_out,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            caviar_model=caviar_model,
            G=G
        )

        L_out_arr[t] = L_t
        U_out_arr[t] = U_t

        # ---- alpha aggregato (diagnostica CORRETTA)
        ws = faci_state_out["expert_ws"]
        a  = faci_state_out["expert_alphas"]
        p  = ws / (np.sum(ws) + 1e-300)
        alpha_bar_arr[t] = float(np.sum(p * a))

    return pd.DataFrame({
        "spread": spread,
        "L": L_out_arr,
        "U": U_out_arr,
        "alpha_outer": alpha_bar_arr
    }).iloc[warm_up:]

def rolling_qr_faci_open(
    y_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=1/1000,
    lookback=240,
    side="lower",
    caviar_model="SAV",
    G=10.0,
):
    """
    One-sided (open) Conformal Prediction + FACI (alpha-based),
    DtACI-faithful version.

    Parameters
    ----------
    side : {"lower", "upper"}
        Which one-sided bound to compute.

    Returns
    -------
    B_t : float
        One-sided conformal bound.
    faci_state : dict
        Updated FACI state.
    scores_history : list
        Updated score history.
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Base quantile predictor
    # ======================================================
    window_data = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])


    if side == "lower":
        #q_t = float(np.quantile(window_data, alpha_target))
        
        _, q_t, _, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
    elif side == "upper":
        #q_t = float(np.quantile(window_data, 1.0 - alpha_target))

        
        _, _ , q_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    # ======================================================
    # Expert probabilities + aggregated alpha
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = np.sum(expert_probs * expert_alphas)

    # ======================================================
    # Conformal calibration (one-sided)
    # ======================================================
    nc = len(scores_history)

    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(scores_history)[qn_idx])

    # ======================================================
    # Bound and score
    # ======================================================
    if side == "lower":
        B_t = float(q_t - qn)
      
        #score_t = max(q_t - y_t,0)
        score_t = q_t - y_t
    else:  # upper
        B_t = float(q_t + qn)
   
        #score_t = max(y_t - q_t,0)
        score_t = y_t - q_t
        
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update (skip at beginning)
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(scores_history[:-1])
        beta_t = np.mean(past_scores >= score_t)

        # ----- pinball losses (DtACI style)
        expert_losses = pinball_loss(
            beta_t,
            expert_alphas,
            alpha_target
        )

        # ----- update expert alphas (local ACI)
        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)

        # ----- weight update (EXACT DtACI)
        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)              # stabilization
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws

    return float(B_t), faci_state, scores_history



def cp_faci_open(
    spread,
    lookback_aci=240,
    alpha_target=0.1,
    caviar_model="SAV",
    G=10,
    I=100,
    gammas=None,
    warm_up_0=1,
    warm_up=20
):


    spread = np.asarray(spread, float)
    T = len(spread)



    # -----------------------------------------
    # FACI hyperparameters
    # -----------------------------------------
    if gammas is None:
        gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)

    alpha_init = alpha_target
    k = len(gammas)

    sigma = sigma_faci(I)
    eta = eta_faci(alpha_target=alpha_target, I=I, k=k)

    # -----------------------------------------
    # Output allocations
    # -----------------------------------------
    L_out_arr = np.full(T, np.nan)
    U_out_arr = np.full(T, np.nan)

    alpha_out_low_arr = np.full(T, np.nan)
    alpha_out_up_arr  = np.full(T, np.nan)

    # -----------------------------------------
    # FACI states + scores (SEPARATI)
    # -----------------------------------------
    faci_state_out_low = init_faci_state(gammas, alpha_init)
    faci_state_out_up  = init_faci_state(gammas, alpha_init)

    scores_out_low = []
    scores_out_up  = []

    # =====================================================
    # PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        # ---------- LOWER ----------
        L_t, faci_state_out_low, scores_out_low = rolling_qr_faci_open(
            y_obs=y_obs,
            scores_history=scores_out_low,
            faci_state=faci_state_out_low,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # ---------- UPPER ----------
        U_t, faci_state_out_up, scores_out_up = rolling_qr_faci_open(
            y_obs=y_obs,
            scores_history=scores_out_up,
            faci_state=faci_state_out_up,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )


    # =====================================================
    # MAIN LOOP (STORE RESULTS)
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        # ---------- LOWER ----------
        L_t, faci_state_out_low, scores_out_low = rolling_qr_faci_open(
            y_obs=y_obs,
            scores_history=scores_out_low,
            faci_state=faci_state_out_low,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # ---------- UPPER ----------
        U_t, faci_state_out_up, scores_out_up = rolling_qr_faci_open(
            y_obs=y_obs,
            scores_history=scores_out_up,
            faci_state=faci_state_out_up,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )

        # ---- save bounds
        L_out_arr[t] = L_t
        U_out_arr[t] = U_t

        # ---- diagnostics: aggregated alphas
        ws_low = faci_state_out_low["expert_ws"]
        a_low  = faci_state_out_low["expert_alphas"]
        alpha_out_low_arr[t] = float(np.sum((ws_low / (np.sum(ws_low) + 1e-300)) * a_low))

        ws_up = faci_state_out_up["expert_ws"]
        a_up  = faci_state_out_up["expert_alphas"]
        alpha_out_up_arr[t] = float(np.sum((ws_up / (np.sum(ws_up) + 1e-300)) * a_up))

    return pd.DataFrame({
        "spread": spread,
        "L": L_out_arr,
        "U": U_out_arr,
        "alpha_out_low": alpha_out_low_arr,
        "alpha_out_up": alpha_out_up_arr,
    }).iloc[warm_up:]


def compute_pointwise_cp_coverage(mc_results):
    """
    Calcola il coverage pointwise:
    Cov_t = P( L_t <= u_t <= U_t )

    Parameters
    ----------
    mc_results : list of dict
        Ogni elemento contiene:
        - "spread": array (T,)
        - "L": array (T,)
        - "U": array (T,)

    Returns
    -------
    coverage_t : np.ndarray (T,)
        Coverage stimato per ogni tempo t.
    """

    n_mc = len(mc_results)
    if n_mc == 0:
        raise ValueError("mc_results è vuoto")

    T = len(mc_results[0]["spread"])
    coverage_counts = np.zeros(T)

    for res in mc_results:
        u = res["spread"]
        L = res["L"]
        U = res["U"]

        valid = (~np.isnan(u)) & (~np.isnan(L)) & (~np.isnan(U))
        covered = (u >= L) & (u <= U) & valid

        coverage_counts += covered.astype(float)

    coverage_t = coverage_counts / n_mc
    return coverage_t


def compute_pointwise_one_sided_coverage(mc_results):
    """
    Coverage pointwise one-sided:
      - lower: P(u_t >= L_t)
      - upper: P(u_t <= U_t)
    """

    n_mc = len(mc_results)
    T = len(mc_results[0]["spread"])

    cov_lower = np.zeros(T)
    cov_upper = np.zeros(T)

    for res in mc_results:
        u = res["spread"]
        L = res["L"]
        U = res["U"]

        valid = (~np.isnan(u)) & (~np.isnan(L)) & (~np.isnan(U))

        cov_lower += ((u >= L) & valid).astype(float)
        cov_upper += ((u <= U) & valid).astype(float)

    return cov_lower / n_mc, cov_upper / n_mc


def plot_pointwise_coverage(
    coverage_t,
    alpha_target,
    title=None
):
    """
    Plot del coverage pointwise con target teorico.
    """

    import matplotlib.pyplot as plt
    T = len(coverage_t)
    t_grid = np.arange(T)

    plt.figure(figsize=(10, 4))
    plt.plot(t_grid, coverage_t, label="Empirical pointwise coverage")
    plt.axhline(
        1 - alpha_target,
        color="red",
        linestyle="--",
        label=f"Target coverage = {1-alpha_target:.2f}"
    )

  

    plt.ylim(0, 1.05)
    plt.xlabel("Time")
    plt.ylabel("Coverage")
    plt.legend()
    plt.grid(alpha=0.3)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()




def run_mc_cp_coverage(
    generator_func,
    n_mc,
    T,
    lookback_aci,
    alpha_outer_target,
    caviar_model,
    G,
    I = 100,
    gammas = None,
    warm_up_0=1,
    warm_up=20,
    **generator_kwargs
):
    """
    Monte Carlo evaluation of CP–ACI bands.

    Computes:
    (i)  CLASSIC coverage metrics (time-averaged, per simulation)
    (ii) POINTWISE coverage over time (averaged across MC simulations)
    (iii) Optional automatic plotting of pointwise coverage

    Returns
    -------
    summary : dict
        Monte Carlo means/stds of classic coverage and width metrics.
    mc_results : list of dict
        Per-simulation results.
    pointwise_results : dict
        Pointwise coverage arrays.
    """

    mc_results = []

    # =========================================================
    # 1) Monte Carlo loop
    # =========================================================
    for mc in range(n_mc):
        np.random.seed(mc)

        gen_kwargs = dict(generator_kwargs)
        gen_kwargs["random_state"] = None

        # --- simulate spread
        spread = generator_func(T=T, **gen_kwargs)

        # --- run CP–ACI trading
        trading_df = cp_faci(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            caviar_model=caviar_model,
            G=G,
            I=I,
            gammas=gammas,
            warm_up_0=warm_up_0,
            warm_up=warm_up
        )

        # --- classic (time-averaged) coverage + widths
        cov_info = compute_cp_coverage(trading_df)

        # --- store minimal info for pointwise coverage
        mc_results.append({
            # classic metrics
            "cov_outer_full":          cov_info["cov_outer_full"],
            "cov_outer_full_lower":    cov_info["cov_outer_full_lower"],
            "cov_outer_full_upper":    cov_info["cov_outer_full_upper"],
            "outer_width_full_mean":   cov_info["outer_width_full_mean"],
            "outer_width_full_median": cov_info["outer_width_full_median"],

            # raw series for pointwise coverage
            "spread": trading_df["spread"].values,
            "L":      trading_df["L"].values,
            "U":      trading_df["U"].values,
        })

    # =========================================================
    # 2) CLASSIC Monte Carlo summaries
    # =========================================================
    cov_outer_full_arr          = np.array([r["cov_outer_full"]          for r in mc_results], float)
    cov_outer_full_lower_arr    = np.array([r["cov_outer_full_lower"]    for r in mc_results], float)
    cov_outer_full_upper_arr    = np.array([r["cov_outer_full_upper"]    for r in mc_results], float)
    outer_width_full_mean_arr   = np.array([r["outer_width_full_mean"]   for r in mc_results], float)
    outer_width_full_median_arr = np.array([r["outer_width_full_median"] for r in mc_results], float)

    summary = {
        "cov_outer_mean (CLASSIC)": np.nanmean(cov_outer_full_arr),
        "cov_outer_std (CLASSIC)":  np.nanstd(cov_outer_full_arr, ddof=1),

        "cov_outer_lower_mean (CLASSIC)": np.nanmean(cov_outer_full_lower_arr),
        "cov_outer_lower_std (CLASSIC)":  np.nanstd(cov_outer_full_lower_arr, ddof=1),

        "cov_outer_upper_mean (CLASSIC)": np.nanmean(cov_outer_full_upper_arr),
        "cov_outer_upper_std (CLASSIC)":  np.nanstd(cov_outer_full_upper_arr, ddof=1),

        "outer_width_mean_mean (CLASSIC)":   np.nanmean(outer_width_full_mean_arr),
        "outer_width_mean_std (CLASSIC)":    np.nanstd(outer_width_full_mean_arr, ddof=1),

        "outer_width_median_mean (CLASSIC)": np.nanmean(outer_width_full_median_arr),
        "outer_width_median_std (CLASSIC)":  np.nanstd(outer_width_full_median_arr, ddof=1),
    }

    # =========================================================
    # 3) POINTWISE coverage (NEW)
    # =========================================================
    cov_outer_t = compute_pointwise_cp_coverage(mc_results)

    cov_outer_lower_t, cov_outer_upper_t = (
        compute_pointwise_one_sided_coverage(mc_results)
    )

    pointwise_results = {
        "cov_outer_t":       cov_outer_t,
        "cov_outer_lower_t": cov_outer_lower_t,
        "cov_outer_upper_t": cov_outer_upper_t,
    }


    return summary, mc_results, pointwise_results







def run_mc_cp_coverage_open(
    generator_func,
    n_mc,
    T,
    lookback_aci,
    alpha_outer_target,
    caviar_model,
    G,
    I = 100,
    gammas = None,
    warm_up_0=1,
    warm_up=20,
    **generator_kwargs
):
    """
    Monte Carlo evaluation of OPEN one-sided CP–ACI bands.

    Computes:
    (i)  CLASSIC coverage metrics (time-averaged, per simulation)
    (ii) POINTWISE coverage over time (averaged across MC simulations)
    (iii) Optional automatic plotting of pointwise coverage

    Returns
    -------
    summary : dict
        Monte Carlo means/stds of classic coverage and width metrics.
    mc_results : list of dict
        Per-simulation results.
    pointwise_results : dict
        Pointwise coverage arrays.
    """

    mc_results = []

    # =========================================================
    # 1) Monte Carlo loop
    # =========================================================
    for mc in range(n_mc):
        np.random.seed(mc)

        gen_kwargs = dict(generator_kwargs)
        gen_kwargs["random_state"] = None

        # --- simulate spread
        spread = generator_func(T=T, **gen_kwargs)

        # --- run CP–ACI (OPEN)
        t_df = cp_faci_open(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            caviar_model=caviar_model,
            G=G,
            I=I,
            gammas=gammas,
            warm_up_0=warm_up_0,
            warm_up=warm_up
        )

        # --- classic (time-averaged) coverage + widths
        cov_info = compute_cp_coverage(t_df)

        mc_results.append({
            # classic metrics
            "cov_outer_full":          cov_info["cov_outer_full"],
            "cov_outer_full_lower":    cov_info["cov_outer_full_lower"],
            "cov_outer_full_upper":    cov_info["cov_outer_full_upper"],
            "outer_width_full_mean":   cov_info["outer_width_full_mean"],
            "outer_width_full_median": cov_info["outer_width_full_median"],

            # raw series for pointwise coverage
            "spread": t_df["spread"].values,
            "L":      t_df["L"].values,
            "U":      t_df["U"].values,
        })

    # =========================================================
    # 2) CLASSIC Monte Carlo summaries
    # =========================================================
    cov_outer_full_arr          = np.array([r["cov_outer_full"]          for r in mc_results], float)
    cov_outer_full_lower_arr    = np.array([r["cov_outer_full_lower"]    for r in mc_results], float)
    cov_outer_full_upper_arr    = np.array([r["cov_outer_full_upper"]    for r in mc_results], float)
    outer_width_full_mean_arr   = np.array([r["outer_width_full_mean"]   for r in mc_results], float)
    outer_width_full_median_arr = np.array([r["outer_width_full_median"] for r in mc_results], float)

    summary = {
        "cov_outer_mean_open": np.nanmean(cov_outer_full_arr),
        "cov_outer_std_open":  np.nanstd(cov_outer_full_arr, ddof=1),

        "cov_outer_lower_mean_open": np.nanmean(cov_outer_full_lower_arr),
        "cov_outer_lower_std_open":  np.nanstd(cov_outer_full_lower_arr, ddof=1),

        "cov_outer_upper_mean_open": np.nanmean(cov_outer_full_upper_arr),
        "cov_outer_upper_std_open":  np.nanstd(cov_outer_full_upper_arr, ddof=1),

        "outer_width_mean_mean_open":   np.nanmean(outer_width_full_mean_arr),
        "outer_width_mean_std_open":    np.nanstd(outer_width_full_mean_arr, ddof=1),

        "outer_width_median_mean_open": np.nanmean(outer_width_full_median_arr),
        "outer_width_median_std_open":  np.nanstd(outer_width_full_median_arr, ddof=1),
    }

    # =========================================================
    # 3) POINTWISE coverage (NEW)
    # =========================================================
    cov_outer_t = compute_pointwise_cp_coverage(mc_results)
    cov_outer_lower_t, cov_outer_upper_t = (
        compute_pointwise_one_sided_coverage(mc_results)
    )

    pointwise_results = {
        "cov_outer_t":       cov_outer_t,
        "cov_outer_lower_t": cov_outer_lower_t,
        "cov_outer_upper_t": cov_outer_upper_t,
    }


    return summary, mc_results, pointwise_results




################################################################################
################################################################################        




def rolling_qr_faci_open_max(
    y_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=1/1000,
    lookback=240,
    side="lower",
    caviar_model="SAV",
    G=10.0,
):
    """
    One-sided (open) Conformal Prediction + FACI (alpha-based),
    DtACI-faithful version.

    Parameters
    ----------
    side : {"lower", "upper"}
        Which one-sided bound to compute.

    Returns
    -------
    B_t : float
        One-sided conformal bound.
    faci_state : dict
        Updated FACI state.
    scores_history : list
        Updated score history.
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Base quantile predictor
    # ======================================================
    window_data = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])


    if side == "lower":
        #q_t = float(np.quantile(window_data, alpha_target))
        
        _, q_t, _, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
    elif side == "upper":
        #q_t = float(np.quantile(window_data, 1.0 - alpha_target))

        
        _, _ , q_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    # ======================================================
    # Expert probabilities + aggregated alpha
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = np.sum(expert_probs * expert_alphas)

    # ======================================================
    # Conformal calibration (one-sided)
    # ======================================================
    nc = len(scores_history)

    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(scores_history)[qn_idx])

    # ======================================================
    # Bound and score
    # ======================================================
    if side == "lower":
        B_t = float(q_t - qn)
      
        score_t = max(q_t - y_t,0)
     
    else:  # upper
        B_t = float(q_t + qn)
   
        score_t = max(y_t - q_t,0)
   
        
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update (skip at beginning)
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(scores_history[:-1])
        beta_t = np.mean(past_scores >= score_t)

        # ----- pinball losses (DtACI style)
        expert_losses = pinball_loss(
            beta_t,
            expert_alphas,
            alpha_target
        )

        # ----- update expert alphas (local ACI)
        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)

        # ----- weight update (EXACT DtACI)
        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)              # stabilization
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws

    return float(B_t), faci_state, scores_history



def cp_faci_open_max(
    spread,
    lookback_aci=240,
    alpha_target=0.1,
    caviar_model="SAV",
    G=10,
    I=100,
    gammas=None,
    warm_up_0=1,
    warm_up=20
):


    spread = np.asarray(spread, float)
    T = len(spread)



    # -----------------------------------------
    # FACI hyperparameters
    # -----------------------------------------
    if gammas is None:
        gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)

    alpha_init = alpha_target
    k = len(gammas)

    sigma = sigma_faci(I)
    eta = eta_faci(alpha_target=alpha_target, I=I, k=k)

    # -----------------------------------------
    # Output allocations
    # -----------------------------------------
    L_out_arr = np.full(T, np.nan)
    U_out_arr = np.full(T, np.nan)

    alpha_out_low_arr = np.full(T, np.nan)
    alpha_out_up_arr  = np.full(T, np.nan)

    # -----------------------------------------
    # FACI states + scores (SEPARATI)
    # -----------------------------------------
    faci_state_out_low = init_faci_state(gammas, alpha_init)
    faci_state_out_up  = init_faci_state(gammas, alpha_init)

    scores_out_low = []
    scores_out_up  = []

    # =====================================================
    # PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        # ---------- LOWER ----------
        L_t, faci_state_out_low, scores_out_low = rolling_qr_faci_open_max(
            y_obs=y_obs,
            scores_history=scores_out_low,
            faci_state=faci_state_out_low,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # ---------- UPPER ----------
        U_t, faci_state_out_up, scores_out_up = rolling_qr_faci_open_max(
            y_obs=y_obs,
            scores_history=scores_out_up,
            faci_state=faci_state_out_up,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )


    # =====================================================
    # MAIN LOOP (STORE RESULTS)
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        # ---------- LOWER ----------
        L_t, faci_state_out_low, scores_out_low = rolling_qr_faci_open_max(
            y_obs=y_obs,
            scores_history=scores_out_low,
            faci_state=faci_state_out_low,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # ---------- UPPER ----------
        U_t, faci_state_out_up, scores_out_up = rolling_qr_faci_open_max(
            y_obs=y_obs,
            scores_history=scores_out_up,
            faci_state=faci_state_out_up,
            alpha_target=alpha_target,
            eta=eta,
            sigma=sigma,
            lookback=lookback_aci,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )

        # ---- save bounds
        L_out_arr[t] = L_t
        U_out_arr[t] = U_t

        # ---- diagnostics: aggregated alphas
        ws_low = faci_state_out_low["expert_ws"]
        a_low  = faci_state_out_low["expert_alphas"]
        alpha_out_low_arr[t] = float(np.sum((ws_low / (np.sum(ws_low) + 1e-300)) * a_low))

        ws_up = faci_state_out_up["expert_ws"]
        a_up  = faci_state_out_up["expert_alphas"]
        alpha_out_up_arr[t] = float(np.sum((ws_up / (np.sum(ws_up) + 1e-300)) * a_up))

    return pd.DataFrame({
        "spread": spread,
        "L": L_out_arr,
        "U": U_out_arr,
        "alpha_out_low": alpha_out_low_arr,
        "alpha_out_up": alpha_out_up_arr,
    }).iloc[warm_up:]









def run_mc_cp_coverage_open_max(
    generator_func,
    n_mc,
    T,
    lookback_aci,
    alpha_outer_target,
    caviar_model,
    G,
    I = 100,
    gammas = None,
    warm_up_0=1,
    warm_up=20,
    **generator_kwargs
):
    """
    Monte Carlo evaluation of OPEN one-sided CP–ACI bands.

    Computes:
    (i)  CLASSIC coverage metrics (time-averaged, per simulation)
    (ii) POINTWISE coverage over time (averaged across MC simulations)
    (iii) Optional automatic plotting of pointwise coverage

    Returns
    -------
    summary : dict
        Monte Carlo means/stds of classic coverage and width metrics.
    mc_results : list of dict
        Per-simulation results.
    pointwise_results : dict
        Pointwise coverage arrays.
    """

    mc_results = []

    # =========================================================
    # 1) Monte Carlo loop
    # =========================================================
    for mc in range(n_mc):
        np.random.seed(mc)

        gen_kwargs = dict(generator_kwargs)
        gen_kwargs["random_state"] = None

        # --- simulate spread
        spread = generator_func(T=T, **gen_kwargs)

        # --- run CP–ACI (OPEN)
        t_df = cp_faci_open_max(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            caviar_model=caviar_model,
            G=G,
            I=I,
            gammas=gammas,
            warm_up_0=warm_up_0,
            warm_up=warm_up
        )

        # --- classic (time-averaged) coverage + widths
        cov_info = compute_cp_coverage(t_df)

        mc_results.append({
            # classic metrics
            "cov_outer_full":          cov_info["cov_outer_full"],
            "cov_outer_full_lower":    cov_info["cov_outer_full_lower"],
            "cov_outer_full_upper":    cov_info["cov_outer_full_upper"],
            "outer_width_full_mean":   cov_info["outer_width_full_mean"],
            "outer_width_full_median": cov_info["outer_width_full_median"],

            # raw series for pointwise coverage
            "spread": t_df["spread"].values,
            "L":      t_df["L"].values,
            "U":      t_df["U"].values,
        })

    # =========================================================
    # 2) CLASSIC Monte Carlo summaries
    # =========================================================
    cov_outer_full_arr          = np.array([r["cov_outer_full"]          for r in mc_results], float)
    cov_outer_full_lower_arr    = np.array([r["cov_outer_full_lower"]    for r in mc_results], float)
    cov_outer_full_upper_arr    = np.array([r["cov_outer_full_upper"]    for r in mc_results], float)
    outer_width_full_mean_arr   = np.array([r["outer_width_full_mean"]   for r in mc_results], float)
    outer_width_full_median_arr = np.array([r["outer_width_full_median"] for r in mc_results], float)

    summary = {
        "cov_outer_mean_open": np.nanmean(cov_outer_full_arr),
        "cov_outer_std_open":  np.nanstd(cov_outer_full_arr, ddof=1),

        "cov_outer_lower_mean_open": np.nanmean(cov_outer_full_lower_arr),
        "cov_outer_lower_std_open":  np.nanstd(cov_outer_full_lower_arr, ddof=1),

        "cov_outer_upper_mean_open": np.nanmean(cov_outer_full_upper_arr),
        "cov_outer_upper_std_open":  np.nanstd(cov_outer_full_upper_arr, ddof=1),

        "outer_width_mean_mean_open":   np.nanmean(outer_width_full_mean_arr),
        "outer_width_mean_std_open":    np.nanstd(outer_width_full_mean_arr, ddof=1),

        "outer_width_median_mean_open": np.nanmean(outer_width_full_median_arr),
        "outer_width_median_std_open":  np.nanstd(outer_width_full_median_arr, ddof=1),
    }

    # =========================================================
    # 3) POINTWISE coverage (NEW)
    # =========================================================
    cov_outer_t = compute_pointwise_cp_coverage(mc_results)
    cov_outer_lower_t, cov_outer_upper_t = (
        compute_pointwise_one_sided_coverage(mc_results)
    )

    pointwise_results = {
        "cov_outer_t":       cov_outer_t,
        "cov_outer_lower_t": cov_outer_lower_t,
        "cov_outer_upper_t": cov_outer_upper_t,
    }


    return summary, mc_results, pointwise_results













################################################################################
################################################################################    










T = 3000
n_mc = 500

mu = 0.5
sd = 1.0


phi=0.9
df=5
lam=-3

### parametri non utilizzati in quanto sto con empirical quantile e full history

caviar_model="SAV"
G=10


lookback_aci = 60
lookback = lookback_aci
########


p = 1                  # AR order

alpha_target = 0.05     # target miscoverage


#eta          = 0.005
#eta_iid=0.0


alpha_target_open = alpha_target
alpha_target_classic = 2*alpha_target



I = 500
#gammas = None
gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)
warm_up_0=2
warm_up=10




summary_normal_ar1, res_normal_ar1,pointwise_res_normal_ar1 = run_mc_cp_coverage_open(
    generator_func=sim_normal_ar1,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sigma=sd,
    phi=phi
)



summary_t_ar1, res_t_ar1,pointwise_res_t_ar1= run_mc_cp_coverage_open(
    generator_func=sim_t_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    phi=phi
)


summary_skewt_ar1, res_skewt_ar1,pointwise_res_skewt_ar1 = run_mc_cp_coverage_open(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    lam=lam,
    phi=phi
)


summary_skewt_ar1_max, res_skewt_ar1_max,pointwise_res_skewt_ar1_max = run_mc_cp_coverage_open_max(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    lam=lam,
    phi=phi
)

def print_summary(name, summary):
    print(f"=== {name} ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.4f}")
    print()  # riga vuota


print_summary("Normal AR(1) (open)", summary_normal_ar1)

print_summary("Student-t AR(1) (open)", summary_t_ar1)

print_summary("Skew-t AR(1) (open)", summary_skewt_ar1)

print_summary("Skew-t AR(1) MAX (open)", summary_skewt_ar1_max)



summary_normal_ar1_CLASSIC, res_normal_ar1_CLASSIC,pointwise_res_normal_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_normal_ar1,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sigma=sd,
    phi=phi
)





summary_t_ar1_CLASSIC, res_t_ar1_CLASSIC,pointwise_res_t_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_t_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    phi=phi
)




summary_skewt_ar1_CLASSIC, res_skewt_ar1_CLASSIC,pointwise_res_skewt_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    caviar_model=caviar_model,
    G=G,
    I=I,
    gammas=gammas,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    lam=lam,
    phi=phi
)


def print_summary(name, summary):
    print(f"=== {name} ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.4f}")
    print()  # riga vuota



print_summary("Normal AR(1) (CLASSIC)", summary_normal_ar1_CLASSIC)
print_summary("Student-t AR(1) (CLASSIC)", summary_t_ar1_CLASSIC)
print_summary("Skew-t AR(1) (CLASSIC)", summary_skewt_ar1_CLASSIC)



########################################################
########################################################    





# OPEN AR




def rolling_ar_forecast(
    y_window,
    p=1,
    alpha=0.10
):
    """
    Fit AR(p) on y_window and produce:
      - point forecast m_t
      - parametric prediction interval (L_hat, U_hat)
      - predictive std sigma_pred (implicit)

    Returns:
      m_t, L_hat, U_hat, sigma_pred
    """

    y = np.asarray(y_window, float)

    if len(y) <= p + 2:
        # fallback
        m_t = float(np.mean(y))
        sigma = float(np.std(y, ddof=1)) if len(y) > 1 else 1e-6
        z = 1.96  # approx
        return m_t, m_t - z * sigma, m_t + z * sigma, sigma

    model = AutoReg(y, lags=p, trend="c", old_names=False)
    res = model.fit()

    # one-step-ahead forecast
    pred = res.get_prediction(start=len(y), end=len(y))
    m_t = float(pred.predicted_mean[0])
    


    # QUESTI SONO I REALI PREDICTION INTERVALS CHE USO NEL NAIVE METHOD

    # prediction interval (parametric)
    ci = pred.conf_int(alpha=alpha)
    L_hat = float(ci[0, 0])
    U_hat = float(ci[0, 1])



    # QUESTI INTERVALLI LI CALCOLO SOLAMENTE PER CALCOLARE LA VARIANZA
    # prediction interval (parametric)
    ci_var = pred.conf_int(alpha=0.1)
    L_hat_var = float(ci_var[0, 0])
    U_hat_var = float(ci_var[0, 1])

    # predictive std (approx)
    sigma_pred = float((U_hat_var - L_hat_var) / (2 * 1.645))  # rough, ok for scaling



    return m_t, L_hat, U_hat, sigma_pred



# MODELLO 0 — Naive AR + parametric prediction interval (NO CP)
def rolling_ar_naive(
    y_obs,
    lookback=240,
    p=1,
    alpha=0.10
):
    """
    Naive baseline:
      AR(p) + parametric prediction interval
      NO conformal, NO ACI
    """

    # y = np.asarray(y_obs, float)
    # window = y[:-1][-lookback:]
    
    window = np.asarray(y_obs[:-1], float)
    
    # y_t = y_obs[-1]

    m_t, L_t, U_t, _ = rolling_ar_forecast(
        window, p=p, alpha=alpha #QUI alpha=alpha_target LO USO...
    )

    return float(m_t), float(L_t), float(U_t)



def rolling_ar_parametric_faci(
    y_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=0.01,
    lookback=240,
    p=1
):
    """
    Rolling AR(p) + parametric PI + FACI (alpha-based),
    DtACI-faithful version.

    Score:
        s_t = |y_t - m_t| / sigma_pred
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Base AR predictor
    # ======================================================
    window = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])

    m_t,L_raw, U_raw, sigma_pred = rolling_ar_forecast(
        window, p=p, alpha=alpha_target
    )
    sigma_pred = max(sigma_pred, 1e-8)

    # ======================================================
    # Expert probabilities + aggregated alpha
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = np.sum(expert_probs * expert_alphas)



    # ======================================================
    # Conformal calibration
    # ======================================================
    nc = len(scores_history)

    
    if nc == 0:
        qn = None
    else:
        
        qn_idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(scores_history)[qn_idx])

    # ======================================================
    # Predictive interval
    # ======================================================
    if qn is None:
        L_t = float(L_raw)
        U_t = float(U_raw)
    else:
        L_t = float(m_t - qn * sigma_pred)
        U_t = float(m_t + qn * sigma_pred)



    # ======================================================
    # Current score
    # ======================================================
    score_t = abs(y_t - m_t) / sigma_pred
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update (DtACI style)
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(scores_history[:-1])
        beta_t = float(np.mean(past_scores >= score_t))

        # ----- pinball losses
        expert_losses = pinball_loss(
            beta_t,
            expert_alphas,
            alpha_target
        )

        # ----- update expert alphas (local ACI)
        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)

        # ----- weight update (EXACT DtACI)
        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws

    return float(m_t), float(L_t), float(U_t), faci_state, scores_history






def rolling_ar_symmetric_faci(
    y_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=0.01,
    lookback=240,
    p=1
):
    """
    Rolling AR(p) + symmetric residual CP + FACI (alpha-based),
    DtACI-faithful version.

    Score:
        s_t = |y_t - m_t|
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Base AR predictor
    # ======================================================
    window = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])

    m_t, L_raw, U_raw, _ = rolling_ar_forecast(
        window, p=p, alpha=alpha_target
    )

    # ======================================================
    # Expert probabilities + aggregated alpha
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = np.sum(expert_probs * expert_alphas)

    # ======================================================
    # Conformal calibration
    # ======================================================
    nc = len(scores_history)


    # ======================================================
    # Conformal quantile
    # ======================================================
    if nc == 0:
        qn = None
    else:
        qn_idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(scores_history)[qn_idx])

    # ======================================================
    # Predictive interval
    # ======================================================
    if qn is None:
        L_t = float(L_raw)
        U_t = float(U_raw)
    else:
        L_t = float(m_t - qn)
        U_t = float(m_t + qn)




    # ======================================================
    # Current score
    # ======================================================
    score_t = abs(y_t - m_t)
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update (DtACI style)
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(scores_history[:-1])
        beta_t = float(np.mean(past_scores >= score_t))

        # ----- pinball losses
        expert_losses = pinball_loss(
            beta_t,
            expert_alphas,
            alpha_target
        )

        # ----- update expert alphas (local ACI)
        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)

        # ----- weight update (EXACT DtACI)
        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws

    return float(m_t), float(L_t), float(U_t), faci_state, scores_history


def rolling_ar_one_sided_faci(
    y_obs,
    scores_low,
    scores_up,
    faci_state_low,
    faci_state_up,
    alpha_low_target=0.10,
    alpha_up_target=0.10,
    eta=2.72,
    sigma=0.01,
    lookback=240,
    p=1
):
    """
    Rolling AR(p) + one-sided FACI (alpha-based),
    DtACI-faithful version (NON standardizzato).

    Scores:
        lower: s_t^L = m_t - y_t
        upper: s_t^U = y_t - m_t
    """

    # ======================================================
    # Base AR predictor
    # ======================================================
    window = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])

    m_t, L_raw, U_raw, _ = rolling_ar_forecast(
        window, p=p, alpha=min(alpha_low_target, alpha_up_target)
    )

    # ======================================================
    # -------- LOWER SIDE ----------------------------------
    # ======================================================
    gammas = faci_state_low["gammas"]
    expert_alphas = faci_state_low["expert_alphas"]
    expert_ws = faci_state_low["expert_ws"]
    k = faci_state_low["k"]

    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar_low = np.sum(expert_probs * expert_alphas)



    
    
    # ================= LOWER SIDE =================
    nc = len(scores_low)

    if nc == 0:
        q_low = None
    else:
     
        idx = int(np.ceil((1.0 - alpha_bar_low) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        q_low = float(np.sort(scores_low)[idx])

    # ---- Bound ----
    if q_low is None:
        L_t = float(L_raw)   # fallback parametrico
    else:
        L_t = float(m_t - q_low)

    # ---- Score ----
    score_low = float(m_t - y_t)


    
    
    
    scores_low.append(float(score_low))
    
    
    
    
    

    if nc > 0:
        past = np.asarray(scores_low[:-1])
        beta_t = float(np.mean(past >= score_low))

        expert_losses = pinball_loss(
            beta_t, expert_alphas, alpha_low_target
        )

        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_low_target - err_i)

        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    faci_state_low["expert_alphas"] = expert_alphas
    faci_state_low["expert_ws"] = expert_ws

    # ======================================================
    # -------- UPPER SIDE ----------------------------------
    # ======================================================
    gammas = faci_state_up["gammas"]
    expert_alphas = faci_state_up["expert_alphas"]
    expert_ws = faci_state_up["expert_ws"]
    k = faci_state_up["k"]

    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar_up = np.sum(expert_probs * expert_alphas)


    
    # ================= UPPER SIDE =================
    nc = len(scores_up)

    if nc == 0:
        q_up = None
    else:
       
        idx = int(np.ceil((1.0 - alpha_bar_up) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        q_up = float(np.sort(scores_up)[idx])

    # ---- Bound ----
    if q_up is None:
        U_t = float(U_raw)   # fallback parametrico
    else:
        U_t = float(m_t + q_up)

    # ---- Score ----
    score_up = float(y_t - m_t)

    
    
    scores_up.append(float(score_up))

    if nc > 0:
        past = np.asarray(scores_up[:-1])
        beta_t = float(np.mean(past >= score_up))

        expert_losses = pinball_loss(
            beta_t, expert_alphas, alpha_up_target
        )

        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_up_target - err_i)

        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    faci_state_up["expert_alphas"] = expert_alphas
    faci_state_up["expert_ws"] = expert_ws

    return (
        float(m_t),
        float(L_t),
        float(U_t),
        faci_state_low,
        faci_state_up,
        scores_low,
        scores_up
    )
    
    
def rolling_ar_one_sided_std_faci(
    y_obs,
    scores_low,
    scores_up,
    faci_state_low,
    faci_state_up,
    alpha_low_target=0.10,
    alpha_up_target=0.10,
    eta=2.72,
    sigma=0.01,
    lookback=240,
    p=1
):
    """
    Rolling AR(p) + one-sided FACI with STANDARDIZED scores,
    DtACI-faithful version.

    Scores:
        lower: (m_t - y_t) / sigma_pred
        upper: (y_t - m_t) / sigma_pred
    """

    # ======================================================
    # Base AR predictor
    # ======================================================
    window = np.asarray(y_obs[:-1], dtype=float)
    y_t = float(y_obs[-1])

    m_t, L_raw, U_raw, sigma_pred = rolling_ar_forecast(
        window, p=p, alpha=min(alpha_low_target, alpha_up_target)
    )
    sigma_pred = max(sigma_pred, 1e-8)

    # ======================================================
    # -------- LOWER SIDE ----------------------------------
    # ======================================================
    gammas = faci_state_low["gammas"]
    expert_alphas = faci_state_low["expert_alphas"]
    expert_ws = faci_state_low["expert_ws"]
    k = faci_state_low["k"]

    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar_low = np.sum(expert_probs * expert_alphas)


    
    # ================= LOWER SIDE =================
    nc = len(scores_low)

    if nc == 0:
        q_low = None
    else:
        
        idx = int(np.ceil((1.0 - alpha_bar_low) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        q_low = float(np.sort(scores_low)[idx])

    # ---- Bound ----
    if q_low is None:
        L_t = float(L_raw)   # fallback parametrico
    else:
        L_t = float(m_t - q_low * sigma_pred)

    # ---- Score ----
    score_low = float((m_t - y_t) / sigma_pred)

    
    
    
    scores_low.append(float(score_low))

    if nc > 0:
        past = np.asarray(scores_low[:-1])
        beta_t = float(np.mean(past >= score_low))

        expert_losses = pinball_loss(
            beta_t, expert_alphas, alpha_low_target
        )

        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_low_target - err_i)

        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    faci_state_low["expert_alphas"] = expert_alphas
    faci_state_low["expert_ws"] = expert_ws

    # ======================================================
    # -------- UPPER SIDE ----------------------------------
    # ======================================================
    gammas = faci_state_up["gammas"]
    expert_alphas = faci_state_up["expert_alphas"]
    expert_ws = faci_state_up["expert_ws"]
    k = faci_state_up["k"]

    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar_up = np.sum(expert_probs * expert_alphas)


    
    # ================= UPPER SIDE =================
    nc = len(scores_up)

    if nc == 0:
        q_up = None
    else:
        idx = int(np.ceil((1.0 - alpha_bar_up) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        q_up = float(np.sort(scores_up)[idx])

    # ---- Bound ----
    if q_up is None:
        U_t = float(U_raw)   # fallback parametrico
    else:
        U_t = float(m_t + q_up * sigma_pred)

    # ---- Score ----
    score_up = float((y_t - m_t) / sigma_pred)

    
    
    scores_up.append(float(score_up))

    if nc > 0:
        past = np.asarray(scores_up[:-1])
        beta_t = float(np.mean(past >= score_up))

        expert_losses = pinball_loss(
            beta_t, expert_alphas, alpha_up_target
        )

        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_up_target - err_i)

        log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
        log_weights -= np.max(log_weights)
        expert_bar_ws = np.exp(log_weights)

        expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
        expert_ws = (
            (1.0 - sigma) * expert_bar_ws / expert_bar_ws_sum
            + sigma / k
        )

    faci_state_up["expert_alphas"] = expert_alphas
    faci_state_up["expert_ws"] = expert_ws

    return (
        float(m_t),
        float(L_t),
        float(U_t),
        faci_state_low,
        faci_state_up,
        scores_low,
        scores_up
    )






def _alpha_bar_from_state(st):
    """
    Compute aggregated alpha from FACI/DTACI state.
    Works with either:
      - expert_ws  (DtACI-faithful)
      - log_w      (legacy FACI)
    """
    a = st["expert_alphas"]

    if "expert_ws" in st:  # DtACI / close version
        ws = st["expert_ws"]
        p = ws / (np.sum(ws) + 1e-300)
    else:                  # legacy log_w
        log_w = st["log_w"]
        p = np.exp(log_w - logsumexp(log_w))

    return float(np.sum(p * a))



def cp_faci_ar(
    spread,
    model_type="ar_symmetric_faci",
    lookback=240,
    alpha_target=0.10,
    p=1,
    gammas=None,
    I=100,
    warm_up_0=1,
    warm_up=20
):
    """
    Pairs trading with AR-based predictive bands + FACI / DtACI-style aggregation.

    Two-stage warm-up:
      - [warm_up_0, warm_up): shadow training (populate score buffers / stabilize FACI states)
      - [warm_up, T-1): main loop (store results)
    """

    spread = np.asarray(spread, float)
    T = len(spread)


    # -------------------------------------------------
    # 1) Default gammas
    # -------------------------------------------------
    if gammas is None:
        gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)

    k = len(gammas)

    # -------------------------------------------------
    # 2) FACI theoretical hyperparameters
    # -------------------------------------------------
    sigma = sigma_faci(I)
    eta = eta_faci(alpha_target=alpha_target, I=I, k=k)

    # -------------------------------------------------
    # 3) Allocations
    # -------------------------------------------------
    L_arr = np.full(T, np.nan)
    U_arr = np.full(T, np.nan)

    alpha_low_arr = np.full(T, np.nan)
    alpha_up_arr  = np.full(T, np.nan)

    # -------------------------------------------------
    # 4) States + score histories
    # -------------------------------------------------
    scores = []
    scores_low, scores_up = [], []

    faci_state     = init_faci_state(gammas, alpha_target)
    faci_state_low = init_faci_state(gammas, alpha_target)
    faci_state_up  = init_faci_state(gammas, alpha_target)

    # =====================================================
    # 5A) PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[: t + 1]

        if model_type == "ar_naive":

            m_t, L_t, U_t = rolling_ar_naive(
                y_obs, lookback=lookback, p=p, alpha=alpha_target
            )
            alpha_bar = alpha_target

        elif model_type == "ar_parametric_faci":

            m_t, L_t, U_t, faci_state, scores = rolling_ar_parametric_faci(
                y_obs=y_obs,
                scores_history=scores,
                faci_state=faci_state,
                alpha_target=alpha_target,
                eta=eta,
                sigma=sigma,
                lookback=lookback,
                p=p
            )
            alpha_bar = _alpha_bar_from_state(faci_state)

        elif model_type == "ar_symmetric_faci":

            m_t, L_t, U_t, faci_state, scores = rolling_ar_symmetric_faci(
                y_obs=y_obs,
                scores_history=scores,
                faci_state=faci_state,
                alpha_target=alpha_target,
                eta=eta,
                sigma=sigma,
                lookback=lookback,
                p=p
            )
            alpha_bar = _alpha_bar_from_state(faci_state)

        elif model_type == "ar_one_sided_faci":

            m_t, L_t, U_t, faci_state_low, faci_state_up, scores_low, scores_up = (
                rolling_ar_one_sided_faci(
                    y_obs=y_obs,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    faci_state_low=faci_state_low,
                    faci_state_up=faci_state_up,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    eta=eta,
                    sigma=sigma,
                    lookback=lookback,
                    p=p
                )
            )
            alpha_low_arr[t] = _alpha_bar_from_state(faci_state_low)
            alpha_up_arr[t]  = _alpha_bar_from_state(faci_state_up)

        elif model_type == "ar_one_sided_std_faci":

            m_t, L_t, U_t, faci_state_low, faci_state_up, scores_low, scores_up = (
                rolling_ar_one_sided_std_faci(
                    y_obs=y_obs,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    faci_state_low=faci_state_low,
                    faci_state_up=faci_state_up,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    eta=eta,
                    sigma=sigma,
                    lookback=lookback,
                    p=p
                )
            )
            alpha_low_arr[t] = _alpha_bar_from_state(faci_state_low)
            alpha_up_arr[t]  = _alpha_bar_from_state(faci_state_up)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")



    # =====================================================
    # 5B) MAIN LOOP (STORE RESULTS)
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[: t + 1]

        if model_type == "ar_naive":

            m_t, L_t, U_t = rolling_ar_naive(
                y_obs, lookback=lookback, p=p, alpha=alpha_target
            )
            alpha_bar = alpha_target

        elif model_type == "ar_parametric_faci":

            m_t, L_t, U_t, faci_state, scores = rolling_ar_parametric_faci(
                y_obs=y_obs,
                scores_history=scores,
                faci_state=faci_state,
                alpha_target=alpha_target,
                eta=eta,
                sigma=sigma,
                lookback=lookback,
                p=p
            )
            alpha_bar = _alpha_bar_from_state(faci_state)

        elif model_type == "ar_symmetric_faci":

            m_t, L_t, U_t, faci_state, scores = rolling_ar_symmetric_faci(
                y_obs=y_obs,
                scores_history=scores,
                faci_state=faci_state,
                alpha_target=alpha_target,
                eta=eta,
                sigma=sigma,
                lookback=lookback,
                p=p
            )
            alpha_bar = _alpha_bar_from_state(faci_state)

        elif model_type == "ar_one_sided_faci":

            m_t, L_t, U_t, faci_state_low, faci_state_up, scores_low, scores_up = (
                rolling_ar_one_sided_faci(
                    y_obs=y_obs,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    faci_state_low=faci_state_low,
                    faci_state_up=faci_state_up,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    eta=eta,
                    sigma=sigma,
                    lookback=lookback,
                    p=p
                )
            )
            alpha_low_arr[t] = _alpha_bar_from_state(faci_state_low)
            alpha_up_arr[t]  = _alpha_bar_from_state(faci_state_up)

        elif model_type == "ar_one_sided_std_faci":

            m_t, L_t, U_t, faci_state_low, faci_state_up, scores_low, scores_up = (
                rolling_ar_one_sided_std_faci(
                    y_obs=y_obs,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    faci_state_low=faci_state_low,
                    faci_state_up=faci_state_up,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    eta=eta,
                    sigma=sigma,
                    lookback=lookback,
                    p=p
                )
            )
            alpha_low_arr[t] = _alpha_bar_from_state(faci_state_low)
            alpha_up_arr[t]  = _alpha_bar_from_state(faci_state_up)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Save bounds
        L_arr[t] = L_t
        U_arr[t] = U_t

        # Save alpha diagnostics (two-sided)
        if model_type in {"ar_naive", "ar_parametric_faci", "ar_symmetric_faci"}:
            alpha_low_arr[t] = alpha_bar
            alpha_up_arr[t]  = alpha_bar

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    return pd.DataFrame({
        "spread": spread,
        "L": L_arr,
        "U": U_arr,
        "alpha_out_low": alpha_low_arr,
        "alpha_out_up":  alpha_up_arr,
    }).iloc[warm_up:]





def run_mc_cp_coverage_ar(
    generator_func,
    n_mc,
    T,
    lookback,
    alpha_outer_target,
    p,
    model_type,
    gammas,
    I,
    warm_up_0=1,
    warm_up=20,
    **generator_kwargs
):
    """
    Monte Carlo evaluation of CP–FACI coverage properties
    for AR-based predictive bands.

    Returns
    -------
    summary : dict
        Mean/std of classic coverage and width metrics.
    mc_results : list of dict
        Per-replication metrics + raw series (spread, L, U).
    pointwise_results : dict
        Pointwise coverage arrays.
    """

    mc_results = []

    # =========================================================
    # 1) Monte Carlo loop
    # =========================================================
    for mc in range(n_mc):
        np.random.seed(mc)

        gen_kwargs = dict(generator_kwargs)
        gen_kwargs["random_state"] = None

        # --- generate synthetic spread
        spread = generator_func(T=T, **gen_kwargs)

        # --- compute FACI AR-based bands
        bands_df = cp_faci_ar(
            spread=spread,
            model_type=model_type,
            lookback=lookback,
            alpha_target=alpha_outer_target,
            p=p,
            gammas=gammas,
            I=I,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
        )

        # --- classic (time-averaged) coverage + widths
        cov_info = compute_cp_coverage(bands_df)

        mc_results.append({
            "cov_outer_full":          cov_info["cov_outer_full"],
            "cov_outer_full_lower":    cov_info["cov_outer_full_lower"],
            "cov_outer_full_upper":    cov_info["cov_outer_full_upper"],
            "outer_width_full_mean":   cov_info["outer_width_full_mean"],
            "outer_width_full_median": cov_info["outer_width_full_median"],

            "spread": bands_df["spread"].values,
            "L":      bands_df["L"].values,
            "U":      bands_df["U"].values,
        })

    # =========================================================
    # 2) CLASSIC Monte Carlo summaries
    # =========================================================
    cov_outer_arr    = np.array([r["cov_outer_full"]       for r in mc_results], float)
    cov_lower_arr    = np.array([r["cov_outer_full_lower"] for r in mc_results], float)
    cov_upper_arr    = np.array([r["cov_outer_full_upper"] for r in mc_results], float)
    width_mean_arr   = np.array([r["outer_width_full_mean"] for r in mc_results], float)
    width_median_arr = np.array([r["outer_width_full_median"] for r in mc_results], float)

    summary = {
        f"cov_outer_mean ({model_type})":       np.nanmean(cov_outer_arr),
        f"cov_outer_std ({model_type})":        np.nanstd(cov_outer_arr, ddof=1),

        f"cov_outer_lower_mean ({model_type})": np.nanmean(cov_lower_arr),
        f"cov_outer_lower_std ({model_type})":  np.nanstd(cov_lower_arr, ddof=1),

        f"cov_outer_upper_mean ({model_type})": np.nanmean(cov_upper_arr),
        f"cov_outer_upper_std ({model_type})":  np.nanstd(cov_upper_arr, ddof=1),

        f"outer_width_mean_mean ({model_type})":   np.nanmean(width_mean_arr),
        f"outer_width_mean_std ({model_type})":    np.nanstd(width_mean_arr, ddof=1),

        f"outer_width_median_mean ({model_type})": np.nanmean(width_median_arr),
        f"outer_width_median_std ({model_type})":  np.nanstd(width_median_arr, ddof=1),
    }

    # =========================================================
    # 3) POINTWISE coverage
    # =========================================================
    cov_outer_t = compute_pointwise_cp_coverage(mc_results)
    cov_outer_lower_t, cov_outer_upper_t = (
        compute_pointwise_one_sided_coverage(mc_results)
    )

    pointwise_results = {
        "cov_outer_t":       cov_outer_t,
        "cov_outer_lower_t": cov_outer_lower_t,
        "cov_outer_upper_t": cov_outer_upper_t,
    }

    return summary, mc_results, pointwise_results





model_types = [
    #"ar_naive",
    "ar_parametric_faci",
    "ar_symmetric_faci",
    "ar_one_sided_faci",
    "ar_one_sided_std_faci",
]


two_sided_models = {
    #"ar_naive",
    "ar_parametric_faci",
    "ar_symmetric_faci",
}




generators = {

    "Normal AR(1)": (
        sim_normal_ar1,
        dict(mu=mu, sigma=sd, phi=phi),
        "ar"
    ),

    "Student-t AR(1)": (
        sim_t_ar1_locscale,
        dict(mu=mu, sd=sd, df=df, phi=phi),
        "ar"
    ),

    "Skew-t AR(1)": (
        sim_skewt_ar1_locscale,
        dict(mu=mu, sd=sd, df=df, lam=lam, phi=phi),
        "ar"
    ),
}



#####
TARGET_GENERATORS = {
    "Skew-t AR(1)",
}
pointwise_store = {}
#####

mc_summaries = {}

for model_type in model_types:

    print(f"\n==============================")
    print(f"MODEL: {model_type}")
    print(f"==============================")

    mc_summaries[model_type] = {}

    # -------------------------
    # ALPHA SELEZIONATO IN BASE AL TIPO DI MODELLO
    # -------------------------
    alpha_used = (
        2 * alpha_target
        if model_type in two_sided_models
        else alpha_target
    )

    for gen_name, (gen_func, gen_kwargs, gen_type) in generators.items():

        summary, _, pointwise = run_mc_cp_coverage_ar(
            generator_func=gen_func,
            n_mc=n_mc,
            T=T,
            lookback=lookback,
            alpha_outer_target=alpha_used,
            p=p,
            model_type=model_type,
            gammas=gammas,
            I=I,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
            **gen_kwargs
        )

        # -------------------------
        # Salvataggio pointwise selettivo
        # -------------------------
        if gen_name in TARGET_GENERATORS:
            pointwise_store.setdefault(model_type, {})
            pointwise_store[model_type][gen_name] = pointwise

        mc_summaries[model_type][gen_name] = summary

        print(
            f"\n--- {gen_name} ({gen_type.upper()}, "
            f"alpha={alpha_used}) ---"
        )
        for k, v in summary.items():
            print(f"{k:30s}: {v:.4f}")


def print_summary_table(mc_summaries):
    for model, res in mc_summaries.items():
        print(f"\n==================== {model} ====================")
        for gen, summ in res.items():
            print(f"\n{gen}")
            for k, v in summ.items():
                print(f"  {k:28s}: {v:.4f}")



print_summary_table(mc_summaries)






