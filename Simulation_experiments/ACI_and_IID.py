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



def rolling_qr_aci(
    y_obs, alpha_t,
    scores_history,
    lookback=240,
    eta=0.01, alpha_target=0.10,
    caviar_model="SAV", G=10.0
):

    """
    Rolling CAViaR forecasting (low/high) + ACI + optional scaling.
    - Sostituisce i quantili empirici con previsioni CAViaR (VaR_{t|t-1}).
    - Usa SOLO score passati in calibrazione (come nel tuo codice originale).
    Ritorna: mu_med_t, L_t, U_t, ESI_t, alpha_t_new, scores_history
    """

    model = caviar_model

    # 1) Finestra rolling [t-lookback, t-1]

    # Z = y_obs[:-1]
    # window_data = np.asarray(Z[-lookback:], dtype=float)

    window_data = np.asarray(y_obs[:-1], float)    ############################ FULL HISTORY



    y_t = float(y_obs[-1])


    tau_low, tau_high = alpha_target/2.0, 1.0 - alpha_target/2.0


    ########## PREDITTORE: QUANTILE EMPIRICO

    # q_low_t = np.quantile(window_data, tau_low)
    # q_high_t = np.quantile(window_data, tau_high)
    
    
    _, q_low_t, q_high_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha=alpha_target # QUI alpha=alpha_target LO USO...
    )
    


    # 3) Calibrazione CP: usa SOLO score passati
    #recent_scores = scores_history[-lookback:]

    recent_scores = scores_history############################ FULL HISTORY


    nc = len(recent_scores)
    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])

    # 4) Intervallo predittivo
    L_t = float(q_low_t  - qn)
    U_t = float(q_high_t + qn)



    # 5) Ora calcolo lo score corrente e lo APPENDO
    score_t = max(q_low_t - y_t, y_t - q_high_t)
    scores_history.append(float(score_t))           # ← dopo


    # 6) ACI update
    covered = 1 if (L_t <= y_t <= U_t) else 0
    err = 1 - covered
    alpha_t_new = float(alpha_t + eta * (alpha_target - err))
    #alpha_t_new = float(np.clip(alpha_t + eta * (alpha_target - err), 1e-4, 0.5))

    # 7) ESI
    den = max(abs(L_t), 1e-9)
    ESI_t = float(U_t / den)


    return float(L_t), float(U_t), float(ESI_t), float(alpha_t_new), scores_history




def cp_aci(
    spread,
    lookback_aci=240,
    alpha_target=0.1,          # α_out (0.1 -> 90% coverage)
    eta=0.01,
    caviar_model="SAV",
    G=10,
    warm_up_0=1,
    warm_up=20
):
    """
    ACI + one-sided/two-sided bands for a spread.
    Introduces a two-stage warm-up:
      - [warm_up_0, warm_up): shadow training to populate scores_history / stabilize alpha
      - [warm_up, T-1): main loop (results are considered usable)
    """

    spread = np.asarray(spread, float)
    T = len(spread)

    # -----------------------------------------
    # Output arrays
    # -----------------------------------------
    L_out_arr = np.full(T, np.nan)
    U_out_arr = np.full(T, np.nan)
    alpha_out_arr = np.full(T, np.nan)

    # -----------------------------------------
    # ACI state (outer)
    # -----------------------------------------
    alpha_out_t = float(alpha_target)
    scores_outer = []

    # =====================================================
    # 1) SHADOW TRAINING / PRE-WARM
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        L_out_t, U_out_t, ESI_out, alpha_out_new, scores_outer = rolling_qr_aci(
            y_obs=y_obs,
            alpha_t=alpha_out_t,
            scores_history=scores_outer,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            caviar_model=caviar_model,
            G=G
        )

        alpha_out_t = float(alpha_out_new)



    # (diagnostica utile)
    print("Initial nc (outer):", len(scores_outer), "warm_up:", warm_up)

    # =====================================================
    # 2) MAIN LOOP
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        L_out_t, U_out_t, ESI_out, alpha_out_new, scores_outer = rolling_qr_aci(
            y_obs=y_obs,
            alpha_t=alpha_out_t,
            scores_history=scores_outer,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            caviar_model=caviar_model,
            G=G
        )

        alpha_out_t = float(alpha_out_new)

        L_out_arr[t] = L_out_t
        U_out_arr[t] = U_out_t
        alpha_out_arr[t] = alpha_out_t

    result_df = pd.DataFrame(
        {
            "spread": spread,
            "L": L_out_arr,
            "U": U_out_arr,
            "alpha_outer": alpha_out_arr,
        }
    ).iloc[warm_up:]
    return result_df








def rolling_qr_aci_open(
    y_obs,
    alpha_t,
    scores_history,
    lookback=240,
    eta=0.01,
    alpha_target=0.10,
    side="lower",              # NEW: specifica se vogliamo il bound lower o upper
    caviar_model="SAV",
    G=10.0
):


    # ---------------------------
    # 1) Finestra rolling [t-lookback, t-1]
    # ---------------------------
    #Z = y_obs[:-1]
    #window_data = np.asarray(Z[-lookback:], dtype=float)
    window_data = np.asarray(y_obs[:-1], float)    ############################ FULL HISTORY


    y_t = float(y_obs[-1])

    # ---------------------------
    # 2) Quantile "grezzo" q_t
    #    (qui uso ancora quantile empirico; puoi riattivare CAViaR più avanti)
    # ---------------------------

    
    if side == "lower":
        # tail sinistra: quantile al livello alpha_target (es. 0.1 → 10%-quantile)
        #tau = alpha_target
        #q_t = float(np.quantile(window_data, tau))
        
        _, q_t, _, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
        
    elif side == "upper":
        # tail destra: quantile al livello 1 - alpha_target (es. 0.9)
        #tau = 1.0 - alpha_target
        #q_t = float(np.quantile(window_data, tau))
        
        _, _ , q_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
        
    else:
        raise ValueError(f"rolling_qr_aci: side deve essere 'lower' o 'upper', got {side}")
    # Se vuoi riattivare CAViaR, qui dovresti stimare q_t con il modello.
    # (Ho lasciato il codice CAViaR commentato nella tua versione precedente.)

    # ---------------------------
    # 3) Calibrazione CP one–sided (solo score passati)
    # ---------------------------
    
    #recent_scores = scores_history[-lookback:]
    recent_scores = scores_history ############################ FULL HISTORY

    nc = len(recent_scores)

    if nc == 0:
        qn = 0.0
    else:
        # stesso schema ACI: quantile (1 - alpha_t) degli score passati
        qn_idx = int(np.ceil((1.0 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])

    # ---------------------------
    # 4) Costruzione del bound B_t e calcolo dello score corrente
    # ---------------------------
    if side == "lower":
        # bound inferiore: B_t = q_t - qn
        B_t = float(q_t - qn)
   
     
        #score_t = max(q_t - y_t, 0)
        score_t = q_t - y_t


        # evento di "copertura" (per ACI): y_t >= B_t
        covered = 1 if (y_t >= B_t) else 0

    else:  # side == "upper"
        # bound superiore: B_t = q_t + qn
        B_t = float(q_t + qn)
    
       
        #score_t = max(y_t - q_t, 0)
        score_t = y_t - q_t
        

        # evento di "copertura": y_t <= B_t
        covered = 1 if (y_t <= B_t) else 0

    scores_history.append(float(score_t))

    # ---------------------------
    # 5) ACI update (stesso schema di prima, ma per bound one–sided)
    # ---------------------------
    err = 1 - covered                     # 1 se violazione, 0 se coperto
     
    alpha_t_new = float(alpha_t + eta * (alpha_target - err))
    
    return B_t, alpha_t_new, scores_history





def cp_aci_open(
    spread,
    lookback_aci=240,
    alpha_target=0.1,          # α_out (e.g. 0.1 -> 90% one-sided bound)
    eta=0.01,
    caviar_model="SAV",
    G=10,
    warm_up_0=1,
    warm_up=20
):


    spread = np.asarray(spread, float)
    T = len(spread)


    # -----------------------------------------
    # Output arrays
    # -----------------------------------------
    L_out_arr          = np.full(T, np.nan)
    U_out_arr          = np.full(T, np.nan)
    alpha_out_low_arr  = np.full(T, np.nan)
    alpha_out_up_arr   = np.full(T, np.nan)

    # -----------------------------------------
    # Separate ACI states (lower/upper)
    # -----------------------------------------
    alpha_out_low_t = float(alpha_target)
    alpha_out_up_t  = float(alpha_target)

    scores_out_low = []   # score buffer for lower tail
    scores_out_up  = []   # score buffer for upper tail

    # =====================================================
    # 1) PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        # --- OUTER LOWER ---
        L_out_t, alpha_out_low_t, scores_out_low = rolling_qr_aci_open(
            y_obs=y_obs,
            alpha_t=alpha_out_low_t,
            scores_history=scores_out_low,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # --- OUTER UPPER ---
        U_out_t, alpha_out_up_t, scores_out_up = rolling_qr_aci_open(
            y_obs=y_obs,
            alpha_t=alpha_out_up_t,
            scores_history=scores_out_up,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )


    # =====================================================
    # 2) MAIN LOOP
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        # --- OUTER LOWER ---
        L_out_t, alpha_out_low_t, scores_out_low = rolling_qr_aci_open(
            y_obs=y_obs,
            alpha_t=alpha_out_low_t,
            scores_history=scores_out_low,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # --- OUTER UPPER ---
        U_out_t, alpha_out_up_t, scores_out_up = rolling_qr_aci_open(
            y_obs=y_obs,
            alpha_t=alpha_out_up_t,
            scores_history=scores_out_up,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )

        # save
        L_out_arr[t]         = L_out_t
        U_out_arr[t]         = U_out_t
        alpha_out_low_arr[t] = alpha_out_low_t
        alpha_out_up_arr[t]  = alpha_out_up_t

    result_df = pd.DataFrame(
        {
            "spread": spread,
            "L": L_out_arr,
            "U": U_out_arr,
            "alpha_out_low": alpha_out_low_arr,
            "alpha_out_up": alpha_out_up_arr,
        }
    ).iloc[warm_up:]

    return result_df




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
    eta,
    caviar_model,
    G,
    warm_up_0=1,
    warm_up=20,
    plot_pointwise=True,
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
        trading_df = cp_aci(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            eta=eta,
            caviar_model=caviar_model,
            G=G,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
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
    eta,
    caviar_model,
    G,
    warm_up_0=1,
    warm_up=20,
    plot_pointwise=True,
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
        t_df = cp_aci_open(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            eta=eta,
            caviar_model=caviar_model,
            G=G,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
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






#########################################################################
#########################################################################
#########################################################################
















def rolling_qr_aci_open_max(
    y_obs,
    alpha_t,
    scores_history,
    lookback=240,
    eta=0.01,
    alpha_target=0.10,
    side="lower",              # NEW: specifica se vogliamo il bound lower o upper
    caviar_model="SAV",
    G=10.0
):


    # ---------------------------
    # 1) Finestra rolling [t-lookback, t-1]
    # ---------------------------
    #Z = y_obs[:-1]
    #window_data = np.asarray(Z[-lookback:], dtype=float)
    window_data = np.asarray(y_obs[:-1], float)    ############################ FULL HISTORY


    y_t = float(y_obs[-1])

    # ---------------------------
    # 2) Quantile "grezzo" q_t
    #    (qui uso ancora quantile empirico; puoi riattivare CAViaR più avanti)
    # ---------------------------

    
    if side == "lower":
        # tail sinistra: quantile al livello alpha_target (es. 0.1 → 10%-quantile)
        #tau = alpha_target
        #q_t = float(np.quantile(window_data, tau))
        
        _, q_t, _, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
        
    elif side == "upper":
        # tail destra: quantile al livello 1 - alpha_target (es. 0.9)
        #tau = 1.0 - alpha_target
        #q_t = float(np.quantile(window_data, tau))
        
        _, _ , q_t, _ = rolling_ar_forecast( 
        window_data, p=1, alpha= 2*alpha_target)
        
    else:
        raise ValueError(f"rolling_qr_aci: side deve essere 'lower' o 'upper', got {side}")
    # Se vuoi riattivare CAViaR, qui dovresti stimare q_t con il modello.
    # (Ho lasciato il codice CAViaR commentato nella tua versione precedente.)

    # ---------------------------
    # 3) Calibrazione CP one–sided (solo score passati)
    # ---------------------------
    
    #recent_scores = scores_history[-lookback:]
    recent_scores = scores_history ############################ FULL HISTORY

    nc = len(recent_scores)

    if nc == 0:
        qn = 0.0
    else:
        # stesso schema ACI: quantile (1 - alpha_t) degli score passati
        qn_idx = int(np.ceil((1.0 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])

    # ---------------------------
    # 4) Costruzione del bound B_t e calcolo dello score corrente
    # ---------------------------
    if side == "lower":
        # bound inferiore: B_t = q_t - qn
        B_t = float(q_t - qn)
   
     
        score_t = max(q_t - y_t, 0)



        # evento di "copertura" (per ACI): y_t >= B_t
        covered = 1 if (y_t >= B_t) else 0

    else:  # side == "upper"
        # bound superiore: B_t = q_t + qn
        B_t = float(q_t + qn)
    
       
        score_t = max(y_t - q_t, 0)
       
        

        # evento di "copertura": y_t <= B_t
        covered = 1 if (y_t <= B_t) else 0

    scores_history.append(float(score_t))

    # ---------------------------
    # 5) ACI update (stesso schema di prima, ma per bound one–sided)
    # ---------------------------
    err = 1 - covered                     # 1 se violazione, 0 se coperto
     
    alpha_t_new = float(alpha_t + eta * (alpha_target - err))
    
    return B_t, alpha_t_new, scores_history





def cp_aci_open_max(
    spread,
    lookback_aci=240,
    alpha_target=0.1,          # α_out (e.g. 0.1 -> 90% one-sided bound)
    eta=0.01,
    caviar_model="SAV",
    G=10,
    warm_up_0=1,
    warm_up=20
):


    spread = np.asarray(spread, float)
    T = len(spread)


    # -----------------------------------------
    # Output arrays
    # -----------------------------------------
    L_out_arr          = np.full(T, np.nan)
    U_out_arr          = np.full(T, np.nan)
    alpha_out_low_arr  = np.full(T, np.nan)
    alpha_out_up_arr   = np.full(T, np.nan)

    # -----------------------------------------
    # Separate ACI states (lower/upper)
    # -----------------------------------------
    alpha_out_low_t = float(alpha_target)
    alpha_out_up_t  = float(alpha_target)

    scores_out_low = []   # score buffer for lower tail
    scores_out_up  = []   # score buffer for upper tail

    # =====================================================
    # 1) PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        # --- OUTER LOWER ---
        L_out_t, alpha_out_low_t, scores_out_low = rolling_qr_aci_open_max(
            y_obs=y_obs,
            alpha_t=alpha_out_low_t,
            scores_history=scores_out_low,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # --- OUTER UPPER ---
        U_out_t, alpha_out_up_t, scores_out_up = rolling_qr_aci_open_max(
            y_obs=y_obs,
            alpha_t=alpha_out_up_t,
            scores_history=scores_out_up,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )


    # =====================================================
    # 2) MAIN LOOP
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        # --- OUTER LOWER ---
        L_out_t, alpha_out_low_t, scores_out_low = rolling_qr_aci_open_max(
            y_obs=y_obs,
            alpha_t=alpha_out_low_t,
            scores_history=scores_out_low,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="lower",
            caviar_model=caviar_model,
            G=G
        )

        # --- OUTER UPPER ---
        U_out_t, alpha_out_up_t, scores_out_up = rolling_qr_aci_open_max(
            y_obs=y_obs,
            alpha_t=alpha_out_up_t,
            scores_history=scores_out_up,
            lookback=lookback_aci,
            eta=eta,
            alpha_target=alpha_target,
            side="upper",
            caviar_model=caviar_model,
            G=G
        )

        # save
        L_out_arr[t]         = L_out_t
        U_out_arr[t]         = U_out_t
        alpha_out_low_arr[t] = alpha_out_low_t
        alpha_out_up_arr[t]  = alpha_out_up_t

    result_df = pd.DataFrame(
        {
            "spread": spread,
            "L": L_out_arr,
            "U": U_out_arr,
            "alpha_out_low": alpha_out_low_arr,
            "alpha_out_up": alpha_out_up_arr,
        }
    ).iloc[warm_up:]

    return result_df






def run_mc_cp_coverage_open_max(
    generator_func,
    n_mc,
    T,
    lookback_aci,
    alpha_outer_target,
    eta,
    caviar_model,
    G,
    warm_up_0=1,
    warm_up=20,
    plot_pointwise=True,
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
        t_df = cp_aci_open_max(
            spread=spread,
            lookback_aci=lookback_aci,
            alpha_target=alpha_outer_target,
            eta=eta,
            caviar_model=caviar_model,
            G=G,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
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






#########################################################################
#########################################################################
#########################################################################










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


eta          = 0.005
eta_iid=0.0
alpha_target_open = alpha_target
alpha_target_classic = 2*alpha_target


warm_up_0 = 2
warm_up   = 10



summary_normal_iid, res_normal_iid, pointwise_res_normal_iid = run_mc_cp_coverage_open(
    generator_func=sim_normal_iid,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,      # parametri del generatore
    sigma=sd
)



summary_normal_ar1, res_normal_ar1,pointwise_res_normal_ar1 = run_mc_cp_coverage_open(
    generator_func=sim_normal_ar1,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sigma=sd,
    phi=phi
)



summary_t_iid, res_t_iid, pointwise_res_t_iid = run_mc_cp_coverage_open(
    generator_func=sim_t_iid_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sd=sd,
    df=df
)

summary_t_ar1, res_t_ar1,pointwise_res_t_ar1= run_mc_cp_coverage_open(
    generator_func=sim_t_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sd=sd,
    df=df,
    phi=phi
)



summary_skewt_iid, res_skewt_iid,pointwise_res_skewt_iid = run_mc_cp_coverage_open(
    generator_func=sim_skewt_iid_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sd=sd,
    df=df,
    lam=lam
)


summary_skewt_ar1, res_skewt_ar1,pointwise_res_skewt_ar1 = run_mc_cp_coverage_open(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sd=sd,
    df=df,
    lam=lam,
    phi=phi
)

summary_skewt_iid_max, res_skewt_iid_max,pointwise_res_skewt_iid_max = run_mc_cp_coverage_open_max(
    generator_func=sim_skewt_iid_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sd=sd,
    df=df,
    lam=lam
)


summary_skewt_ar1_max, res_skewt_ar1_max,pointwise_res_skewt_ar1_max = run_mc_cp_coverage_open_max(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_open,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
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

print_summary("Normal IID (open)", summary_normal_iid)
print_summary("Normal AR(1) (open)", summary_normal_ar1)
print_summary("Student-t IID (open)", summary_t_iid)
print_summary("Student-t AR(1) (open)", summary_t_ar1)
print_summary("Skew-t IID (open)", summary_skewt_iid)
print_summary("Skew-t AR(1) (open)", summary_skewt_ar1)
print_summary("Skew-t IID MAX (open)", summary_skewt_iid_max)
print_summary("Skew-t AR(1) MAX (open)", summary_skewt_ar1_max)


summary_normal_iid_CLASSIC, res_normal_iid_CLASSIC,pointwise_res_normal_iid_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_normal_iid,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,      # parametri del generatore
    sigma=sd
)



summary_normal_ar1_CLASSIC, res_normal_ar1_CLASSIC,pointwise_res_normal_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_normal_ar1,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,    
    mu=mu,
    sigma=sd,
    phi=phi
)



summary_t_iid_CLASSIC, res_t_iid_CLASSIC,pointwise_res_t_iid_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_t_iid_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df
)


summary_t_ar1_CLASSIC, res_t_ar1_CLASSIC,pointwise_res_t_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_t_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    phi=phi
)


summary_skewt_iid_CLASSIC, res_skewt_iid_CLASSIC,pointwise_res_skewt_iid_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_skewt_iid_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta_iid,
    caviar_model=caviar_model,
    G=G,
    warm_up_0=warm_up_0,
    warm_up=warm_up,
    mu=mu,
    sd=sd,
    df=df,
    lam=lam
)


summary_skewt_ar1_CLASSIC, res_skewt_ar1_CLASSIC,pointwise_res_skewt_ar1_CLASSIC = run_mc_cp_coverage(
    generator_func=sim_skewt_ar1_locscale,
    n_mc=n_mc,
    T=T,
    lookback_aci=lookback_aci,
    alpha_outer_target=alpha_target_classic,
    eta=eta,
    caviar_model=caviar_model,
    G=G,
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


print_summary("Normal IID (CLASSIC)", summary_normal_iid_CLASSIC)
print_summary("Normal AR(1) (CLASSIC)", summary_normal_ar1_CLASSIC)
print_summary("Student-t IID (CLASSIC)", summary_t_iid_CLASSIC)
print_summary("Student-t AR(1) (CLASSIC)", summary_t_ar1_CLASSIC)

print_summary("Skew-t IID (CLASSIC)", summary_skewt_iid_CLASSIC)
print_summary("Skew-t AR(1) (CLASSIC)", summary_skewt_ar1_CLASSIC)




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

# MODELLO 1 — AR + parametric PI + ACI scaling (è la baseline CP)

def rolling_ar_parametric_aci(
    y_obs,
    alpha_t,
    scores_history,
    lookback=240,
    eta=0.005,
    alpha_target=0.10,
    p=1
):

    #y = np.asarray(y_obs, float)
    #window = y[:-1][-lookback:]
    
    window = np.asarray(y_obs[:-1], float)
    y_t = y_obs[-1]

    m_t, L_raw, U_raw, sigma_pred = rolling_ar_forecast(
        window, p=p, alpha=alpha_target #QUI alpha=alpha_target non lo uso...
    )

    # ===== conformal quantile (DISCRETE) =====

    #recent_scores = scores_history[-lookback:]

    recent_scores = scores_history ############################ FULL HISTORY

    nc = len(recent_scores)




    ########################################
    
    if nc == 0:
        # usa direttamente GARCH parametric
        L_t = float(L_raw)
        U_t = float(U_raw)
    else:
        idx = int(np.ceil((1.0 - alpha_t) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[idx])

        L_t = float(m_t - qn * sigma_pred)
        U_t = float(m_t + qn * sigma_pred)
    
    ########################################

    # score normalizzato
    score_t = abs(y_t - m_t) / max(sigma_pred, 1e-8)
    scores_history.append(score_t)

    covered = int(L_t <= y_t <= U_t)
    err = 1 - covered
    alpha_t_new = alpha_t + eta * (alpha_target - err)
    #alpha_t_new = np.clip(alpha_t + eta * (alpha_target - err), 1e-4, 0.5)

    return float(m_t), float(L_t), float(U_t), float(alpha_t_new), scores_history



# MODELLO 2 — AR + symmetric residual conformal + ACI

def rolling_ar_symmetric_aci(
    y_obs,
    alpha_t,
    scores_history,
    lookback=240,
    eta=0.005,
    alpha_target=0.10,
    p=1
):
    
    #y = np.asarray(y_obs, float)
    #window = y[:-1][-lookback:]
    
    window = np.asarray(y_obs[:-1], float)
    y_t = y_obs[-1]

    m_t, L_raw, U_raw, _ = rolling_ar_forecast(
        window, p=p, alpha=alpha_target #QUI alpha=alpha_target non lo uso...
    )

    # ===== conformal quantile (DISCRETE) =====
    #recent_scores = scores_history[-lookback:]

    recent_scores = scores_history############################ FULL HISTORY

    nc = len(recent_scores)
    

    ########################################
    
    if nc == 0:
        L_t = float(L_raw)
        U_t = float(U_raw)
    else:
        idx = int(np.ceil((1 - alpha_t) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[idx])

        L_t = m_t - qn
        U_t = m_t + qn
        
    ######################################## 
        
        
    score_t = abs(y_t - m_t)
    scores_history.append(score_t)

    covered = int(L_t <= y_t <= U_t)
    err = 1 - covered
    alpha_t_new = alpha_t + eta * (alpha_target - err)
    #alpha_t_new = np.clip(alpha_t + eta * (alpha_target - err), 1e-4, 0.5)

    return float(m_t), float(L_t), float(U_t), float(alpha_t_new), scores_history




# MODELLO 3 — AR + one-sided ACI (lower/upper indipendenti)

def rolling_ar_one_sided_aci(
    y_obs,
    alpha_low_t,
    alpha_up_t,
    scores_low,
    scores_up,
    lookback=240,
    eta=0.005,
    alpha_low_target=0.10,
    alpha_up_target=0.10,
    p=1
):

    #y = np.asarray(y_obs, float)
    #window = y[:-1][-lookback:]
    
    window = np.asarray(y_obs[:-1], float)
    y_t = y_obs[-1]

    m_t, L_raw, U_raw, _ = rolling_ar_forecast(
        window, p=p, alpha=min(alpha_low_target, alpha_up_target) #QUI alpha=alpha_target non lo uso...
    )


    ########################################
    
    # ===== LOWER SIDE =====
    #recent_low = scores_low[-lookback:]
    recent_low = scores_low   # FULL HISTORY
    nc_low = len(recent_low)

    if nc_low == 0:
        q_low = None
    else:
        recent_low = np.asarray(recent_low, float)
        idx = int(np.ceil((1.0 - alpha_low_t) * (nc_low + 1)) - 1)
        idx = int(np.clip(idx, 0, nc_low - 1))
        q_low = float(np.sort(recent_low)[idx])


    # ===== UPPER SIDE =====
    #recent_up = scores_up[-lookback:]
    recent_up = scores_up   # FULL HISTORY
    nc_up = len(recent_up)

    if nc_up == 0:
        q_up = None
    else:
        recent_up = np.asarray(recent_up, float)
        idx = int(np.ceil((1.0 - alpha_up_t) * (nc_up + 1)) - 1)
        idx = int(np.clip(idx, 0, nc_up - 1))
        q_up = float(np.sort(recent_up)[idx])


    # ===== FINAL BOUNDS =====
    if q_low is None:
        L_t = float(L_raw)
    else:
        L_t = m_t - q_low

    if q_up is None:
        U_t = float(U_raw)
    else:
        U_t = m_t + q_up
        
    ########################################




    score_low = m_t - y_t
    score_up  = y_t - m_t

    #score_low = max(m_t - y_t, 0.0)
    #score_up  = max(y_t - m_t, 0.0)

    scores_low.append(score_low)
    scores_up.append(score_up)

    err_low = int(y_t < L_t)
    err_up  = int(y_t > U_t)

    alpha_low_new = alpha_low_t + eta * (alpha_low_target - err_low)
    alpha_up_new  = alpha_up_t  + eta * (alpha_up_target  - err_up )
    
    #alpha_low_new = np.clip(alpha_low_t + eta * (alpha_low_target - err_low), 1e-4, 0.5)
    #alpha_up_new  = np.clip(alpha_up_t  + eta * (alpha_up_target  - err_up ), 1e-4, 0.5)

    return (
        float(m_t), float(L_t), float(U_t),
        float(alpha_low_new), float(alpha_up_new),
        scores_low, scores_up
    )







def rolling_ar_one_sided_std_aci(
    y_obs,
    alpha_low_t,
    alpha_up_t,
    scores_low,
    scores_up,
    lookback=240,
    eta=0.005,
    alpha_low_target=0.10,
    alpha_up_target=0.10,
    p=1
):
    """
    MODELLO 5 — AR + one-sided CP-ACI con score standardizzati

    Score:
      lower: (m_t - y_t) / sigma_pred
      upper: (y_t - m_t) / sigma_pred
    """

    #y = np.asarray(y_obs, float)
    #window = y[:-1][-lookback:]
    
    window = np.asarray(y_obs[:-1], float)
    y_t = y_obs[-1]

    # === AR forecast ===
    m_t, L_raw, U_raw, sigma_pred = rolling_ar_forecast(
        window, p=p, alpha=min(alpha_low_target, alpha_up_target) #QUI alpha=alpha_target non lo uso...
    )

    sigma = max(sigma_pred, 1e-8)  # safety

    
    ########################################
    # === Quantili conformal (empirici, schema CP corretto) ===
    def conformal_quantile(scores, alpha_t):
        nc = len(scores)
        if nc == 0:
            return None
        #recent = np.asarray(scores[-lookback:], float)
        
        
        recent = np.asarray(scores, float)############################ FULL HISTORY
        
        
        nc = len(recent)
        idx = int(np.ceil((1.0 - alpha_t) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        return float(np.sort(recent)[idx])


    q_low = conformal_quantile(scores_low, alpha_low_t)
    q_up  = conformal_quantile(scores_up,  alpha_up_t)

    if q_low is None:
        L_t = float(L_raw)
    else:
        L_t = m_t - q_low * sigma

    if q_up is None:
        U_t = float(U_raw)
    else:
        U_t = m_t + q_up * sigma
    ########################################


    # === Score standardizzati ===
    score_low = (m_t - y_t) / sigma
    score_up  = (y_t - m_t) / sigma


    #score_low = max((m_t - y_t) / sigma, 0.0)
    #score_up  = max((y_t - m_t) / sigma, 0.0)

    scores_low.append(score_low)
    scores_up.append(score_up)

    # === Copertura ===
    err_low = int(y_t < L_t)
    err_up  = int(y_t > U_t)

    alpha_low_new = alpha_low_t + eta * (alpha_low_target - err_low)
    alpha_up_new = alpha_up_t + eta * (alpha_up_target - err_up)
    #alpha_low_new = np.clip(alpha_low_t + eta * (alpha_low_target - err_low), 1e-4, 0.5)
    #alpha_up_new  = np.clip(alpha_up_t  + eta * (alpha_up_target  - err_up ), 1e-4, 0.5)
    return (
        float(m_t),
        float(L_t),
        float(U_t),
        float(alpha_low_new),
        float(alpha_up_new),
        scores_low,
        scores_up
    )





def cp_aci_ar(
    spread,
    model_type="ar_symmetric_aci",
    lookback=240,

    # --- ACI params ---
    alpha_target=0.10,
    eta=0.005,

    # --- AR params ---
    p=1,

    # --- NEW: warm-up control ---
    warm_up_0=1,
    warm_up=20
):


    spread = np.asarray(spread, float)
    T = len(spread)


    # -------------------------------------------------
    # Allocations
    # -------------------------------------------------
    L_arr = np.full(T, np.nan)
    U_arr = np.full(T, np.nan)
    alpha_low_arr = np.full(T, np.nan)
    alpha_up_arr  = np.full(T, np.nan)

    # -------------------------------------------------
    # ACI states
    # -------------------------------------------------
    alpha_t = float(alpha_target)
    scores = []

    alpha_low_t = float(alpha_target)
    alpha_up_t  = float(alpha_target)
    scores_low, scores_up = [], []

    # =====================================================
    # 1) PRE-WARM / SHADOW TRAINING
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = spread[:t + 1]

        if model_type == "ar_naive":
            m_t, L_t, U_t = rolling_ar_naive(
                y_obs, lookback=lookback, p=p, alpha=alpha_target
            )

        elif model_type == "ar_parametric_aci":
            m_t, L_t, U_t, alpha_t, scores = rolling_ar_parametric_aci(
                y_obs,
                alpha_t=alpha_t,
                scores_history=scores,
                lookback=lookback,
                eta=eta,
                alpha_target=alpha_target,
                p=p
            )

        elif model_type == "ar_symmetric_aci":
            m_t, L_t, U_t, alpha_t, scores = rolling_ar_symmetric_aci(
                y_obs,
                alpha_t=alpha_t,
                scores_history=scores,
                lookback=lookback,
                eta=eta,
                alpha_target=alpha_target,
                p=p
            )

        elif model_type == "ar_one_sided_aci":
            m_t, L_t, U_t, alpha_low_t, alpha_up_t, scores_low, scores_up = (
                rolling_ar_one_sided_aci(
                    y_obs,
                    alpha_low_t=alpha_low_t,
                    alpha_up_t=alpha_up_t,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    lookback=lookback,
                    eta=eta,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    p=p
                )
            )

        elif model_type == "ar_one_sided_std_aci":
            m_t, L_t, U_t, alpha_low_t, alpha_up_t, scores_low, scores_up = (
                rolling_ar_one_sided_std_aci(
                    y_obs,
                    alpha_low_t=alpha_low_t,
                    alpha_up_t=alpha_up_t,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    lookback=lookback,
                    eta=eta,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    p=p
                )
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")



    # =====================================================
    # 2) MAIN LOOP (STORE RESULTS)
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = spread[:t + 1]

        if model_type == "ar_naive":
            m_t, L_t, U_t = rolling_ar_naive(
                y_obs, lookback=lookback, p=p, alpha=alpha_target
            )

        elif model_type == "ar_parametric_aci":
            m_t, L_t, U_t, alpha_t, scores = rolling_ar_parametric_aci(
                y_obs,
                alpha_t=alpha_t,
                scores_history=scores,
                lookback=lookback,
                eta=eta,
                alpha_target=alpha_target,
                p=p
            )

        elif model_type == "ar_symmetric_aci":
            m_t, L_t, U_t, alpha_t, scores = rolling_ar_symmetric_aci(
                y_obs,
                alpha_t=alpha_t,
                scores_history=scores,
                lookback=lookback,
                eta=eta,
                alpha_target=alpha_target,
                p=p
            )

        elif model_type == "ar_one_sided_aci":
            m_t, L_t, U_t, alpha_low_t, alpha_up_t, scores_low, scores_up = (
                rolling_ar_one_sided_aci(
                    y_obs,
                    alpha_low_t=alpha_low_t,
                    alpha_up_t=alpha_up_t,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    lookback=lookback,
                    eta=eta,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    p=p
                )
            )

        elif model_type == "ar_one_sided_std_aci":
            m_t, L_t, U_t, alpha_low_t, alpha_up_t, scores_low, scores_up = (
                rolling_ar_one_sided_std_aci(
                    y_obs,
                    alpha_low_t=alpha_low_t,
                    alpha_up_t=alpha_up_t,
                    scores_low=scores_low,
                    scores_up=scores_up,
                    lookback=lookback,
                    eta=eta,
                    alpha_low_target=alpha_target,
                    alpha_up_target=alpha_target,
                    p=p
                )
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        L_arr[t] = L_t
        U_arr[t] = U_t
        alpha_low_arr[t] = alpha_low_t
        alpha_up_arr[t]  = alpha_up_t

    return pd.DataFrame(
        {
            "spread": spread,
            "L": L_arr,
            "U": U_arr,
            "alpha_out_low": alpha_low_arr,
            "alpha_out_up": alpha_up_arr,
        }
    ).iloc[warm_up:]







def run_mc_cp_coverage_ar(
    generator_func,
    n_mc,
    T,
    lookback,
    alpha_outer_target,
    eta,
    p,
    warm_up_0,
    warm_up,
    model_type,
    plot_pointwise=True,
    **generator_kwargs
):
    """
    Monte Carlo evaluation of CP–ACI coverage properties for AR-based bands.

    In addition to classic time-averaged coverage, computes pointwise coverage
    over time averaged across MC replications, and (optionally) plots it.

    Returns
    -------
    summary : dict
        Mean/std of classic coverage and width metrics.
    mc_results : list of dict
        Per-replication metrics + raw series (spread, L, U) for pointwise analysis.
    pointwise_results : dict
        Pointwise coverage arrays:
          - cov_outer_t (two-sided)
          - cov_outer_lower_t (one-sided: spread >= L)
          - cov_outer_upper_t (one-sided: spread <= U)
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

        # --- compute AR-based bands
        bands_df = cp_aci_ar(
            spread=spread,
            model_type=model_type,
            lookback=lookback,
            alpha_target=alpha_outer_target,
            eta=eta,
            p=p,
            warm_up_0=warm_up_0,
            warm_up=warm_up
        )

        # --- classic (time-averaged) coverage + widths
        cov_info = compute_cp_coverage(bands_df)

        # --- store both classic metrics and raw series for pointwise
        mc_results.append({
            "cov_outer_full":          cov_info["cov_outer_full"],
            "cov_outer_full_lower":    cov_info["cov_outer_full_lower"],
            "cov_outer_full_upper":    cov_info["cov_outer_full_upper"],
            "outer_width_full_mean":   cov_info["outer_width_full_mean"],
            "outer_width_full_median": cov_info["outer_width_full_median"],

            # raw series needed for pointwise coverage
            "spread": bands_df["spread"].values,
            "L":      bands_df["L"].values,
            "U":      bands_df["U"].values,
        })

    # =========================================================
    # 2) CLASSIC Monte Carlo summaries
    # =========================================================
    cov_outer_arr    = np.array([r["cov_outer_full"]          for r in mc_results], float)
    cov_lower_arr    = np.array([r["cov_outer_full_lower"]    for r in mc_results], float)
    cov_upper_arr    = np.array([r["cov_outer_full_upper"]    for r in mc_results], float)
    width_mean_arr   = np.array([r["outer_width_full_mean"]   for r in mc_results], float)
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
    # 3) POINTWISE coverage (NEW)
    # =========================================================
    cov_outer_t = compute_pointwise_cp_coverage(mc_results)
    cov_outer_lower_t, cov_outer_upper_t = compute_pointwise_one_sided_coverage(mc_results)

    pointwise_results = {
        "cov_outer_t":       cov_outer_t,
        "cov_outer_lower_t": cov_outer_lower_t,
        "cov_outer_upper_t": cov_outer_upper_t,
    }
    ONE_SIDED_MODELS = {
        "ar_one_sided_aci",
        "ar_one_sided_std_aci",
    }

    # =========================================================
    # 4) AUTOMATIC PLOTTING (correct alpha handling)
    # =========================================================
    if plot_pointwise:

        # -------------------------------
        # TWO-SIDED (closed intervals)
        # -------------------------------
        if model_type not in ONE_SIDED_MODELS:

            plot_pointwise_coverage(
                cov_outer_t,
                alpha_target=alpha_outer_target,
                title=f"{model_type.upper()} — Pointwise two-sided coverage"
            )

            plot_pointwise_coverage(
                cov_outer_lower_t,
                alpha_target=alpha_outer_target / 2,
                title=f"{model_type.upper()} — Pointwise LOWER-side coverage"
            )

            plot_pointwise_coverage(
                cov_outer_upper_t,
                alpha_target=alpha_outer_target / 2,
                title=f"{model_type.upper()} — Pointwise UPPER-side coverage"
            )

        # -------------------------------
        # ONE-SIDED (open intervals)
        # -------------------------------
        else:

            plot_pointwise_coverage(
                cov_outer_t,
                alpha_target=2 * alpha_outer_target,
                title=f"{model_type.upper()} — OPEN pointwise two-sided coverage"
            )

            plot_pointwise_coverage(
                cov_outer_lower_t,
                alpha_target=alpha_outer_target,
                title=f"{model_type.upper()} — OPEN pointwise LOWER-side coverage"
            )

            plot_pointwise_coverage(
                cov_outer_upper_t,
                alpha_target=alpha_outer_target,
                title=f"{model_type.upper()} — OPEN pointwise UPPER-side coverage"
            )

    return summary, mc_results, pointwise_results









model_types = [
    "ar_naive",
    "ar_parametric_aci",
    "ar_symmetric_aci",
    "ar_one_sided_aci",
    "ar_one_sided_std_aci",
]


generators = {
    "Normal IID": (
        sim_normal_iid,
        dict(mu=mu, sigma=sd),
        "iid"
    ),
    "Normal AR(1)": (
        sim_normal_ar1,
        dict(mu=mu, sigma=sd, phi=phi),
        "ar"
    ),
    "Student-t IID": (
        sim_t_iid_locscale,
        dict(mu=mu, sd=sd, df=df),
        "iid"
    ),
    "Student-t AR(1)": (
        sim_t_ar1_locscale,
        dict(mu=mu, sd=sd, df=df, phi=phi),
        "ar"
    ),
    "Skew-t IID": (
        sim_skewt_iid_locscale,
        dict(mu=mu, sd=sd, df=df, lam=lam),
        "iid"
    ),
    "Skew-t AR(1)": (
        sim_skewt_ar1_locscale,
        dict(mu=mu, sd=sd, df=df, lam=lam, phi=phi),
        "ar"
    ),
}



two_sided_models = {
    "ar_naive",
    "ar_parametric_aci",
    "ar_symmetric_aci",
}


#####
TARGET_GENERATORS = {
    "Skew-t IID",
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

        # -------------------------
        # ETA SELEZIONATO IN BASE AL PROCESSO
        # -------------------------
        eta_used = eta_iid if gen_type == "iid" else eta

        summary, _, pointwise = run_mc_cp_coverage_ar(
            generator_func=gen_func,
            n_mc=n_mc,
            T=T,
            lookback=lookback,
            alpha_outer_target=alpha_used,
            eta=eta_used,
            p=p,
            warm_up_0=warm_up_0,
            warm_up=warm_up,
            model_type=model_type,
            plot_pointwise=False,   
            **gen_kwargs
        )

        ############
        if gen_name in TARGET_GENERATORS:
            pointwise_store.setdefault(model_type, {})
            pointwise_store[model_type][gen_name] = pointwise
        ##############

        mc_summaries[model_type][gen_name] = summary

        print(
            f"\n--- {gen_name} ({gen_type.upper()}, "
            f"eta={eta_used}, alpha={alpha_used}) ---"
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





