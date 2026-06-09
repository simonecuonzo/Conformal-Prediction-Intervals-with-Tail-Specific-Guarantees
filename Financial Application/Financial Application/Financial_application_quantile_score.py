# =========================================================
# 1️⃣ LIBRERIE E PARAMETRI BASE
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize, minimize_scalar

import os
from scipy.stats import chi2
from arch import arch_model

from scipy.stats import t as student_t
from scipy.stats import norm





plt.rcParams['figure.figsize'] = (11,4)
plt.rcParams['axes.grid'] = True



def evaluate_interval(df, low_col, alpha_target):
    y = df["r"].values
    L = df[low_col].values
    #U = df[high_col].values
    covered = (y >= L) #& (y <= U)
    return dict(
        coverage=np.nanmean(covered),
        misscoverage=1-np.nanmean(covered),
        #mean_width=np.nanmean(U-L),


        exceedances=np.sum(~covered),
        target_alpha=alpha_target
    )





def lr_uc(y, q, alpha):
    y = np.asarray(y, float)
    q = np.asarray(q, float)

    hits = (y < q).astype(int)
    n = hits.size
    n1 = hits.sum()
    n0 = n - n1

    if n1 == 0 or n1 == n:
        return dict(stat=np.nan, pvalue=np.nan)

    p_hat = n1 / n

    ll_h0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
    ll_h1 = n0 * np.log(1 - p_hat) + n1 * np.log(p_hat)

    LRuc = -2.0 * (ll_h0 - ll_h1)
    pval = 1.0 - chi2.cdf(LRuc, 1)

    return dict(
        stat=float(LRuc),
        pvalue=float(pval),
        hits=int(n1),
        T=int(n),
        hit_rate=float(p_hat),
    )


def lr_ind(y, q):
    y = np.asarray(y, float)
    q = np.asarray(q, float)

    hits = (y < q).astype(int)

    tr = hits[1:] - hits[:-1]

    n01 = np.sum(tr == 1)
    n10 = np.sum(tr == -1)
    n11 = np.sum((hits[1:][tr == 0]) == 1)
    n00 = np.sum((hits[1:][tr == 0]) == 0)

    n0 = n00 + n01
    n1 = n10 + n11
    n = n0 + n1

    if n1 == 0:
        return dict(stat=np.nan, pvalue=np.nan)

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p   = n1 / n

    #ll_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_h0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    ll_h1 = (
        n00 * np.log(1 - p01) +
        n01 * np.log(p01) +
        n10 * np.log(1 - p11)
    )
    if p11 > 0:
        ll_h1 += n11 * np.log(p11)

    LRind = -2.0 * (ll_h0 - ll_h1)
    pval  = 1.0 - chi2.cdf(LRind, 1)

    return dict(
        stat=float(LRind),
        pvalue=float(pval),
        n00=int(n00), n01=int(n01), n10=int(n10), n11=int(n11)
    )


def lr_cc(y, q, alpha):
    uc  = lr_uc(y, q, alpha)
    ind = lr_ind(y, q)

    if np.isnan(uc["stat"]) or np.isnan(ind["stat"]):
        return dict(stat=np.nan, pvalue=np.nan)

    LRcc = uc["stat"] + ind["stat"]
    pval = 1.0 - chi2.cdf(LRcc, 2)

    return dict(
        stat=float(LRcc),
        pvalue=float(pval),
        uc=uc,
        ind=ind
    )



def dq_test(y, q, alpha, hit_lags=4, forecast_lags=1):
    y = np.asarray(y, float)
    q = np.asarray(q, float)

    hits = (y < q).astype(int)
    n = hits.size

    p = hit_lags
    ql = forecast_lags
    pq = max(p, ql - 1)

    try:
        Y = hits[pq:] - alpha
        X = np.zeros((n - pq, 1 + p + ql))
        X[:, 0] = 1.0  # constant

        # lagged hits
        for i in range(p):
            X[:, 1 + i] = hits[pq - (i + 1): -(i + 1)]

        # VaR and lagged VaR
        for j in range(ql):
            if j == 0:
                X[:, 1 + p + j] = q[pq:]
            else:
                X[:, 1 + p + j] = q[pq - j: -j]

        beta = np.linalg.solve(X.T @ X, X.T @ Y)
        DQ = beta.T @ (X.T @ X) @ beta / (alpha * (1 - alpha))
        #H = n - pq
        #DQ = beta.T @ (X.T @ X) @ beta / (H * alpha * (1 - alpha))
        pval = 1.0 - chi2.cdf(DQ, 1 + p + ql)

    except Exception:
        DQ, pval = np.nan, np.nan

    return dict(
        stat=float(DQ),
        pvalue=float(pval),
        dof=int(1 + p + ql)
    )

def backtest_var(y, q, alpha=0.05, lags=1):
    """
    y: array-like returns (OOS)
    q: array-like VaR forecasts (stesso allineamento di y)
    """
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    mask = ~(np.isnan(y) | np.isnan(q))
    y, q = y[mask], q[mask]

    # Indicatori base
    I = (y <= q).astype(int)
    AE = I.mean() / alpha

    # Test
    uc  = lr_uc(y, q, alpha)
    ind = lr_ind(y, q)
    cc  = lr_cc(y, q, alpha)


    hit_lags=lags
    forecast_lags=0


    dq  = dq_test(
        y, q, alpha,
        hit_lags=hit_lags,
        forecast_lags=forecast_lags
    )

    summary = {
        "AE": AE,
        "LRuc.stat": uc["stat"],   "LRuc.pvalue": uc["pvalue"],
        "LRind.stat": ind["stat"], "LRind.pvalue": ind["pvalue"],
        "LRcc.stat": cc["stat"],   "LRcc.pvalue": cc["pvalue"],
        "DQ.stat": dq["stat"],     "DQ.pvalue": dq["pvalue"],
        "hits": int(uc["hits"]),
        "T": int(uc["T"]),
    }

    return summary, dict(uc=uc, ind=ind, cc=cc, dq=dq), I


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



# =========================================================
# 3️⃣ DATI REALI
# =========================================================

tickers = ["SPY"]
rets = pd.read_csv("returns_2017-2025_SPY_GLD.csv",
                       parse_dates=[0],        # prima colonna = date
                       index_col=0)            # usala come indice
print(rets.head(), rets.info())

########################################################

"""


# ==========================================
# EQUITY-ONLY DATASET
# ==========================================

tickers = [
    "XLE",
    "TQQQ"
]

rets = pd.read_csv(
    "EQUITY_only_log_returns_2018_2021.csv",
    parse_dates=[0],
    index_col=0
)

print(rets.head(), rets.info())

"""

# =========================================================
# 4️⃣ STIMA MODELLI
# =========================================================
warm_up_0 = 240 # 250
warm_up = 250 # 250
        
alpha_target = 0.1


lookback = 250


lags_dq = 1




I = 500


#gammas = np.array([0.001 * 2**k for k in range(8)], dtype=float)
gammas = np.array([0.005, 0.008, 0.01, 0.015, 0.02], dtype=float)




k = len(gammas)

sigma = sigma_faci(I)
eta = eta_faci(
    alpha_target=alpha_target,
    I=I,
    k=k
)




#####################



def rolling_garch_t_forecast(y_window, alpha=0.10, two_sided=True):
    """
    Fit GARCH(1,1) with symmetric Student-t errors.

    Returns:
        mu_t         : mean forecast
        L_hat, U_hat : parametric prediction interval
        sigma_pred   : forecasted conditional std
        nu           : estimated degrees of freedom
    """

    y = np.asarray(y_window, float)


    # ===== SCALE RETURNS =====
    scale_factor = 100.0
    y_scaled = y * scale_factor

    am = arch_model(
        y_scaled,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t"
    )

    res = am.fit(disp="off")

    fcast = res.forecast(horizon=1)
    mu_scaled = float(fcast.mean.iloc[-1, 0])
    sigma2_scaled = float(fcast.variance.iloc[-1, 0])

    mu_t = mu_scaled / scale_factor
    sigma_pred = np.sqrt(max(sigma2_scaled, 1e-12)) / scale_factor

    nu = float(res.params["nu"])

    scale_corr = np.sqrt((nu - 2.0) / nu)

    #q_low = student_t.ppf(alpha/2, df=nu)
    #q_high = student_t.ppf(1 - alpha/2, df=nu)
    if two_sided:
        q_low  = student_t.ppf(alpha/2, df=nu)
        q_high = student_t.ppf(1 - alpha/2, df=nu)
    else:
        q_low  = student_t.ppf(alpha, df=nu)
        q_high = student_t.ppf(1 - alpha, df=nu)
    L_hat = mu_t + q_low * scale_corr * sigma_pred
    U_hat = mu_t + q_high * scale_corr * sigma_pred

    return mu_t, float(L_hat), float(U_hat), float(sigma_pred)#, nu



'''


def rolling_garch_t_forecast(y_window, alpha=0.10, two_sided=True):

    y = np.asarray(y_window, float)


    # ===== SCALE RETURNS =====
    scale_factor = 100.0
    y_scaled = y * scale_factor

    am = arch_model(
        y_scaled,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"   # 👈 GAUSSIAN
    )

    res = am.fit(disp="off")

    fcast = res.forecast(horizon=1)

    mu_scaled = float(fcast.mean.iloc[-1, 0])
    sigma2_scaled = float(fcast.variance.iloc[-1, 0])

    mu_t = mu_scaled / scale_factor
    sigma_t = np.sqrt(max(sigma2_scaled, 1e-12)) / scale_factor

    # ===== GAUSSIAN QUANTILES =====
    #z_low = norm.ppf(alpha/2)
    #z_high = norm.ppf(1 - alpha/2)
    
    if two_sided:
        z_low = norm.ppf(alpha/2)
        z_high = norm.ppf(1 - alpha/2)
    else:
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)
        
    L_hat = mu_t + z_low * sigma_t
    U_hat = mu_t + z_high * sigma_t

    return mu_t, float(L_hat), float(U_hat), float(sigma_t)
'''


            
def rolling_garch_faci_open(
    r_obs,
    scores_history,
    faci_state,
    alpha_target=0.10,
    eta=2.72,
    sigma=1e-3,
    lookback=240,
):
    """
    Rolling GARCH(1,1)-t quantile forecasting
    + two independent one-sided conformal FACI updates
      (lower and upper updated simultaneously).

    Returns
    -------
    L_t : float
        Lower conformal bound
    U_t : float
        Upper conformal bound
    faci_state : dict
    scores_history : dict
    q_raw : dict
        Raw GARCH quantiles
    """

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    k = faci_state["k"]

    # ----- LOWER experts
    lower_alphas = faci_state["lower_alphas"]
    lower_ws = faci_state["lower_ws"]

    # ----- UPPER experts
    upper_alphas = faci_state["upper_alphas"]
    upper_ws = faci_state["upper_ws"]

    # ======================================================
    # Data
    # ======================================================
    r_obs = np.asarray(r_obs, float)
    r_t = float(r_obs[-1])

    r_window = r_obs[:-1][-lookback:]

    # ======================================================
    # GARCH quantile forecasts
    # ======================================================
    mu_t, L_hat, U_hat, sigma_pred = rolling_garch_t_forecast(
        r_window,
        alpha=alpha_target,
        two_sided=False
    )

    # ======================================================
    # LOWER SIDE
    # ======================================================
    lower_probs = lower_ws / (np.sum(lower_ws) + 1e-300)
    alpha_lower = float(np.sum(lower_probs * lower_alphas))

    lower_scores = scores_history["lower"][-lookback:]
    nc_lower = len(lower_scores)

    if nc_lower == 0:
        qn_lower = 0.0
    else:
        idx = int(np.ceil((1.0 - alpha_lower) * (nc_lower + 1)) - 1)
        idx = int(np.clip(idx, 0, nc_lower - 1))
        qn_lower = float(np.sort(lower_scores)[idx])

    L_t = L_hat - qn_lower

    lower_score_t = L_hat - r_t
    scores_history["lower"].append(float(lower_score_t))

    # ======================================================
    # UPPER SIDE
    # ======================================================
    upper_probs = upper_ws / (np.sum(upper_ws) + 1e-300)
    alpha_upper = float(np.sum(upper_probs * upper_alphas))

    upper_scores = scores_history["upper"][-lookback:]
    nc_upper = len(upper_scores)

    if nc_upper == 0:
        qn_upper = 0.0
    else:
        idx = int(np.ceil((1.0 - alpha_upper) * (nc_upper + 1)) - 1)
        idx = int(np.clip(idx, 0, nc_upper - 1))
        qn_upper = float(np.sort(upper_scores)[idx])

    U_t = U_hat + qn_upper

    upper_score_t = r_t - U_hat
    scores_history["upper"].append(float(upper_score_t))

    # ======================================================
    # LOWER FACI UPDATE
    # ======================================================
    if nc_lower > 0:

        past_lower = np.asarray(lower_scores, dtype=float)

        beta_lower = float(np.mean(past_lower >= lower_score_t))

        lower_losses = pinball_loss(
            beta_lower,
            lower_alphas,
            alpha_target
        )

        err_lower = (lower_alphas > beta_lower).astype(float)

        lower_alphas += gammas * (alpha_target - err_lower)

        log_w = np.log(lower_ws + 1e-300) - eta * lower_losses
        log_w -= np.max(log_w)

        w_bar = np.exp(log_w)

        lower_ws = (
            (1.0 - sigma) * w_bar / (np.sum(w_bar) + 1e-300)
            + sigma / k
        )

    # ======================================================
    # UPPER FACI UPDATE
    # ======================================================
    if nc_upper > 0:

        past_upper = np.asarray(upper_scores, dtype=float)

        beta_upper = float(np.mean(past_upper >= upper_score_t))

        upper_losses = pinball_loss(
            beta_upper,
            upper_alphas,
            alpha_target
        )

        err_upper = (upper_alphas > beta_upper).astype(float)

        upper_alphas += gammas * (alpha_target - err_upper)

        log_w = np.log(upper_ws + 1e-300) - eta * upper_losses
        log_w -= np.max(log_w)

        w_bar = np.exp(log_w)

        upper_ws = (
            (1.0 - sigma) * w_bar / (np.sum(w_bar) + 1e-300)
            + sigma / k
        )

    # ======================================================
    # Save updated state
    # ======================================================
    faci_state["lower_alphas"] = lower_alphas
    faci_state["lower_ws"] = lower_ws

    faci_state["upper_alphas"] = upper_alphas
    faci_state["upper_ws"] = upper_ws

    # ======================================================
    # Raw quantiles
    # ======================================================
    q_raw = {
        "lower": float(L_hat),
        "upper": float(U_hat),
    }

    return (
        float(L_t),
        float(U_t),
        faci_state,
        scores_history,
        q_raw,
    )


def rolling_cqr_faci_joint_garch(
    r_obs,
    scores_history,
    faci_state,
    lookback=240,
    eta=2.72,
    sigma=1/1000,
    alpha_target=0.10,
):
    """
    Two–sided CQR + FACI with GARCH(1,1)-t DIRECTLY on RETURNS.

    Joint score:
        score_t = max(q_low_t - r_t, r_t - q_high_t)
    """

    r_obs = np.asarray(r_obs, dtype=float)
    T = len(r_obs)


    r_t = float(r_obs[-1])

    # ======================================================
    # Unpack FACI state
    # ======================================================
    gammas = faci_state["gammas"]
    expert_alphas = faci_state["expert_alphas"]
    expert_ws = faci_state["expert_ws"]
    k = faci_state["k"]

    # ======================================================
    # Past returns only
    # ======================================================
    r_window = r_obs[:-1][-lookback:]


    # ======================================================
    # GARCH two-sided quantiles (alpha/2, 1-alpha/2)
    # ======================================================

    mu_t, q_low_t, q_high_t, sigma_pred = rolling_garch_t_forecast(
        r_window,
        alpha=alpha_target,
        two_sided=True
    )

    q_low_t = float(q_low_t)
    q_high_t = float(q_high_t)

    # ======================================================
    # Expert aggregation
    # ======================================================
    expert_probs = expert_ws / (np.sum(expert_ws) + 1e-300)
    alpha_bar = float(np.sum(expert_probs * expert_alphas))

    # ======================================================
    # Conformal calibration (RECENT scores only)
    # ======================================================
    recent_scores = scores_history[-lookback:]
    nc = len(recent_scores)

    if nc == 0:
        qn = 0.0
    else:
        idx = int(np.ceil((1.0 - alpha_bar) * (nc + 1)) - 1)
        idx = int(np.clip(idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[idx])

    # ======================================================
    # Final interval
    # ======================================================
    L_t = q_low_t - qn
    U_t = q_high_t + qn

    # ======================================================
    # Joint CQR score (based on *raw* parametric quantiles)
    # ======================================================
    score_t = max(q_low_t - r_t, r_t - q_high_t)
    scores_history.append(float(score_t))

    # ======================================================
    # FACI update
    # ======================================================
    if nc > 0:
        past_scores = np.asarray(recent_scores, dtype=float)
        beta_t = float(np.mean(past_scores >= score_t))

        losses = pinball_loss(beta_t, expert_alphas, alpha_target)

        log_w = np.log(expert_ws + 1e-300) - eta * losses
        log_w -= np.max(log_w)
        w_bar = np.exp(log_w)

        expert_ws = (
            (1.0 - sigma) * w_bar / (np.sum(w_bar) + 1e-300)
            + sigma / k
        )

        err_i = (expert_alphas > beta_t).astype(float)
        expert_alphas += gammas * (alpha_target - err_i)
        #expert_alphas = np.clip(expert_alphas, 0.0, 1.0)

    # ======================================================
    # Save state
    # ======================================================
    faci_state["expert_alphas"] = expert_alphas
    faci_state["expert_ws"] = expert_ws


    q_raw = {
        "lower": float(q_low_t),
        "upper": float(q_high_t),
    }

    return (
        float(L_t),
        float(U_t),
        faci_state,
        scores_history,
        q_raw)




# =========================================================
# RESULTS CONTAINER
# =========================================================

results_all = {
    "q (One-sided)": {},
    "q (Two-sided)": {},
}

# =========================================================
# =========================================================
# ONE-SIDED FACI
# =========================================================
# =========================================================

def run_model_step_faci_one_sided(
    y_obs,
    scores_history,
    faci_state,
    lookback,
    alpha_target,
    eta=2.72,
    sigma_mix=1e-3,
):
    """
    Independent lower/upper one-sided FACI updates.

    Returns
    -------
    L_t
    U_t
    faci_state
    scores_history
    q_raw
    """

    L_t, U_t, faci_state, scores_history, q_raw = rolling_garch_faci_open(
        r_obs=y_obs,
        scores_history=scores_history,
        faci_state=faci_state,
        alpha_target=alpha_target,
        eta=eta,
        sigma=sigma_mix,
        lookback=lookback,
    )

    return (
        L_t,
        U_t,
        faci_state,
        scores_history,
        q_raw,
    )


# =========================================================
# RUN ONE-SIDED
# =========================================================

results_model_one = {}

for ticker in tickers:

    y = rets[ticker].to_numpy()
    T = len(y)

    # -----------------------------------------------------
    # FACI hyperparameters
    # -----------------------------------------------------
    k = len(gammas)

    sigma = sigma_faci(I)

    eta = eta_faci(
        alpha_target=alpha_target,
        I=I,
        k=k
    )

    # -----------------------------------------------------
    # FACI STATE (independent lower/upper)
    # -----------------------------------------------------

    faci_state = {
        "gammas": gammas,
        "k": k,

        # LOWER
        "lower_alphas": np.full(k, alpha_target),
        "lower_ws": np.ones(k),

        # UPPER
        "upper_alphas": np.full(k, alpha_target),
        "upper_ws": np.ones(k),
    }
    
    # -----------------------------------------------------
    # SCORE HISTORIES
    # -----------------------------------------------------
    scores_hist = {
        "lower": [],
        "upper": [],
    }

    # -----------------------------------------------------
    # STORAGE
    # -----------------------------------------------------
    L_list = []
    U_list = []

    qL_raw_list = []
    qU_raw_list = []

    alpha_lower_list = []
    alpha_upper_list = []

    # =====================================================
    # 1️⃣ PRE-WARM
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = y[:t+1]

        (
            L_t,
            U_t,
            faci_state,
            scores_hist,
            q_raw,
        ) = run_model_step_faci_one_sided(
            y_obs=y_obs,
            scores_history=scores_hist,
            faci_state=faci_state,
            lookback=lookback,
            alpha_target=alpha_target,
            eta=eta,
            sigma_mix=sigma,
        )

    # =====================================================
    # 2️⃣ MAIN LOOP
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = y[:t+1]

        (
            L_t,
            U_t,
            faci_state,
            scores_hist,
            q_raw,
        ) = run_model_step_faci_one_sided(
            y_obs=y_obs,
            scores_history=scores_hist,
            faci_state=faci_state,
            lookback=lookback,
            alpha_target=alpha_target,
            eta=eta,
            sigma_mix=sigma,
        )

        # -------------------------------------------------
        # Save bounds
        # -------------------------------------------------
        L_list.append(L_t)
        U_list.append(U_t)

        # -------------------------------------------------
        # Raw quantiles
        # -------------------------------------------------
        qL_raw_list.append(q_raw["lower"])
        qU_raw_list.append(q_raw["upper"])

        # -------------------------------------------------
        # Diagnostic alphas
        # -------------------------------------------------
        pL = (
            faci_state["lower_ws"]
            / (np.sum(faci_state["lower_ws"]) + 1e-300)
        )

        pU = (
            faci_state["upper_ws"]
            / (np.sum(faci_state["upper_ws"]) + 1e-300)
        )

        alpha_lower_list.append(
            float(np.sum(pL * faci_state["lower_alphas"]))
        )

        alpha_upper_list.append(
            float(np.sum(pU * faci_state["upper_alphas"]))
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    dates = rets.index[warm_up + 1 : T]

    results_model_one[ticker] = pd.DataFrame({

        # conformal bounds
        "L": L_list,
        "U": U_list,

        # raw quantiles
        "q_lower_raw": qL_raw_list,
        "q_upper_raw": qU_raw_list,

        # adaptive alphas
        "alpha_lower": alpha_lower_list,
        "alpha_upper": alpha_upper_list,

        # returns
        "r": y[warm_up + 1:],

    }, index=dates)

results_all["q (One-sided)"] = results_model_one


# =========================================================
# =========================================================
# TWO-SIDED FACI (JOINT)
# =========================================================
# =========================================================

def run_model_step_faci_two_sided(
    y_obs,
    scores_history,
    faci_state,
    lookback,
    alpha_target,
    eta=2.72,
    sigma_mix=1e-3,
):
    """
    Joint two-sided FACI/CQR update.
    """

    (
        L_t,
        U_t,
        faci_state,
        scores_history,
        q_raw,
    ) = rolling_cqr_faci_joint_garch(
        r_obs=y_obs,
        scores_history=scores_history,
        faci_state=faci_state,
        lookback=lookback,
        eta=eta,
        sigma=sigma_mix,
        alpha_target=2 * alpha_target
    )

    return (
        L_t,
        U_t,
        faci_state,
        scores_history,
        q_raw,
    )


# =========================================================
# RUN TWO-SIDED
# =========================================================

results_model_two = {}

for ticker in tickers:

    y = rets[ticker].to_numpy()
    T = len(y)

    # -----------------------------------------------------
    # FACI hyperparameters
    # -----------------------------------------------------
    alpha_model = 2.0 * alpha_target

    k = len(gammas)

    sigma = sigma_faci(I)

    eta = eta_faci(
        alpha_target=alpha_model,
        I=I,
        k=k
    )

    # -----------------------------------------------------
    # FACI STATE (joint)
    # -----------------------------------------------------
    faci_state = init_faci_state(
        gammas=gammas,
        alpha_init=alpha_model
    )

    scores_hist = []

    # -----------------------------------------------------
    # STORAGE
    # -----------------------------------------------------
    L_list = []
    U_list = []

    qL_raw_list = []
    qU_raw_list = []

    alpha_joint_list = []

    # =====================================================
    # 1️⃣ PRE-WARM
    # =====================================================
    for t in range(warm_up_0, warm_up):

        y_obs = y[:t+1]

        (
            L_t,
            U_t,
            faci_state,
            scores_hist,
            q_raw,
        ) = run_model_step_faci_two_sided(
            y_obs=y_obs,
            scores_history=scores_hist,
            faci_state=faci_state,
            lookback=lookback,
            alpha_target=alpha_target,
            eta=eta,
            sigma_mix=sigma,
        )

    # =====================================================
    # 2️⃣ MAIN LOOP
    # =====================================================
    for t in range(warm_up, T - 1):

        y_obs = y[:t+1]

        (
            L_t,
            U_t,
            faci_state,
            scores_hist,
            q_raw,
        ) = run_model_step_faci_two_sided(
            y_obs=y_obs,
            scores_history=scores_hist,
            faci_state=faci_state,
            lookback=lookback,
            alpha_target=alpha_target,
            eta=eta,
            sigma_mix=sigma,
        )

        # -------------------------------------------------
        # Save bounds
        # -------------------------------------------------
        L_list.append(L_t)
        U_list.append(U_t)

        # -------------------------------------------------
        # Raw quantiles
        # -------------------------------------------------
        qL_raw_list.append(q_raw["lower"])
        qU_raw_list.append(q_raw["upper"])

        # -------------------------------------------------
        # Joint alpha
        # -------------------------------------------------
        ws = faci_state["expert_ws"]
        aa = faci_state["expert_alphas"]

        p = ws / (np.sum(ws) + 1e-300)

        alpha_joint_list.append(
            float(np.sum(p * aa))
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    dates = rets.index[warm_up + 1 : T]

    results_model_two[ticker] = pd.DataFrame({

        # conformal bounds
        "L": L_list,
        "U": U_list,

        # raw quantiles
        "q_lower_raw": qL_raw_list,
        "q_upper_raw": qU_raw_list,

        # joint alpha
        "alpha_joint": alpha_joint_list,

        # returns
        "r": y[warm_up + 1:],

    }, index=dates)

results_all["q (Two-sided)"] = results_model_two



########################################################
# save results
########################################################



import pickle

bundle = {
    "results_all": results_all,
}

with open("results_garch_CQR_FACI.pkl", "wb") as f:
    pickle.dump(bundle, f)



with open("results_garch_CQR_FACI.pkl", "rb") as f:
    bundle = pickle.load(f)

results_all = bundle["results_all"]

########################################################
# PLOT AND EVALUATE RESULTS
########################################################

output_dir = "./results_rolling_garch_CQR_one_sided_FACI"
os.makedirs(output_dir, exist_ok=True)

models_to_compare = list(results_all.keys())

# ======================================================
# 1) PLOT RETURNS + LOWER/UPPER RAW + LOWER/UPPER CP
# ======================================================

colors = plt.cm.tab10.colors

for ticker in tickers:

    plt.figure(figsize=(12, 5))

    plt.plot(
        rets[ticker].iloc[warm_up + 1:],
        color="black",
        lw=0.8,
        label="Return"
    )

    for model, col in zip(models_to_compare, colors):

        df = results_all[model][ticker]

        # -------------------------
        # CP lower / upper
        # -------------------------
        plt.plot(
            df.index,
            df["L"],
            color=col,
            lw=1.5,
            alpha=0.8,
            label=f"CP lower {model}"
        )

        plt.plot(
            df.index,
            df["U"],
            color=col,
            lw=1.5,
            ls="--",
            alpha=0.8,
            label=f"CP upper {model}"
        )

        # -------------------------
        # RAW lower / upper
        # -------------------------
        plt.plot(
            df.index,
            df["q_lower_raw"],
            color=col,
            lw=1.0,
            ls=":",
            alpha=0.9,
            label=f"RAW lower {model}"
        )

        plt.plot(
            df.index,
            df["q_upper_raw"],
            color=col,
            lw=1.0,
            ls="-.",
            alpha=0.9,
            label=f"RAW upper {model}"
        )

    plt.title(f"GARCH-t + FACI — {ticker}")
    plt.legend(ncol=2)
    plt.grid(True)

    output_path = os.path.join(
        output_dir,
        f"predictive_bands_{ticker}_FACI.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Salvato: {output_path}")


# ======================================================
# 2) EVALUATION LOWER / UPPER
# ======================================================

def evaluate_interval(df, q_col, alpha_target, side="lower"):

    y = df["r"].values
    q = df[q_col].values

    if side == "lower":
        covered = y >= q
    elif side == "upper":
        covered = y <= q
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    return dict(
        coverage=np.nanmean(covered),
        misscoverage=1.0 - np.nanmean(covered),
        exceedances=np.sum(~covered),
        target_alpha=alpha_target
    )


metrics = []

for model in models_to_compare:

    for ticker in tickers:

        df = results_all[model][ticker]

        # -------------------------
        # CP lower
        # -------------------------
        m_cp_lower = evaluate_interval(
            df,
            "L",
            alpha_target,
            side="lower"
        )
        m_cp_lower.update({
            "model": model,
            "asset": ticker,
            "band": "CP_lower"
        })
        metrics.append(m_cp_lower)

        # -------------------------
        # CP upper
        # -------------------------
        m_cp_upper = evaluate_interval(
            df,
            "U",
            alpha_target,
            side="upper"
        )
        m_cp_upper.update({
            "model": model,
            "asset": ticker,
            "band": "CP_upper"
        })
        metrics.append(m_cp_upper)

        # -------------------------
        # RAW lower
        # -------------------------
        m_raw_lower = evaluate_interval(
            df,
            "q_lower_raw",
            alpha_target,
            side="lower"
        )
        m_raw_lower.update({
            "model": model,
            "asset": ticker,
            "band": "RAW_lower"
        })
        metrics.append(m_raw_lower)

        # -------------------------
        # RAW upper
        # -------------------------
        m_raw_upper = evaluate_interval(
            df,
            "q_upper_raw",
            alpha_target,
            side="upper"
        )
        m_raw_upper.update({
            "model": model,
            "asset": ticker,
            "band": "RAW_upper"
        })
        metrics.append(m_raw_upper)


metrics_df = (
    pd.DataFrame(metrics)
      .set_index(["band", "model", "asset"])
      .round(4)
)

print("=== PERFORMANCE FACI — LOWER / UPPER — CP vs RAW ===")
print(metrics_df)


# ======================================================
# 3) PLOT VIOLATIONS LOWER / UPPER
# ======================================================

def plot_var_violations(
    dates,
    y,
    q,
    model,
    ticker,
    title="Out-of-sample returns & VaR",
    color="tab:red",
    quant="PM",
    side="lower"
):
    """
    side='lower':
        violation if y <= q

    side='upper':
        violation if y >= q
    """

    plt.figure(figsize=(12, 4))

    plt.plot(dates, y, lw=1.0, label="Returns")
    plt.plot(dates, q, lw=1.5, color=color, label=f"{side} bound")

    if side == "lower":
        viol = y <= q
    elif side == "upper":
        viol = y >= q
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    plt.scatter(
        np.array(dates)[viol],
        y[viol],
        marker="o",
        s=20,
        facecolors="none",
        edgecolors="k",
        label="Violations"
    )

    plt.title(title)
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(
        output_dir,
        f"{quant}_plot_violations_{side}_{model}_{ticker}_FACI.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Salvato: {output_path}")


# ======================================================
# 4) SAVE PLOT DATA
# ======================================================

def save_plot_data(
    dates,
    y,
    q,
    model,
    ticker,
    quant,          # PM / CP
    side,           # lower / upper
    score,          # raw / truncated
    outdir="plots_data_FACI"
):
    subdir = f"{quant}_{side}_{score}"
    os.makedirs(os.path.join(outdir, subdir), exist_ok=True)

    fname = f"{model}_{ticker}.npz"
    path = os.path.join(outdir, subdir, fname)

    np.savez(
        path,
        dates=np.asarray(dates),
        y=np.asarray(y),
        q=np.asarray(q),
        model=model,
        ticker=ticker,
        quant=quant,
        side=side,
        score=score
    )

    print(f"💾 Salvato plot data: {path}")


# ======================================================
# 5) BACKTEST RAW LOWER / UPPER
# ======================================================

alpha = alpha_target
bt_rows_raw = []

baseline_model = models_to_compare[0]

for ticker in results_all[baseline_model]:

    df = results_all[baseline_model][ticker].dropna(
        subset=["q_lower_raw", "q_upper_raw", "r"]
    )

    y = df["r"].values

    # -------------------------
    # RAW LOWER
    # -------------------------
    q_lower = df["q_lower_raw"].values

    summ, det, I = backtest_var(
        y,
        q_lower,
        alpha=alpha,
        lags=lags_dq
    )

    summ.update({
        "model": "garch_parametric_raw",
        "asset": ticker,
        "side": "lower"
    })

    bt_rows_raw.append(summ)

    save_plot_data(
        dates=df.index.to_numpy(),
        y=y,
        q=q_lower,
        model="garch_parametric_raw",
        ticker=ticker,
        quant="PM",
        side="lower",
        score="raw"
    )

    plot_var_violations(
        df.index.to_numpy(),
        y,
        q_lower,
        model="garch_parametric_raw",
        ticker=ticker,
        title=f"Lower violations — RAW GARCH / {ticker}",
        color="tab:red",
        quant="PM",
        side="lower"
    )

    # -------------------------
    # RAW UPPER
    # -------------------------
    q_upper = df["q_upper_raw"].values

    # upper-tail backtest as lower-tail on -y and -q_upper
    summ, det, I = backtest_var(
        -y,
        -q_upper,
        alpha=alpha,
        lags=lags_dq
    )

    summ.update({
        "model": "garch_parametric_raw",
        "asset": ticker,
        "side": "upper"
    })

    bt_rows_raw.append(summ)

    save_plot_data(
        dates=df.index.to_numpy(),
        y=y,
        q=q_upper,
        model="garch_parametric_raw",
        ticker=ticker,
        quant="PM",
        side="upper",
        score="raw"
    )

    plot_var_violations(
        df.index.to_numpy(),
        y,
        q_upper,
        model="garch_parametric_raw",
        ticker=ticker,
        title=f"Upper violations — RAW GARCH / {ticker}",
        color="tab:red",
        quant="PM",
        side="upper"
    )


bt_table_raw = pd.DataFrame(bt_rows_raw).set_index(
    ["model", "asset", "side"]
)

print("=== VaR BACKTEST — PARAMETRIC GARCH LOWER / UPPER ===")
print(bt_table_raw.round(4))

bt_table_raw.to_csv(
    "BACKTEST_GARCH_PARAMETRIC_FACI_lower_upper.csv",
    index=True
)


# ======================================================
# 6) BACKTEST CP LOWER / UPPER
# ======================================================

alpha = alpha_target
bt_rows_cp = []

for model in models_to_compare:

    for ticker in results_all[model]:

        df = results_all[model][ticker].dropna(
            subset=["L", "U", "r"]
        )

        y = df["r"].values

        # -------------------------
        # CP LOWER
        # -------------------------
        q_lower = df["L"].values

        summ, det, I = backtest_var(
            y,
            q_lower,
            alpha=alpha,
            lags=lags_dq
        )

        summ.update({
            "model": model,
            "asset": ticker,
            "side": "lower"
        })

        bt_rows_cp.append(summ)

        save_plot_data(
            dates=df.index.to_numpy(),
            y=y,
            q=q_lower,
            model=model,
            ticker=ticker,
            quant="CP",
            side="lower",
            score="raw"
        )

        plot_var_violations(
            df.index.to_numpy(),
            y,
            q_lower,
            model=model,
            ticker=ticker,
            title=f"Lower violations — CP {model} / {ticker}",
            color="tab:blue",
            quant="CP",
            side="lower"
        )

        # -------------------------
        # CP UPPER
        # -------------------------
        q_upper = df["U"].values

        # upper-tail backtest as lower-tail on -y and -q_upper
        summ, det, I = backtest_var(
            -y,
            -q_upper,
            alpha=alpha,
            lags=lags_dq
        )

        summ.update({
            "model": model,
            "asset": ticker,
            "side": "upper"
        })

        bt_rows_cp.append(summ)

        save_plot_data(
            dates=df.index.to_numpy(),
            y=y,
            q=q_upper,
            model=model,
            ticker=ticker,
            quant="CP",
            side="upper",
            score="raw"
        )

        plot_var_violations(
            df.index.to_numpy(),
            y,
            q_upper,
            model=model,
            ticker=ticker,
            title=f"Upper violations — CP {model} / {ticker}",
            color="tab:blue",
            quant="CP",
            side="upper"
        )


bt_table_cp = pd.DataFrame(bt_rows_cp).set_index(
    ["model", "asset", "side"]
)

print("=== VaR BACKTEST — CP LOWER / UPPER ===")
print(bt_table_cp.round(4))

bt_table_cp.to_csv(
    "BACKTEST_GARCH_CP_FACI_lower_upper.csv",
    index=True
)


# ======================================================
# 7) COMPARISON PLOT LOWER + UPPER
# ======================================================

def plot_var_all_models_single_raw(
    results_all,
    rets,
    ticker,
    warm_up,
    output_dir
):

    models = list(results_all.keys())

    dates = rets.index[warm_up + 1:]
    y = rets[ticker].iloc[warm_up + 1:]

    plt.figure(figsize=(14, 6))

    # -------------------------------
    # Returns
    # -------------------------------
    plt.plot(
        dates,
        y,
        color="black",
        lw=0.8,
        label="Returns"
    )

    # -------------------------------
    # RAW lower / upper only once
    # -------------------------------
    first_model = models[0]
    df_raw = results_all[first_model][ticker]

    plt.plot(
        df_raw.index,
        df_raw["q_lower_raw"],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="GARCH raw lower"
    )

    plt.plot(
        df_raw.index,
        df_raw["q_upper_raw"],
        color="red",
        linestyle="-.",
        linewidth=2.0,
        label="GARCH raw upper"
    )

    # -------------------------------
    # CP lower / upper for each model
    # -------------------------------
    colors = ["green", "blue", "purple", "orange"]

    for model, col in zip(models, colors):

        df = results_all[model][ticker]

        plt.plot(
            df.index,
            df["L"],
            color=col,
            linewidth=1.8,
            alpha=0.9,
            label=f"{model} lower"
        )

        plt.plot(
            df.index,
            df["U"],
            color=col,
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"{model} upper"
        )

    plt.title(f"Lower/Upper prediction comparison — {ticker}")
    plt.ylabel("Return / bound")
    plt.xlabel("Date")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"VAR_comparison_lower_upper_{ticker}_CQR.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: {output_path}")

    plt.show()


# ======================================================
# 8) CUMULATIVE COVERAGE LOWER / UPPER
# ======================================================

def compute_cumulative_coverage(y, q, side="lower"):
    """
    side='lower':
        Coverage_t = mean(y_i >= q_i)

    side='upper':
        Coverage_t = mean(y_i <= q_i)
    """

    y = np.asarray(y, float)
    q = np.asarray(q, float)

    if side == "lower":
        covered = (y >= q).astype(int)
    elif side == "upper":
        covered = (y <= q).astype(int)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    cum_coverage = np.cumsum(covered) / np.arange(1, len(covered) + 1)

    return cum_coverage


def plot_coverage_all_models(
    results_all,
    rets,
    ticker,
    warm_up,
    alpha_target,
    output_dir,
    side="lower"
):

    plt.figure(figsize=(12, 6))

    target_cov = 1.0 - alpha_target

    # -------------------------------
    # choose columns by side
    # -------------------------------
    if side == "lower":
        raw_col = "q_lower_raw"
        cp_col = "L"
    elif side == "upper":
        raw_col = "q_upper_raw"
        cp_col = "U"
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    # -------------------------------
    # RAW baseline
    # -------------------------------
    first_model = list(results_all.keys())[0]

    df_raw = results_all[first_model][ticker].dropna(
        subset=[raw_col, "r"]
    )

    y_raw = df_raw["r"].values
    q_raw = df_raw[raw_col].values

    cum_cov_raw = compute_cumulative_coverage(
        y_raw,
        q_raw,
        side=side
    )

    plt.plot(
        df_raw.index,
        cum_cov_raw,
        color="black",
        linewidth=2.5,
        linestyle=":",
        label="GARCH raw"
    )

    # -------------------------------
    # CP models
    # -------------------------------
    for model in results_all:

        df = results_all[model][ticker].dropna(
            subset=[cp_col, "r"]
        )

        y = df["r"].values
        q = df[cp_col].values

        cum_cov = compute_cumulative_coverage(
            y,
            q,
            side=side
        )

        plt.plot(
            df.index,
            cum_cov,
            linewidth=1.5,
            label=model
        )

    # -------------------------------
    # Target coverage
    # -------------------------------
    plt.axhline(
        target_cov,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target = {target_cov:.2f}"
    )

    plt.title(f"Cumulative {side} coverage over time — {ticker}")
    plt.ylabel("Coverage")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)

    plt.ylim(0.8, 1.0)
    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"COVERAGE_{side}_{ticker}_CQR.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: {output_path}")

    plt.show()


# ======================================================
# 9) FINAL PLOTS FOR ALL TICKERS
# ======================================================

for ticker in tickers:

    print(f"📊 Plotting lower/upper comparison for {ticker}")

    plot_var_all_models_single_raw(
        results_all=results_all,
        rets=rets,
        ticker=ticker,
        warm_up=warm_up,
        output_dir=output_dir
    )

    plot_coverage_all_models(
        results_all=results_all,
        rets=rets,
        ticker=ticker,
        warm_up=warm_up,
        alpha_target=alpha_target,
        output_dir=output_dir,
        side="lower"
    )

    plot_coverage_all_models(
        results_all=results_all,
        rets=rets,
        ticker=ticker,
        warm_up=warm_up,
        alpha_target=alpha_target,
        output_dir=output_dir,
        side="upper"
    )
    
    
    
    
    
    