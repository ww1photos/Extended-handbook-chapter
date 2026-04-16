import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

momentsfile = "../data/base_moments_Germany_wages.xlsx"

alpha = 0.9 #Extension - penalty for being below search effort threshold
s_min = 0.1 #Extension - search effort threshold

def mu(xi, t):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    return mu_S + pi * np.minimum(t - kappa, np.zeros(len(t)))


def predictedMoments(xi, b, s, logphi):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    haz = np.zeros(len(b))
    logw_reemp = np.zeros(len(b))

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t]) / sigma
        if omega >= 7:
            omega = 7
        haz[t] = s[t] * (1 - norm.cdf(omega))
        logw_reemp[t] = muv[t] + sigma * norm.pdf(omega) / (
            1 - norm.cdf(omega)
        )

    haz[-1] = haz[-2]
    logw_reemp[-1] = logw_reemp[-2]

    return haz, logw_reemp


def optimalPath(xi, b):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    s = np.zeros(lastperiod)
    logphi = np.zeros(lastperiod)

    # Assuming steadyState is another function to compute steady state
    s[-1], logphi[-1] = steadyState(xi, b[-1])

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t + 1]) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            muv[t + 1]
            - logphi[t + 1]
            + sigma
            * norm.pdf(omega)
            / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0
            
        rhs = (delta / (1 - delta) * integral) / k
        rhs = max(rhs, 1e-8)
        
        s[t] = min(rhs ** (1 / gamma), 1)
        
     # --- Lagged benefit ---
        if t == 0:
            s_lag = s_min  
        else:
            s_lag = s[t-1]
     
        if s_lag < s_min:
            b_eff = b[t] * max(1 - alpha * (s_min - s_lag), 0)
        else:
            b_eff = b[t]
     
        b_eff = max(b_eff, 1e-6)
     
        logphi[t] = (
         (1 - delta) * (np.log(b_eff) - k * (s[t] ** (1 + gamma)) / (1 + gamma))
         + delta * logphi[t + 1]
         + delta * s[t] * integral
     )

    return s, logphi

def steadyState(xi, b_S):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi

    # @njit(cache=True)
    def steadyStateSystem(x):
        s, q = x
        omega = (q - mu_S) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            mu_S
            - q
            + sigma * norm.pdf(omega) / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0

        
        rhs = (delta / (1 - delta) * integral) / k
        rhs = max(rhs, 1e-8)
        
        f1 = s - rhs ** (1 / gamma)
      
        #Extension - New steady-state reservation wage
        
        if s < s_min:
            b_eff = b_S * max(1 - alpha * (s_min - s), 0)
        else:
            b_eff = b_S
        
        b_eff = max(b_eff, 1e-6)
        
        f2 = (
            -q
            + np.log(b_eff)
            - k * (min(s, 1) ** (1 + gamma)) / (1 + gamma)
            + delta / (1 - delta) * min(s, 1) * integral
        )

        # Return sum of squares of the deviations
        return f1**2 + f2**2

    # Bounds for s and q
    # bounds = [(0, 1), (0, None)]  # s between 0 and 1, q greater than 0
    bounds = [(0, None), (0, None)]  # s between 0 and 1, q greater than 0

    # Initial guess
    x0 = [0.05, mu_S]

    # Solve the system of equations
    res = minimize(steadyStateSystem, x0, bounds=bounds, method="L-BFGS-B")
    s_S = res.x[0]
    logphi_S = res.x[1]

    # print(res.x)
    # optimized_x = gradient_descent(steadyStateSystem, x0, bounds)
    # s_S = optimized_x[0]
    # logphi_S = optimized_x[1]
    # print(optimized_x)
    
    s_S_cap = min(s_S,1)

    return s_S_cap, logphi_S


def solveModel(xi, institutions):
    T, b = institutions

    s, logphi = optimalPath(xi, b)
    haz, logw_reemp = predictedMoments(xi, b, s, logphi)
    surv = np.ones(len(b))
    for t in range(1, len(b)):
        surv[t] = surv[t - 1] * (1 - haz[t - 1])
    # T=96
    
    Tminb = T - len(b)
    haz_long = np.concatenate((haz, np.ones(Tminb) * haz[-1]))
    logw_reemp_long = np.concatenate((logw_reemp, np.ones(Tminb) * logw_reemp[-1]))

    surv_long = np.ones(T)
    for t in range(1, T):
        surv_long[t] = surv_long[t - 1] * (1 - haz_long[t - 1])

    D = np.sum(surv_long)
    dens_long = haz_long * surv_long
    if np.sum(dens_long)!=0:
        E_logw_reemp = np.sum(dens_long * logw_reemp_long) / np.sum(dens_long)
    else: 
        E_logw_reemp = 0

    # E_logw_reemp = 0
    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp

def solveMultiTypeModel(params, institutions):
    """
    Solves the  model in a multi-type setup.
        Arguments:
            params (array): Array of parameter values for all types.
        Returns:
    """

    delta, k1, gamma, mu1, sigma, kappa, pi, k2, k3, k4, mu2, mu3, mu4, q2, q3, q4 = params

    # Parameters for single type model
    xi = np.copy(params[0:7])

    # Variables for 2-type estimation
    q1 = 1 - q2 - q3 - q4

    # Type 1
    s1, logphi1, haz1, w_reemp1, surv1, D1, E_w_reemp1 = solveModel(xi, institutions)

    # Type 2
    xi[1] = k2  # Adjust the k parameter for Type 2
    xi[3] = mu2
    s2, logphi2, haz2, w_reemp2, surv2, D2, E_w_reemp2 = solveModel(
        xi, institutions
    )

    # Type 3
    xi[1] = k3  # Adjust the k parameter for Type 2
    xi[3] = mu3
    s3, logphi3, haz3, w_reemp3, surv3, D3, E_w_reemp3 = solveModel(
        xi, institutions
    )

    # Type 4
    xi[1] = k4  # Adjust the k parameter for Type 2
    xi[3] = mu4
    s4, logphi4, haz4, w_reemp4, surv4, D4, E_w_reemp4 = solveModel(
        xi, institutions
    )

    # Aggregate Survival
    survival = q1 * surv1 + q2 * surv2 + q3 * surv3 + q4 * surv4

    # Calculate share of each type left at beginning in unemployment
    # weight1 = q1 * surv1 / survival
    # weight2 = q2 * surv2 / survival
    # weight3 = q3 * surv2 / survival
    # Create a mask for elements in 'survival' that are not equal to 0
    mask = survival != 0

    # Initialize 'weight1' with zeros
    weight1 = np.zeros_like(survival, dtype=float)
    weight2 = np.zeros_like(survival, dtype=float)
    weight3 = np.zeros_like(survival, dtype=float)
    weight4 = np.zeros_like(survival, dtype=float)

    # Perform the division only where 'mask' is True
    weight1[mask] = q1 * surv1[mask] / survival[mask]
    weight2[mask] = q2 * surv2[mask] / survival[mask]
    weight3[mask] = q3 * surv3[mask] / survival[mask]
    weight4[mask] = q4 * surv4[mask] / survival[mask]
    

    # Calculate aggregate hazard and average reemployment wage of leavers
    haz_agg = weight1 * haz1 + weight2 * haz2 + weight3 * haz3 + weight4 * haz4
    if min(haz_agg)==0:
        w_reemp_agg = (weight1 * w_reemp1 + weight2  * w_reemp2 + weight3  * w_reemp3 + weight4  * w_reemp4) 
    else:
        w_reemp_agg = (weight1 * haz1 * w_reemp1 + weight2 * haz2 * w_reemp2 + weight3 * haz3 * w_reemp3 + weight4 * haz4 * w_reemp4) / haz_agg

    # Aggregate expected duration
    D = q1 * D1 + q2 * D2 + q3 * D3 + q4 * D4

    # Aggregate expected wage
    W = q1 * E_w_reemp1 + q2 * E_w_reemp2 + q3 * E_w_reemp3 + q4 * E_w_reemp4

    return haz_agg, w_reemp_agg, survival, D, W


import os
from matplotlib.backends.backend_pdf import PdfPages

output_dir = r"C:\Users\rohan\OneDrive\Documents\Aarhus Uni\8. semester\Micro and Macro Models of the Labour Market\PROJECT\Code\Benchmark\search_model_2Type\log\figures_Est1_compiled"
pdf_path = os.path.join(output_dir, "all_figures.pdf")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":

    # === Estimated parameters (from estimate.py output) ===
    params = np.array([
        0.98,   # delta
        12.0,   # k1
        0.2,    # gamma
        4.0,    # mu1
        0.5,    # sigma
        24.0,   # kappa
        0.0,    # pi
        50.0,   # k2
        1.0,    # k3
        1.0,    # k4
        4.0,    # mu2
        2.5,    # mu3
        3.0,    # mu4
        0.5,    # q2
        0.0,    # q3
        0.0     # q4
    ])

    T = 31
    
    P1 = 12
    P2 = 18
    
    ben1 = np.ones(T) * 800 / 30
    ben2 = np.ones(T) * 800 / 30
    
    ben1[:P1] = 1100 / 30
    ben2[:P2] = 1100 / 30
    
    inst1 = T, ben1
    inst2 = T, ben2
    
    timevec = np.arange(T) + 1
    
    # === Extract parameters ===
    delta, k1, gamma, mu1, sigma, kappa, pi, k2, k3, k4, mu2, mu3, mu4, q2, q3, q4 = params

    xi = np.array([delta, k1, gamma, mu1, sigma, kappa, pi])

    # --- Type 1 ---
    s1, logphi1, haz1, w1, surv1, D1, Ew1 = solveModel(xi, inst1)

    # --- Type 2 ---
    xi[1] = k2
    xi[3] = mu2
    s2, logphi2, haz2, w2, surv2, D2, Ew2 = solveModel(xi, inst1)
    
    #Weights
    q1 = 1 - q2
    denom = np.maximum(q1 * surv1 + q2 * surv2, 1e-8)
    weight1 = q1 * surv1 / denom
    weight2 = q2 * surv2 / denom
    
    import matplotlib.pyplot as plt

                    #PLOTS
with PdfPages(pdf_path) as pdf:
        
#Agg Search Effort

    fig = plt.figure()
    plt.plot(timevec, s1, label="Type 1", linestyle="--")
    plt.plot(timevec, s2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.ylim(bottom=0, top=1.0)
    plt.xlabel("Months")
    plt.ylabel("Search Effort")
    plt.title("Search Effort over Time")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_s.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    
#Agg Reservation wage

    fig = plt.figure()
    plt.plot(timevec, logphi1, label="Type 1", linestyle="--")
    plt.plot(timevec, logphi2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Log Reservation Wage")
    plt.title("Log Reservation Wage")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_phi.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    
#Agg Survival function

    fig = plt.figure()
    plt.plot(timevec, surv1, label="Type 1", linestyle="--")
    plt.plot(timevec, surv2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Survival")
    plt.title("Survival Function")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_surv.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    
#Agg Type shares

    fig = plt.figure()
    plt.plot(timevec, weight1, label="Type 1", linestyle="--")
    plt.plot(timevec, weight2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.ylim(bottom=0, top=1.0)
    plt.xlabel("Months")
    plt.ylabel("Share")
    plt.title("Type Shares (Among Unemployed)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_share.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)

#Agg Hazard rates

    fig = plt.figure()
    plt.plot(timevec, haz1, label="Type 1", linestyle="--")
    plt.plot(timevec, haz2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Hazard")
    plt.title("Exit Hazard")
    plt.savefig(os.path.join(output_dir, "fig_type_agg_hazard.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)

#Agg Log reemployment wages

    fig = plt.figure()
    plt.plot(timevec, w1, label="Type 1", linestyle="--")
    plt.plot(timevec, w2, label="Type 2", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Log Wage")
    plt.title("Log Reemployment Wage")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_w.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    
#Aggregate hazard + wage with P = 12 and P = 18

    haz_agg_12, w_agg_12, surv_agg_12, D, W = solveMultiTypeModel(params, inst1)
    haz_agg_18, w_agg_18, surv_agg_18, D, W = solveMultiTypeModel(params, inst2)
    
    fig = plt.figure()
    plt.plot(timevec, haz_agg_12, label="Hazard, P=12", linestyle="--")
    plt.plot(timevec, haz_agg_18, label="Hazard, P=18", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.axvline(x=18, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Exit Hazard")
    plt.title("Exit Hazard")
    plt.legend(loc="lower right")
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, "fig_haz.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(timevec[:25], w_agg_12[:25], label="Wage, P=12", linestyle="--")
    plt.plot(timevec[:25], w_agg_18[:25], label="Wage, P=18", linestyle="-.")
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.axvline(x=18, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Log Wage")
    plt.title("Reemployment Wage")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "fig_wage.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)

    
# =========================
# 3. TYPE-SPECIFIC POLICY EFFECTS
# =========================
    for i, (k_val, mu_val) in enumerate([(k1, mu1), (k2, mu2)], start=1):

        xi = np.array([delta, k_val, gamma, mu_val, sigma, kappa, pi])

        s_12, logphi_12, haz_12, w_12, *_ = solveModel(xi, inst1)
        s_18, logphi_18, haz_18, w_18, *_ = solveModel(xi, inst2)

# Search effort
        fig = plt.figure()
        plt.plot(timevec, s_12, label="P=12")
        plt.plot(timevec, s_18, label="P=18")
        plt.axvline(x=12, linestyle="dashed")
        plt.axvline(x=18, linestyle="dashed")
        plt.title(f"Type {i}: Search Effort")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"fig_type{i}_s.png"), bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)


# Hazard

        fig = plt.figure()
        plt.plot(timevec, haz_12, label="P=12")
        plt.plot(timevec, haz_18, label="P=18")
        plt.axvline(x=12, linestyle="dashed")
        plt.axvline(x=18, linestyle="dashed")
        plt.title(f"Type {i}: Hazard")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"fig_type{i}_haz.png"), bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)
        
# Reservation wage

        fig = plt.figure()
        plt.plot(timevec, logphi_12, label="P=12", linestyle="--")
        plt.plot(timevec, logphi_18, label="P=18", linestyle="-.")
        plt.axvline(x=12, linestyle="dashed")
        plt.axvline(x=18, linestyle="dashed")
        plt.xlabel("Months")
        plt.ylabel("Log Reservation Wage")
        plt.title(f"Type {i}: Log Reservation Wage")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"fig_type{i}_phi.png"), bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)
        
# Reemployment wages

        fig = plt.figure()
        plt.plot(timevec, w_12, label="P=12", linestyle="--")
        plt.plot(timevec, w_18, label="P=18", linestyle="-.")
        plt.axvline(x=12, linestyle="dashed")
        plt.axvline(x=18, linestyle="dashed")
        plt.xlabel("Months")
        plt.ylabel("Log Reemployment Wage")
        plt.title(f"Type {i}: Log Reemployment Wage")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"fig_type{i}_w.png"), bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)