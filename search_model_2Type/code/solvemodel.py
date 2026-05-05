import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"Computes hazard rate and expected reemployment wages"
def predictedMoments(xi, b, s, logphi):
    
    "Extension: new parameters"
    #delta, k, gamma, mu_S, sigma = xi
    delta, k, gamma, mu_S, sigma, eta, alpha, = xi
    
    lastperiod = len(b)
    muv = np.full(lastperiod, mu_S)
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


"#Solves optimal search effort and reservation wage"
def optimalPath(xi, b):
    
    "Extension: new parameters"
    #delta, k, gamma, mu_S, sigma = xi
    delta, k, gamma, mu_S, sigma, eta, alpha, = xi
    
    lastperiod = len(b)
    muv = np.full(lastperiod, mu_S)
    s = np.zeros(lastperiod)
    logphi = np.zeros(lastperiod)

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
            
            
        #s[t] = min((1 / k * delta / (1 - delta) * integral) ** (1 / gamma), 1)
        
        "Extension: new search effort FOC"
        
        #penalty = eta * (np.log(b[t]) - np.log(alpha * b[t]))
        penalty = -eta*(np.log(alpha))

        s[t] = min((1 / k * (delta / (1 - delta) * integral + penalty)) ** (1 / gamma),1)
        
        "Extension: new utility function for reservation wage"
    
        "Penalty propability function"
        p = eta * (1 - s[t])
        p = np.clip(p, 0.0, 1.0)
     
        u_new = (1 - p) * np.log(b[t]) + p * np.log(alpha * b[t])
        
        logphi[t] = (
            (1 - delta) * (u_new - k * (s[t] ** (1 + gamma)) / (1 + gamma))
            + delta * logphi[t + 1]
            + delta * s[t] * integral
        )

    return s, logphi

"Solves for steady-state values of search effort and reservation wage"
def steadyState(xi, b_S):
    
    "Extension: new parameters"
    #delta, k, gamma, mu_S, sigma = xi
    delta, k, gamma, mu_S, sigma, eta, alpha, = xi

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
        
        "Extension: new steady-state search effort and reservation wages"
        
        "Search effort FOC"
        #f1 = s - (1 / k * delta / (1 - delta) * integral) ** (1 / gamma)
        
        penalty = eta * (np.log(b_S) - np.log(alpha * b_S))
        
        f1 = s - (1 / k * (delta / (1 - delta) * integral + penalty)) ** (1 / gamma)
        
        "Reservation wage FOC"
        #f2 = (
            #-q
            #+ np.log(b_S)
            #- k * (min(s, 1) ** (1 + gamma)) / (1 + gamma)
            #+ delta / (1 - delta) * min(s, 1) * integral
        #)
        
        p = eta * (1 - s)
        p = np.clip(p, 0.0, 1.0)

        u_new = (1 - p) * np.log(b_S) + p * np.log(alpha * b_S)
        
        f2 = (
            -q
            + u_new
            - k * (min(s, 1) ** (1 + gamma)) / (1 + gamma)
            + delta / (1 - delta) * min(s, 1) * integral
        )

        #Return sum of squares of the deviations
        return f1**2 + f2**2

    #Bounds for s and q
    bounds = [(0, None), (0, None)]

    #Initial guess
    x0 = [0.05, mu_S]

    #Solve the system of equations
    res = minimize(steadyStateSystem, x0, bounds=bounds, method="L-BFGS-B")
    s_S = res.x[0]
    logphi_S = res.x[1]
    
    s_S_cap = min(s_S,1)

    return s_S_cap, logphi_S

"Solves the model for single type worker with parameters, moments and optimal behaviour"
def solveModel(xi, institutions):
    T, b = institutions

    s, logphi = optimalPath(xi, b)
    haz, logw_reemp = predictedMoments(xi, b, s, logphi)
    surv = np.ones(len(b))
    for t in range(1, len(b)):
        surv[t] = surv[t - 1] * (1 - haz[t - 1])
    
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

    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp

"Computes elasticity of search effort w.r.t. benefits"
def computeElasticity(xi, institutions, eps=1e-4):
    
    T, b = institutions

    #Baseline
    s_base, _, _, _, _, _, _ = solveModel(xi, institutions)

    #Perturb benefits proportionally
    b_up = b * (1 + eps)
    institutions_up = (T, b_up)

    s_up, _, _, _, _, _, _ = solveModel(xi, institutions_up)

    #Elasticity
    elasticity = (np.log(s_up + 1e-8) - np.log(s_base + 1e-8)) / np.log(1 + eps)

    return elasticity

"Solves the model for multiple type workers"
def solveMultiTypeModel(params, institutions):

    delta, k1, gamma, mu1, sigma, k2, mu2, q2 = params
   
    "Extension: new parameters"
    #xi = np.array([delta, k1, gamma, mu1, sigma])
    xi = np.array([delta, k1, gamma, mu1, sigma, eta, alpha])

    #Variables for 2-type estimation
    q1 = 1 - q2

    #Type 1
    s1, logphi1, haz1, w_reemp1, surv1, D1, E_w_reemp1 = solveModel(xi, institutions)

    #Type 2
    xi[1] = k2  
    xi[3] = mu2
    s2, logphi2, haz2, w_reemp2, surv2, D2, E_w_reemp2 = solveModel(
        xi, institutions
    )

    #Aggregate Survival
    survival = q1 * surv1 + q2 * surv2

    #Create a mask for elements in 'survival' that are not equal to 0
    mask = survival != 0

    #Initialize 'weight1' with zeros
    weight1 = np.zeros_like(survival, dtype=float)
    weight2 = np.zeros_like(survival, dtype=float)

    #Perform the division only where 'mask' is True
    weight1[mask] = q1 * surv1[mask] / survival[mask]
    weight2[mask] = q2 * surv2[mask] / survival[mask]

    #Calculate aggregate hazard and average reemployment wage of leavers
    haz_agg = weight1 * haz1 + weight2 * haz2
    if min(haz_agg)==0:
        w_reemp_agg = (weight1 * w_reemp1 + weight2  * w_reemp2)
    else:
        w_reemp_agg = (weight1 * haz1 * w_reemp1 + weight2 * haz2 * w_reemp2) / haz_agg

    # Aggregate expected duration
    D = q1 * D1 + q2 * D2

    # Aggregate expected wage
    W = q1 * E_w_reemp1 + q2 * E_w_reemp2

    return haz_agg, w_reemp_agg, survival, D, W

import os
from matplotlib.backends.backend_pdf import PdfPages

output_dir = r"C:\Users\rohan\OneDrive\Documents\Aarhus Uni\8. semester\Micro and Macro Models of the Labour Market\PROJECT\Code\HandbookChapter\search_model_2Type\log\figures_Est1_compiled"
pdf_path = os.path.join(output_dir, "all_figures.pdf")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":

    #Calibrated parameters
    params = np.array([
        0.98,   # delta
        12.0,   # k1
        0.2,    # gamma
        4.0,    # mu1
        0.5,    # sigma
        50.0,   # k2
        4.0,    # mu2
        0.5,    # q2
    ])
    
    "Extension: new parameters eta and alpha"
    eta = 0.51 #Higher eta = higher baseline penalty and slope of search effort
    alpha = 0.5 #Higher alpha = lower penalty

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
    
    #Extract parameters 
    delta, k1, gamma, mu1, sigma, k2, mu2, q2 = params

    "Extension: new parameters"
    #xi = np.array([delta, k1, gamma, mu1, sigma])
    xi = np.array([delta, k1, gamma, mu1, sigma, eta, alpha])
    
    #Type 1
    s1, logphi1, haz1, w1, surv1, D1, Ew1 = solveModel(xi, inst1)

    #Type 2
    xi[1] = k2
    xi[3] = mu2
    s2, logphi2, haz2, w2, surv2, D2, Ew2 = solveModel(xi, inst1)
    
    #Weights
    q1 = 1 - q2
    denom = np.maximum(q1 * surv1 + q2 * surv2, 1e-8)
    weight1 = q1 * surv1 / denom
    weight2 = q2 * surv2 / denom
    
    import matplotlib.pyplot as plt

#Plots

with PdfPages(pdf_path) as pdf:
        
#Agg Search Effort

    fig = plt.figure()
    plt.plot(timevec, s1, label="Type 1", linestyle="--")
    plt.plot(timevec, s2, label="Type 2", linestyle="-.")
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Hazard")
    plt.title("Exit Hazard")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fig_type_agg_hazard.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)

#Agg Log reemployment wages

    fig = plt.figure()
    plt.plot(timevec, w1, label="Type 1", linestyle="--")
    plt.plot(timevec, w2, label="Type 2", linestyle="-.")
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
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
    plt.plot([], [], ' ', label=f'α = {alpha}')
    plt.axvline(x=12, color="gray", linestyle="dashed")
    plt.axvline(x=18, color="gray", linestyle="dashed")
    plt.xlabel("Months")
    plt.ylabel("Log Wage")
    plt.title("Reemployment Wage")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "fig_wage.png"), bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)

    

# Type-specific plots

    for i, (k_val, mu_val) in enumerate([(k1, mu1), (k2, mu2)], start=1):

        "Extension: new parameters"
        #xi = np.array([delta, k_val, gamma, mu_val, sigma])
        xi = np.array([delta, k_val, gamma, mu_val, sigma, eta, alpha])

        s_12, logphi_12, haz_12, w_12, *_ = solveModel(xi, inst1)
        s_18, logphi_18, haz_18, w_18, *_ = solveModel(xi, inst2)

# Search effort
        fig = plt.figure()
        plt.plot(timevec, s_12, label="P=12")
        plt.plot(timevec, s_18, label="P=18")
        plt.plot([], [], ' ', label=f'α = {alpha}')
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
        plt.plot([], [], ' ', label=f'α = {alpha}')
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
        plt.plot([], [], ' ', label=f'α = {alpha}')
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
        plt.plot([], [], ' ', label=f'α = {alpha}')
        plt.axvline(x=12, linestyle="dashed")
        plt.axvline(x=18, linestyle="dashed")
        plt.xlabel("Months")
        plt.ylabel("Log Reemployment Wage")
        plt.title(f"Type {i}: Log Reemployment Wage")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"fig_type{i}_w.png"), bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)
        
        
#Combined plots for different alphas

        alpha_values = [0.1, 0.5, 0.9]
        
# Search effort for different alphas
        
        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([
                delta, k_val, gamma, mu_val, sigma,
                eta, alpha_val
            ])
            
            s_alpha, _, _, _, _, _, _ = solveModel(xi_alpha, inst1)
            
            plt.plot(timevec, s_alpha, label=f"α={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.ylim(bottom=0, top=1.0)
        plt.xlabel("Months")
        plt.ylabel("Search Effort")
        plt.title(f"Type {i}: Search Effort for different α")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_search_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     
        
# Hazard rate for different alphas
        
        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([
                delta, k_val, gamma, mu_val, sigma,
                eta, alpha_val
            ])
            
            _, _, haz_alpha, _, _, _, _ = solveModel(xi_alpha, inst1)
            
            plt.plot(timevec, haz_alpha, label=f"α={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.ylim(bottom=0, top=1.0)
        plt.xlabel("Months")
        plt.ylabel("Hazard")
        plt.title(f"Type {i}: Hazard for different α")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_haz_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     
        
# Reservation wage for different alphas
        
        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([
                delta, k_val, gamma, mu_val, sigma,
                eta, alpha_val
            ])
            
            _, logphi_alpha, _, _, _, _, _ = solveModel(xi_alpha, inst1)
            
            plt.plot(timevec, logphi_alpha, label=f"α={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.ylim(bottom=3.0, top=4.5)
        plt.xlabel("Months")
        plt.ylabel("Log Reservation Wage")
        plt.title(f"Type {i}: Reservation Wage for different α")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_phi_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     
        
# Reemployment wage for different alphas

        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([
                delta, k_val, gamma, mu_val, sigma,
                eta, alpha_val
            ])
            
            _, _, _, w_alpha, _, _, _ = solveModel(xi_alpha, inst1)
            
            plt.plot(timevec, w_alpha, label=f"α={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.ylim(bottom=4.0, top=4.5)
        plt.xlabel("Months")
        plt.ylabel("Log Reemployment Wage")
        plt.title(f"Type {i}: Reemployment Wage for different α")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_wage_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     
        
# Survival function for different alphas
        
        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([
                delta, k_val, gamma, mu_val, sigma,
                eta, alpha_val
            ])
            
            _, _, _, _, surv, _, _ = solveModel(xi_alpha, inst1)
            
            plt.plot(timevec, surv, label=f"alpha={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.xlabel("Months")
        plt.ylabel("Survival")
        plt.title(f"Type {i}: Survival Function for different α")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_surv_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     

# Elasticity of search effort wrt b

        fig = plt.figure()
        
        for alpha_val in alpha_values:
            
            xi_alpha = np.array([delta, k_val, gamma, mu_val, sigma,
                                 eta, alpha_val])
            
            elasticity = computeElasticity(xi_alpha, inst1)
            
            plt.plot(timevec, elasticity, label=f"α={alpha_val}")
        
        plt.axvline(x=12, linestyle="dashed",color="tab:grey")
        plt.ylim(bottom=-3.5, top=0)
        plt.xlabel("Months")
        plt.ylabel("Elasticity")
        plt.title(f"Type {i}: Elasticity of Search Effort w.r.t. Benefits")
        plt.legend()
        
        filename = os.path.join(output_dir, f"fig_type{i}_elasticity_multiple_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)  
        plt.close(fig)     

# Unemployment duration for different alphas 

        fig = plt.figure()
        
        D_type1 = []
        D_type2 = []
        
        for alpha_val in alpha_values:
            
            # --- Type 1 ---
            xi_1 = np.array([
                delta, k1, gamma, mu1, sigma,
                eta, alpha_val
            ])
            _, _, _, _, _, D1_alpha, _ = solveModel(xi_1, inst1)
            D_type1.append(D1_alpha)
            
            # --- Type 2 ---
            xi_2 = np.array([
                delta, k2, gamma, mu2, sigma,
                eta, alpha_val
            ])
            _, _, _, _, _, D2_alpha, _ = solveModel(xi_2, inst1)
            D_type2.append(D2_alpha)
        
        
        plt.plot(alpha_values, D_type1, color="tab:blue", label="Type 1")
        plt.plot(alpha_values, D_type2, color="tab:orange", label="Type 2")
        plt.axhline(y=12, linestyle="dashed", color="tab:grey")
        plt.xlabel("α")
        plt.ylabel("Expected Duration")
        plt.title("Unemployment Duration for different α")
        
        plt.legend()
        
        filename = os.path.join(output_dir, "fig_duration_types_vs_alpha.png")
        
        plt.savefig(filename, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)