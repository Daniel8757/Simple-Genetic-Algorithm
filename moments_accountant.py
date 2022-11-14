import math

def dp_to_rdp(epsilon, alpha):
    return (1/(alpha-1)) * math.log((math.sinh(alpha*epsilon) - math.sinh((alpha-1)*epsilon))/math.sinh(epsilon))

# takes in rdp alpha, eplison, and a target delta, and return the dp
def rdp_tp_dp(alpha, epsilon, delta):
    return (epsilon + math.log(1/delta)/(alpha-1), delta)

# assume sensitivety one
def gausmech_to_rdp(alpha, sigma):
    return alpha/(2*(sigma**2))

def moments_accountant(sigma, delta, max_alpha, num_steps):
    # candidate of alphas
    candidates = []
    for alpha in range(2, max_alpha+1):
        # add the rdp of one utility sample
        candidates.append(gausmech_to_rdp(sigma, alpha))
    
    # compose over many samples for each alpha
    finished_candidates = []
    for alpha in range(2, max_alpha+1):
        finished_candidates.append((candidates[alpha-2]*num_steps, alpha))

    # convert to dp
    finished_dp = [rdp_tp_dp(alpha, epsilon, delta) for epsilon, alpha in finished_candidates]
    
    # get best candidate
    best_candidate = min(finished_dp)

    return best_candidate

print(moments_accountant(1, 0.00001, 5000, 100))