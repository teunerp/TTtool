import numpy as np

def Wbal3(data, CP, W):
    """
    This function calculates W'bal for a given day, taking into account

    """

    power = data["power"]  # Assuming MATLAB-style indexing where the fourth column is index 3
    # power = power.reset_index()
    # print(power)
    # print(CP)
    # print(W)
    t = tau_w3(power, CP)  # Assuming tau_w3 is the previously defined function
    Sr = 1

    n = len(power)
    P = np.zeros(n)
    Wexp = np.zeros(n)
    Sn = 0
    In = 0
    Wb = np.zeros(n)

    Wb[0] = W

    for i in range(1, n):
        if power[i] < CP:
            Wused = W - Wb[i - 1]
            Wused_new = Wused * np.exp(-1 * Sr / t[i])
            Wrec = Wused - Wused_new
            Wb[i] = Wb[i - 1] + Wrec
        else:
            Wexp[i] = power[i] - CP
            Wb[i] = Wb[i - 1] - Wexp[i]

    return Wb


def tau_w3(power, CP):
    """
    This function calculates tau_w for a given day. The input is a vector
    with power values and a double CP. The output is a double tau_w.
    """
    n = len(power)
    tau_w = np.zeros(n)  # Initializing tau_w as an array of zeros

    for i in range(n):
        # print(i)
        if power[i] < CP:
            Wrecovery = power[i]
#             tau_w[i] = 546 * np.exp(-0.01 * (CP - Wrecovery)) + 316 5187, 789,
            tau_w[i] = 797.4379*(CP-Wrecovery)**-0.4112
        else:
            tau_w[i] = 0  # You may adjust this part based on your specific logic

    return tau_w