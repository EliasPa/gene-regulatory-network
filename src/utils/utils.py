import numpy as np
from matplotlib import pyplot as plt
import time as t


def bin_linear(T, X, bin_width, T_max):

    space = np.linspace(0, T_max, int(T_max / bin_width))
    binned = []
    for bin_start in space:
        bin_end = bin_start + bin_width
        data_in_range = [x for i, x in enumerate(X) if (
            T[i] >= bin_start and T[i] <= bin_end)]
        binned.append(np.mean(data_in_range, axis=0))

    return space, np.array(binned)


def simulate_many(S, M, lac_operon_hazards, c, T_max, P_NAMES, sim_type, sim_function, system_name, Nt=0, bw=30):
    N = 100
    averaged = []
    time = []
    execution_times = []
    for i in range(N):
        start_time = t.time()
        if sim_type == "Gillespie":
            T_g, X_g = sim_function(S, M, lac_operon_hazards, c, t_max=T_max)
            T, X = bin_linear(T_g, X_g, bw, T_max)
        elif sim_type == "Poisson":
            T, X = sim_function(S, M, lac_operon_hazards,
                                c, np.linspace(1, T_max, Nt))
        elif sim_type == "CLE":
            T, X = sim_function(S, M, lac_operon_hazards,
                                c, np.linspace(1, T_max, Nt))
        averaged.append(X)
        time = T
        execution_times.append(t.time() - start_time)

    av = np.mean(averaged, axis=0)
    print("Average execution time for {0} in {1} was {2}".format(
        sim_type, system_name, np.mean(execution_times)))
    plot_result(time, av, title="Averaged {0} {1}".format(
        sim_type, system_name), legend=P_NAMES)


def plot_result(T, X, title, legend):
    plt.figure(figsize=(10, 6))

    plt.plot(T, X[0, :])
    plt.plot(T, X[1, :])
    plt.plot(T, X[2, :])
    plt.plot(T, X[3, :])
    plt.plot(T, X[4, :])
    plt.title(title)
    plt.xlabel('Time')
    plt.legend(legend, loc='upper right')
    plt.show()
