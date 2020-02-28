import numpy as np
from matplotlib import pyplot as plt
from data import data
from utils.utils import plot_result
from gradient_matching import gradient_matching
from gradient_matching import gradient_matching_model_2
from gradient_matching import gradient_matching_model_3


def main():
    gene_data = data.gene_data
    t_data = data.t_data
    data_legend = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]
    #plot_result(t_data, gene_data, "Gene network", data_legend)
    # Full model
    gradient_matching(t_data, gene_data)

    # Reduced model
    gradient_matching_model_2(t_data, gene_data)

    # Model with highest up-down regulating coefficient
    gradient_matching_model_3(t_data, gene_data)


main()
