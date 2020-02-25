import numpy as np
from matplotlib import pyplot as plt
from data import data
from utils.utils import plot_result
from gradient_matching import gradient_matching
from validation.LOOCV import poly_loocv


def main():
    gene_data = data.gene_data
    t_data = data.t_data
    data_legend = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]
    # plot_result(t_data, gene_data, "Gene network", data_legend)
    # gradient_matching(t_data, gene_data)
    total_loss = poly_loocv(t_data,gene_data)
    print("Total loss for regression model: {0}".format(total_loss))

main()
