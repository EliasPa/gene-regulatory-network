import numpy as np
from matplotlib import pyplot as plt
from data import data
from utils.utils import plot_result
from gradient_matching import gradient_matching
from gradient_matching import gradient_matching_model_2
from gradient_matching import gradient_matching_model_3
from validation.LOOCV import poly_loocv


def main():
    gene_data = data.gene_data
    t_data = data.t_data
    data_legend = ["SWI5", "CBF1", "GAL4", "GAL80", "ASH1"]
    models = ["Full model", "Reduced model", "Extreme model"]
    #plot_result(t_data, gene_data, "Gene network", data_legend)
    # Full model
    gradient_matching(t_data, gene_data)

    # Reduced model
    gradient_matching_model_2(t_data, gene_data)

    # Model with highest up-down regulating coefficient
    gradient_matching_model_3(t_data, gene_data)

    # plot_result(t_data, gene_data, "Gene network", data_legend)
    # gradient_matching(t_data, gene_data)
    print("\n-----------------------------------------------------------")
    print("-------- LOSSES FROM LEAVE-ONE-OUT CROSS-VALIDATION--------")
    print("-----------------------------------------------------------\n")
    for i, model in enumerate(models):
      loss, losses = poly_loocv(t_data, gene_data, i)
      print('-'*len(model))
      print(model)
      print('-'*len(model))
      print("Loss per gene: {0}".format(losses))
      print("Average loss: {0} \n".format(loss))
      print('\n')

    print("Program finished succesfully. Output can be seen above.")


main()
