import pygad
import numpy as np
import csv
import os
import math
import cmath
import time
import matplotlib.pyplot as plt
import random
from scipy.optimize import *
from scipy.signal import argrelextrema

def cv_integration(datasets, experiment_dict, mode, xmin, xmax):
    # This function allows one to calculate the ECSA
    fitting_parameters = []
    fitting_parameters.append(["Experiment", "ECSA using desorption (dimensionless)"])
    for dataset, exp_num in zip(datasets, experiment_dict.keys()):
        integral = 0
        ixmin = np.searchsorted(dataset[0], xmin)
        ixmax = np.searchsorted(dataset[0], xmax)
        for i in range(ixmin, ixmax):
            if dataset[1][i + 1] > 0 and dataset[1][i] > 0:
                integral += (dataset[0][i + 1] - dataset[0][i]) * (dataset[1][i + 1] + dataset[1][i]) / 2
        ECSA = integral / (220e-6 * 0.02)
        if mode == "auto":
            fitting_parameters.append([f'n°{exp_num}', ECSA])
        if mode == "manual":
            fitting_parameters.append([os.path.basename(exp_num), ECSA])
    return fitting_parameters

def randomStartingParameters(dataset_type, model, parameter_space):
    #print("random param")
    if dataset_type == "IMP":
        if model == "Linear":
            R_el = random.uniform(parameter_space[0]['low'], parameter_space[0]['high'])
            R1 = random.uniform(parameter_space[1]['low'], parameter_space[1]['high'])
            C1 = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            return [R_el, R1, C1]

        if model == "RQ":
            R_el = random.uniform(parameter_space[0]['low'], parameter_space[0]['high'])
            R1 = random.uniform(parameter_space[1]['low'], parameter_space[1]['high'])
            Q1 = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            alpha1 = random.uniform(parameter_space[3]['low'], parameter_space[3]['high'])
            return [R_el, R1, Q1, alpha1]

        if model == "DoubleRC":
            R_el = random.uniform(parameter_space[0]['low'], parameter_space[0]['high'])
            R1 = random.uniform(parameter_space[1]['low'], parameter_space[1]['high'])
            C1 = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            R2 = random.uniform(parameter_space[3]['low'], parameter_space[3]['high'])
            C2 = random.uniform(parameter_space[4]['low'], parameter_space[4]['high'])
            return [R_el, R1, C1, R2, C2]

        if model == "DoubleRQ":
            R_el = random.uniform(parameter_space[0]['low'], parameter_space[0]['high'])
            R1 = random.uniform(parameter_space[1]['low'], parameter_space[1]['high'])
            Q1 = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            alpha1 = random.uniform(parameter_space[3]['low'], parameter_space[3]['high'])
            R2 = random.uniform(parameter_space[4]['low'], parameter_space[4]['high'])
            Q2 = random.uniform(parameter_space[5]['low'], parameter_space[5]['high'])
            alpha2 = random.uniform(parameter_space[6]['low'], parameter_space[6]['high'])
            return [R_el, R1, Q1, alpha1, R2, Q2, alpha2]
# 60m/g maximum platinum after multiplkying the result

def findStartingParameters(dataset, dataset_type, model, parameter_space, type):
    #print("starting param")
    # This function returns the EEC's parameters to be used as starting parameters for the optimization
    # algorithms
    # It calculates all the parameters but only the resistances are used as it can be seen in makeParameterSpace function
    if type == "simple":
        print("ok")
        if dataset_type == "IMP":
            # Find R_el
            ranks_Real = rankList(dataset[1], "increasing")
            ranks_Im = rankList(dataset[2], "increasing")

            sorted_rank_total = sumRanks(ranks_Real, ranks_Im)

            L = []
            for index, tuple in enumerate(sorted_rank_total):
                if index < round(0.5 * len(dataset[0])) and tuple[1] < round(0.05 * len(dataset[0])):
                    L.append(dataset[1][tuple[1]])
            if L == []:
                L.append(1)
            # Mean of the five percent values with lowest Z' and lowest Z''
            R_el = sum(L) / len(L)

            # Find R_p
            ranks_Real = rankList(dataset[1], "decreasing")
            ranks_Im = rankList(dataset[2], "increasing")

            sorted_rank_total = sumRanks(ranks_Real, ranks_Im)

            L = []

            for index, tuple in enumerate(sorted_rank_total):
                if index < round(0.5 * len(dataset[0])) and tuple[1] > round(0.95 * len(dataset[0])):
                    L.append(dataset[1][tuple[1]])
            if L == []:
                L.append(1)

            # Mean of the five percent values with highest Z' and lowest Z''
            R_p = sum(L) / len(L)
            R_p -= R_el

            # Find f_c
            max_Im = max(dataset[2])
            max_Im_index = dataset[2].index(max_Im)
            f_c = dataset[0][max_Im_index]

    else:
        f = np.array((dataset[0]))
        R_exp = np.array(dataset[1])
        X_exp = np.array(dataset[2])

        # Calculate R_el
        five_percent_index = int(0.10 * len(R_exp))
        R_el = sum(R_exp[:five_percent_index]) / len(R_exp[:five_percent_index])

        # Calculate R_p
        window_size = int(0.15*len(f))
        kernel = np.ones(window_size) / window_size

        X_exp_convolve = np.convolve(X_exp, kernel, mode='same')
        minima_indices_smoothed = argrelextrema(X_exp_convolve, np.less)[0]
        if len(minima_indices_smoothed) == 0:
            #print("R", R_exp)
            #print("indexes", five_percent_index)
            R_p = sum(R_exp[five_percent_index:]) / len(R_exp[five_percent_index:])
            R_p -= R_el
        else:
            #R_p = R_exp[minima_indices_smoothed>0.5*len(f)]/len(minima_indices_smoothed)
            #R_p = R_exp[minima_indices_smoothed]/len(minima_indices_smoothed)
            filtered_indices = [index for index in minima_indices_smoothed if index > int(0.5*len(f))]
            R_p = sum(R_exp[filtered_indices])/len(filtered_indices)
            #print(R_exp[filtered_indices])
            R_p -= R_el
        # Calculate f_c
        max_Im = max(dataset[2])
        max_Im_index = dataset[2].index(max_Im)
        f_c = dataset[0][max_Im_index]

        if model == "Linear":
            R1 = R_p
            C1 = 1/(2 * np.pi * R1 * f_c)

            R1 = compareToParameterSpace(R1, parameter_space[1]['low'], parameter_space[1]['high'])
            C1 = compareToParameterSpace(C1, parameter_space[2]['low'], parameter_space[2]['high'])
            return [R_el, R1, C1]

        if model == "RQ":
            R1 = R_p
            alpha1 = 1
            Q1 = 1 / ((2 * np.pi * f_c)**alpha1 * R1)

            R1 = compareToParameterSpace(R1, parameter_space[1]['low'], parameter_space[1]['high'])
            Q1 = compareToParameterSpace(Q1, parameter_space[2]['low'], parameter_space[2]['high'])
            alpha1 = compareToParameterSpace(alpha1, parameter_space[3]['low'], parameter_space[3]['high'])
            return [R_el, R1, Q1, alpha1]

        if model == "DoubleRC":
            R1 = 0.3*R_p
            C1 = 1 / (2 * np.pi * R1 * f_c)
            R2 = 0.7*R_p
            C2 = 1 / (2 * np.pi * R2 * f_c)

            R1 = compareToParameterSpace(R1, parameter_space[1]['low'], parameter_space[1]['high'])
            C1 = compareToParameterSpace(C1, parameter_space[2]['low'], parameter_space[2]['high'])
            R2 = compareToParameterSpace(R1, parameter_space[3]['low'], parameter_space[3]['high'])
            C2 = compareToParameterSpace(C1, parameter_space[4]['low'], parameter_space[4]['high'])
            return [R_el, R1, C1, R2, C2]

        if model == "DoubleRQ":
            R1 = 0.4*R_p
            R2 = 0.6*R_p
            alpha1 = 1
            Q1 = 1 / ((2 * np.pi * f_c) ** alpha1 * R1)
            alpha2 = 1
            Q2 = 1 / ((2 * np.pi * f_c) ** alpha2 * R2)

            R1 = compareToParameterSpace(R1, parameter_space[1]['low'], parameter_space[1]['high'])
            Q1 = compareToParameterSpace(Q1, parameter_space[2]['low'], parameter_space[2]['high'])
            alpha1 = compareToParameterSpace(alpha1, parameter_space[3]['low'], parameter_space[3]['high'])
            R2 = compareToParameterSpace(R2, parameter_space[4]['low'], parameter_space[4]['high'])
            Q2 = compareToParameterSpace(Q2, parameter_space[5]['low'], parameter_space[5]['high'])
            alpha2 = compareToParameterSpace(alpha2, parameter_space[6]['low'], parameter_space[6]['high'])
            return [R_el, R1, Q1, alpha1, R2, Q2, alpha2]


def rankList(list, type):
    # This function is used by findStartingParameters
    indexed = [(val, index) for index, val in enumerate(list)]
    if type == "increasing":
        sorted_list = sorted(indexed)
    elif type == "decreasing":
        sorted_list = sorted(indexed, reverse=True)
    else:
        return
    ranks = [0] * len(list)

    for rank, (val, index) in enumerate(sorted_list):
        ranks[index] = rank

    return ranks


def sumRanks(ranks_Real, ranks_Im):
    # This function is used by findStartingParameters
    i = 0
    rank_total = []
    for rank1, rank2 in zip(ranks_Real, ranks_Im):
        rank_total.append((rank1 + rank2, i))
        i += 1
    sorted_rank_total = sorted(rank_total)
    return sorted_rank_total

def compareToParameterSpace(param, low_bound, high_bound):
    if param > high_bound:
        return high_bound
    if param < low_bound:
        return low_bound
    return param

def initializePopulation(starting_parameters, gene_space, sol_per_pop):
    population = np.zeros(shape=(sol_per_pop, len(starting_parameters)))
    for sol_idx in range(sol_per_pop):
        for gene_idx in range(len(starting_parameters)):
            '''if gene_idx != 0 and gene_idx != 1 and gene_idx != 4:
                # population[sol_idx][gene_idx] = np.asarray(np.random.normal(starting_parameters[gene_idx], 0.90 * starting_parameters[gene_idx], size=1))
                population[sol_idx][gene_idx] = np.asarray(
                    np.random.uniform(starting_parameters[gene_idx], 0.90 * starting_parameters[gene_idx], size=1))
                if population[sol_idx][gene_idx] > gene_space[gene_idx]['high']:
                    population[sol_idx][gene_idx] = gene_space[gene_idx]['high']
                elif population[sol_idx][gene_idx] < gene_space[gene_idx]['low']:
                    population[sol_idx][gene_idx] = gene_space[gene_idx]['low']

            else:'''
            population[sol_idx][gene_idx] = starting_parameters[gene_idx]
        '''if population[sol_idx][2] > population[sol_idx][5]:
            population[sol_idx][2] = 0.5 * population[sol_idx][5]'''
    return population


def appendFittingParameters(fitting_parameters, model, solution, exp_num):
    # Used to save values for impedance or polarisation
    if model == "Linear":
        fitting_parameters.append([exp_num, solution[0], solution[1], solution[2]])

    if model == "RQ":
        fitting_parameters.append(
            [exp_num, solution[0], solution[1], solution[2], solution[3]])
    if model == "Fouquet":
        fitting_parameters.append(
            [exp_num, solution[0], solution[1], solution[2], solution[3], solution[4], solution[5]])
    if model == "DoubleRC":
        fitting_parameters.append(
            [exp_num, solution[0], solution[1], solution[2], solution[3], solution[4]])
    if model == "DoubleRQ":
        fitting_parameters.append(
            [exp_num, solution[0],
             solution[1], solution[2], solution[3],
             solution[4], solution[5], solution[6]])

    if model == "NotLimited":
        fitting_parameters.append([exp_num, solution[0], solution[1], solution[2], solution[3], solution[4]])

    if model == "Limited":
        fitting_parameters.append(
            [exp_num, solution[0], solution[1], solution[2], solution[3], solution[4], solution[5], solution[6]])


def makeParameterSpace(dataset, search_space_dict, model):
    if model == "Linear" or model == "RQ" or model == "DoubleRC" or model == "DoubleRQ":
        R_low = 0
        R_high = 0.1
        Q_low = 0.001
        Q_high = 10
        alpha_low = 0.5
        alpha_high = 1
    if model == "NotLimited" or model == "Limited":
        In_low = 0.00001
        In_high = 0.1
        A_low = 0.000001
        A_high = 2
        I0_low = 0.0000001
        I0_high = 50
        E0_low = 0.1
        E0_high = 1.2
        R_low = 0.00001
        R_high = 10
        B_low = 0.0001
        B_high = 50
        Ilim_low = 2
        Ilim_high = 3000
    if model == "Linear":
        # R_el, R_ct, C_dl
        num_genes = 3  # Number of genes in each individual
        if search_space_dict['Linear'] == None:
            parameter_space = [{'low': R_low, 'high': R_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high}]  # Range of possible values for each gene
        else:
            parameter_space = search_space_dict['Linear']

    if model == "RQ":
        # R_el, R1, Q1, alpha1
        if search_space_dict['RQ'] == None:
            parameter_space = [{'low': R_low, 'high': R_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high}]  # Range of possible values for each gene
        else:
            parameter_space = search_space_dict['RQ']

    if model == "DoubleRC":
        # R_el, R1, Q1, alpha1
        if search_space_dict['DoubleRC'] == None:
            parameter_space = [{'low': R_low, 'high': R_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high}]  # Range of possible values for each gene
        else:
            parameter_space = search_space_dict['DoubleRC']

    if model == "DoubleRQ":
        # R_el, R1, Q1, alpha1, R2, Q2, alpha2
        num_genes = 7  # Number of genes in each individual
        #starting_parameters = findStartingParameters(dataset, "IMP")

        if search_space_dict['DoubleRQ'] == None:
            # print("None and unchecked")

            parameter_space = [{'low': R_low, 'high': R_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high}]
        else:
            # print("Unchecked")
            parameter_space = search_space_dict['DoubleRQ']

        '''if search_space_dict['DoubleRQ'] == None and search_space_dict['FastSearch'] == False:
            # print("None and unchecked")

            parameter_space = [{'low': R_low, 'high': R_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high},
                               {'low': R_low, 'high': R_high},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high}]

        elif search_space_dict['DoubleRQ'] == None and search_space_dict['FastSearch'] == True:
            # print("None and checked")

            parameter_space = [{'low': 0.9 * starting_parameters[0], 'high': 1.1 * starting_parameters[0]},
                               {'low': 0.9 * starting_parameters[1], 'high': 1.1 * starting_parameters[1]},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high},
                               {'low': 0.9 * starting_parameters[4], 'high': 1.1 * starting_parameters[4]},
                               {'low': Q_low, 'high': Q_high},
                               {'low': alpha_low, 'high': alpha_high}]

        elif search_space_dict['DoubleRQ'] != None and search_space_dict['FastSearch'] == False:
            # print("Unchecked")
            parameter_space = search_space_dict['DoubleRQ']

        elif search_space_dict['DoubleRQ'] != None and search_space_dict['FastSearch'] == True:
            # print("Checked")

            parameter_space = [{'low': 0.9 * starting_parameters[0], 'high': 1.1 * starting_parameters[0]},
                               {'low': 0.9 * starting_parameters[1], 'high': 1.1 * starting_parameters[1]},
                               search_space_dict['DoubleRQ'][2],
                               search_space_dict['DoubleRQ'][3],
                               {'low': 0.9 * starting_parameters[4], 'high': 1.1 * starting_parameters[4]},
                               search_space_dict['DoubleRQ'][5],
                               search_space_dict['DoubleRQ'][6]]'''

    if model == "NotLimited":
        if search_space_dict['NotLimited'] == None:
            parameter_space = [{'low': In_low, 'high': In_high},
                               {'low': A_low, 'high': A_high},
                               {'low': I0_low, 'high': I0_high},
                               {'low': E0_low, 'high': E0_high},
                               {'low': R_low, 'high': R_high}]
        else:
            parameter_space = search_space_dict['NotLimited']

    if model == "Limited":
        if search_space_dict['Limited'] == None:
            parameter_space = [{'low': In_low, 'high': In_high},
                               {'low': A_low, 'high': A_high},
                               {'low': I0_low, 'high': I0_high},
                               {'low': E0_low, 'high': E0_high},
                               {'low': R_low, 'high': R_high},
                               {'low': B_low, 'high': B_high},
                               {'low': Ilim_low, 'high': Ilim_high}]
        else:
            parameter_space = search_space_dict['Limited']

    return parameter_space


def fit_bfgs(dataset, func, parameter_space, model, dataset_type, starting_parameters):

    #print("start", starting_parameters)
    fitness = 0
    if dataset_type == "IMP":
        f = dataset[0]
        R_exp = dataset[1]
        X_exp = dataset[2]

    elif dataset_type == "LOAD":
        current = dataset[0]
        eta = dataset[1]

    for i in range(1):
        if model == "Linear":
            del starting_parameters[3:]
            # starting_parameters[2] = random.uniform(0.001, 10)

        if model == "RQ":
            # R_el, R1, Q1, alpha1
            del starting_parameters[4:]
            '''starting_parameters[2] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            starting_parameters[3] = random.uniform(parameter_space[3]['low'], parameter_space[2]['high'])'''

        if model == "DoubleRC":
            # R_el, R1, C1, R2, C2
            del starting_parameters[5:]
            '''starting_parameters[2] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            starting_parameters[4] = random.uniform(parameter_space[4]['low'], parameter_space[4]['high'])'''

        if model == "DoubleRQ":
            '''starting_parameters[2] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            starting_parameters[3] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            starting_parameters[5] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])
            starting_parameters[6] = random.uniform(parameter_space[2]['low'], parameter_space[2]['high'])'''

        bounds = []
        for gene in parameter_space:
            bounds.append((gene['low'], gene['high']))

        if dataset_type == "IMP":
            # Without bounds
            '''[solution, best_residuals, gopt, Bopt, func_calls, grad_calls, warnflg] = fmin_bfgs(f=func,
                                                            x0=starting_parameters,
                                                            args=(f, R_exp, X_exp),
                                                            maxiter=2000,
                                                            disp=False,
                                                            full_output=True)'''
            # With bounds
            [solution, best_residuals, info] = fmin_l_bfgs_b(func=func,
                                                             x0=starting_parameters,
                                                             args=(f, R_exp, X_exp),
                                                             maxiter=2000,
                                                             approx_grad=True,
                                                             disp=False,
                                                             bounds=bounds)


        elif dataset_type == "LOAD":
            starting_parameters = []
            for gene in parameter_space:
                starting_parameters.append(random.uniform(gene['low'], gene['high']))
                # Without bounds
            '''[solution, best_residuals, gopt, Bopt, func_calls, grad_calls, warnflg] = fmin_bfgs(f=func,
                                                                                                x0=starting_parameters,
                                                                                                args=(current, eta),
                                                                                                maxiter=2000,
                                                                                                disp=False,
                                                                                                full_output=True)'''
            # With bounds
            [solution, best_residuals, info] = fmin_l_bfgs_b(func=func,
                                                             x0=starting_parameters,
                                                             args=(current, eta),
                                                             maxiter=2000,
                                                             approx_grad=True,
                                                             disp=False,
                                                             bounds=bounds)

        current_fitness = best_residuals

        if fitness == 0:
            fitness = current_fitness
            best_solution = solution
        if current_fitness < fitness:
            fitness = current_fitness
            best_solution = solution
            best_solution = solution

    return best_solution, 1 / fitness


def fit_lm(dataset, func, parameter_space, model, dataset_type, starting_parameters):

    fitness = 0
    if dataset_type == "IMP":
        f = dataset[0]
        R_exp = dataset[1]
        X_exp = dataset[2]

    elif dataset_type == "LOAD":
        current = dataset[0]
        eta = dataset[1]

    for i in range(1):
        if model == "Linear":
            del starting_parameters[3:]


        if model == "RQ":
            del starting_parameters[4:]


        if model == "DoubleRC":
            # R_el, R1, C1, R2, C2
            del starting_parameters[5:]

        if model == "DoubleRQ":
            pass

        '''lm doesn't handle bounds 
        bounds = []
        low_bounds = []
        high_bounds = []
        for gene in parameter_space:
            low_bounds.append(gene['low'])
            high_bounds.append(gene['high'])
        bounds.append(low_bounds)
        bounds.append(high_bounds)'''

        if dataset_type == "IMP":
            result = least_squares(fun=func,
                                   x0=starting_parameters,
                                   args=(f, R_exp, X_exp),
                                   method='lm')
        if dataset_type == "LOAD":
            starting_parameters = []
            for gene in parameter_space:
                starting_parameters.append(random.uniform(gene['low'], gene['high']))
            result = least_squares(fun=func,
                                   x0=starting_parameters,
                                   args=(current, eta),
                                   method='lm')
        solution = result.x
        residuals = result.fun
        current_fitness = np.sum(residuals) / len(residuals)

        if fitness == 0:
            fitness = current_fitness
            best_solution = solution
        if current_fitness < fitness:
            fitness = current_fitness
            best_solution = solution
    return best_solution, 1 / fitness


# Impedance fitting
class ImpedanceFitting:
    def __init__(self):
        pass

    def calculate_fit_imp(self, datasets, search_space_dict, experiment_dict, mode, model,
                          optimization_algorithm, num_generations, sol_per_pop, parent_selection_type,
                          percent_parents_mating,
                          crossover_type, crossover_probability, mutation_type, mutation_probability, keep_elitism,
                          fitness_obj, IsInitialPop, fitness_coeff, fitness_function_type):

        self.datasets = datasets
        self.fitness_coeff = fitness_coeff
        self.fitness_function_type = fitness_function_type  # Can be either "normal" or "numpy" for faster calculation
        dataset_type = "IMP"

        # Items to be returned :
        fitting_datasets = []
        fitting_parameters = []
        new_dict = {}
        time_list = []

        if model == "Linear":
            fitting_parameters.append(
                ["Experiment", "R0 (Ohm)", "R1 (Ohm)", "C1 (F)"])
        if model == "RQ":
            fitting_parameters.append(
                ["Experiment", "R0 (Ohm)", "R1 (Ohm)", "Q1 (F.s^(alpha1-1)", "alpha1"])
        if model == "DoubleRC":
            fitting_parameters.append(["Experiment", "Ohmic resistance (Ohm)",
                                       "R1 (Ohm)", "C1 (F)",
                                       "R2 (Ohm)", "C2 (F)"])
        if model == "DoubleRQ":
            fitting_parameters.append(["Experiment", "Ohmic resistance (Ohm)",
                                       "R1 (Ohm)", "Q1 (F.s^(alpha1-1)", "alpha1",
                                       "R2 (Ohm)", "Q2 (F.s^(alpha2-1)", "alpha2"])

        if model == "Fouquet":
            fitting_parameters.append(["Experiment", "Ohmic resistance (Ohm)", "Charge transfer resistance (Ohm)",
                                       "Polarization resistance (Ohm)", "Capacity (F)", "alpha",
                                       "Diffusion resistance (Ohm)", "time constant (s)"])

        if mode == "auto":
            # If the files are selected using the auto mode, the for loop has to browse both the experiment number and the currrent value
            i = 0
            for exp_num, current_labels in experiment_dict.items():
                new_dict[exp_num] = {}
                for current_label in current_labels:
                    start_time = time.perf_counter()  # Get the start time
                    parameter_space = makeParameterSpace(self.datasets[i], search_space_dict, model)
                    if search_space_dict['FastSearch'] == True:
                        starting_parameters = findStartingParameters(self.datasets[i], dataset_type, model, parameter_space, "other")
                    else:
                        starting_parameters = randomStartingParameters(dataset_type, model, parameter_space)
                        print("starting", starting_parameters)
                    if optimization_algorithm == "GA":
                        solution, solution_fitness = self.fit_impedance_ga(self.datasets[i], parameter_space, starting_parameters, model,
                                                                           num_generations, sol_per_pop,
                                                                           parent_selection_type,
                                                                           percent_parents_mating,
                                                                           crossover_type, crossover_probability,
                                                                           mutation_type, mutation_probability,
                                                                           keep_elitism,
                                                                           fitness_obj, IsInitialPop)
                    elif optimization_algorithm == "bfgs":
                        solution, solution_fitness = fit_bfgs(self.datasets[i], self.residuals_impedance,
                                                              parameter_space, model, dataset_type, starting_parameters)

                    elif optimization_algorithm == "lm":
                        solution, solution_fitness = fit_lm(self.datasets[i], self.residuals_impedance_array,
                                                            parameter_space, model, dataset_type, starting_parameters)

                    end_time = time.perf_counter()  # Get the end time
                    elapsed_time = end_time - start_time  # Calculate the time needed by the optimization algorithm

                    # Calculated values are added to the list of parameters
                    appendFittingParameters(fitting_parameters, model, solution, exp_num)

                    # Calculate the simulated dataset using numpy
                    freqs = np.array(self.datasets[i][0])
                    R_num, X_num = self.calculate_R_X_numpy(solution, freqs, model)

                    # Make a global dataset for all the fitting datasets
                    fitting_datasets.append((freqs, R_num, X_num))
                    # Make a dict containing the experiment number, the current value and the fitness value
                    new_dict[exp_num][current_label] = 1 / solution_fitness
                    # Make a list containing the time needed by the optimization algorithm
                    time_list.append(elapsed_time)
                    i += 1

        if mode == "manual":
            # If the files are sected manually, then the experiment number is the name of the file
            for dataset, exp_num in zip(self.datasets, experiment_dict.keys()):

                start_time = time.perf_counter()  # Get the start time
                parameter_space = makeParameterSpace(dataset, search_space_dict, model)
                if search_space_dict['FastSearch'] == True:
                    starting_parameters = findStartingParameters(dataset, dataset_type, model, parameter_space, "other")
                else:
                    starting_parameters = randomStartingParameters(dataset_type, model, parameter_space)
                    #print("starting", starting_parameters)
                if optimization_algorithm == "GA":
                    solution, solution_fitness = self.fit_impedance_ga(dataset, parameter_space, starting_parameters, model, num_generations,
                                                                       sol_per_pop,
                                                                       parent_selection_type, percent_parents_mating,
                                                                       crossover_type, crossover_probability,
                                                                       mutation_type, mutation_probability,
                                                                       keep_elitism,
                                                                       fitness_obj, IsInitialPop)
                elif optimization_algorithm == "bfgs":
                    solution, solution_fitness = fit_bfgs(dataset, self.residuals_impedance, parameter_space, model,
                                                          dataset_type, starting_parameters)

                elif optimization_algorithm == "lm":
                    solution, solution_fitness = fit_lm(dataset, self.residuals_impedance_array, parameter_space, model,
                                                        dataset_type, starting_parameters)

                end_time = time.perf_counter()  # Get the end time
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                #print("solu", solution)
                #print("time", elapsed_time)
                #print(fitting_parameters)
                appendFittingParameters(fitting_parameters, model, solution, exp_num)

                # Calculate the simulated dataset using numpy
                freqs = np.array(dataset[0])
                R_num, X_num = self.calculate_R_X_numpy(solution, freqs, model)

                # Make a global dataset for all the fitting datasets
                fitting_datasets.append((freqs, R_num, X_num))
                new_dict[exp_num] = 1 / solution_fitness
                time_list.append(elapsed_time)

            '''print("exp number", exp_num)
            print("solution fitness:", 1 / solution_fitness)
            print("solution:", solution)
            print("time", elapsed_time)'''

        return fitting_datasets, fitting_parameters, new_dict, time_list

    def fit_impedance_ga(self, dataset, gene_space, starting_parameters, model, num_generations, sol_per_pop, parent_selection_type,
                         percent_parents_mating, crossover_type, crossover_probability, mutation_type,
                         mutation_probability, keep_elitism, fitness_obj, IsInitialPop):
        # The current dataset used for the fitting has be put as a class attribute to be used in the fitness function in order to work because of how PyGad is coded
        self.current_dataset = dataset
        self.model = model
        num_genes = 0



        if model == "Linear":
            # R_el, R_ct, C_dl
            num_genes = 3  # Number of genes in each individual

        if model == "RQ":
            # R_el, R1, Q1, alpha1
            num_genes = 4  # Number of genes in each individual

        if model == "DoubleRC":
            # R_el, R1, C1, R2, C2
            num_genes = 5  # Number of genes in each individual

        if model == "DoubleRQ":
            # R_el, R1, Q1, alpha1, R2, Q2, alpha2
            num_genes = 7  # Number of genes in each individual

        # The initial sets of parameters can be modified using the function initializePopulation

        if IsInitialPop:
            initial_population = initializePopulation(starting_parameters, gene_space, sol_per_pop)
        else:
            initial_population = None

        # To be used by PyGad, the fitness objective has to be rewritten as reach_0.01 for example
        if fitness_obj != 0:
            fitness_str = "reach_" + str(1 / fitness_obj)
        else:
            fitness_str = None

        if keep_elitism == None:
            keep_elitism = 1
        else:
            # keep_elitism=round(keep_elitism * sol_per_pop)
            keep_elitism = keep_elitism

        # Can be used to observe the whole population evolution at each generation
        def callback_generation(ga_instance):
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.scatter(dataset[1], dataset[2])
            # best_solution = ga_instance.best_solution()[0]
            # best_solution_fitness = ga_instance.best_solution()[1]
            for solution in ga_instance.population:
                R_el, R1, Q1, alpha1, R2, Q2, alpha2 = solution

                freqs = np.array(self.current_dataset[0])
                R_num, X_num = self.calculate_R_X_numpy(solution, freqs, model)
                ax.plot(R_num, X_num,
                        label=f"R_el={R_el:.5f}, R1={R1:.5f}, Q1={Q1:.2f}, alpha1={alpha1:.2f}, R2={R2:.5f}Q2={Q2:.2f}, alpha2={alpha2:.2f}")

            # ax.legend()
            plt.draw()
            plt.pause(1)
            plt.close()

        # Create the GA instance
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=round(percent_parents_mating * sol_per_pop),
            fitness_func=self.fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            initial_population=initial_population,
            parent_selection_type=parent_selection_type,
            crossover_probability=crossover_probability,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            keep_elitism=keep_elitism,
            mutation_probability=mutation_probability,
            stop_criteria=fitness_str,
            # on_generation=callback_generation
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Get the results
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        return solution, solution_fitness

    def fitness_function(self, ga_instance, solution, solution_idx):
        if self.fitness_function_type == "normal":
            # A fitness function evaluation using a for loop is implemented to show how much numpy reduces the calculation time
            S = 0

            R_max = max(self.current_dataset[1])
            X_max = max(self.current_dataset[2])

            for k in range(round(len(self.current_dataset[0]) / self.fitness_coeff)):
                k *= self.fitness_coeff
                f = self.current_dataset[0][k]
                R_num, X_num = self.calculate_R_X(solution, f)

                R_exp = self.current_dataset[1][k]
                X_exp = self.current_dataset[2][k]

                # S += math.sqrt(((R_num-R_exp)/R_exp)**2+((X_num-X_exp)/X_exp)**2)
                S += math.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2)

            S = S / len(self.current_dataset[0]) * self.fitness_coeff
            fitness = 1 / (np.abs(S))
            return fitness

        if self.fitness_function_type == "numpy":
            # Here the fitness function is calculated using numpy, enabling lower computation time

            freqs = np.array(self.current_dataset[0])
            x = self.current_dataset[1]
            y = self.current_dataset[2]

            R_num, X_num = self.calculate_R_X_numpy(solution, freqs, self.model)

            R_max = np.max(x)
            X_max = np.max(y)

            R_exp = np.array(x)
            X_exp = np.array(y)
            residuals = np.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2)
            S = np.sum(residuals) / len(self.current_dataset[0])

            fitness = 1 / (np.abs(S))
            return fitness

        '''
        # Other fitness functions : 
        #First idea : Bad because R_exp and X_exp get very big since dividing by a greater value is more efficient to diminish F_obj than subtracting by a smaller value
        #It's not really the case since it's X_exp and not X_num so the problem comes from somewhere else

        for k in range(len(self.current_dataset[0])):
            f = self.current_dataset[0][k]
            R_num, X_num = self.calculate_R_X(solution, f)
            R_exp, X_exp = self.current_dataset[1][k], self.current_dataset[2][k]

            S += abs((R_num - R_exp)) / R_exp + abs((X_num - X_exp)) / X_exp
        '''
        # Bad one for the same reasons
        '''
        for k in range(len(self.current_dataset[0])):
            f = self.current_dataset[0][k]
            R_num, X_num = self.calculate_R_X(solution, f)
            R_exp, X_exp = self.current_dataset[1][k], self.current_dataset[2][k]


            S += math.sqrt(((R_num - R_exp)/R_exp) ** 2 + ((X_num - X_exp)/X_exp) ** 2)
        '''

        # Ok but it is better to normalize to be able to compare fitness values from a dataset to another
        '''
        for k in range(len(self.current_dataset[0])):
            f = self.current_dataset[0][k]
            R_num, X_num = self.calculate_R_X(solution, f)
            R_exp, X_exp = self.current_dataset[1][k], self.current_dataset[2][k]

            S += abs(R_num - R_exp) + abs(X_num - X_exp)
        '''

    def calculate_R_X(self, solution, f):
        # Returns the real and imaginary parts calculated at one frequency
        if self.model == "Linear":
            R_el = solution[0]
            R_ct = solution[1]
            C_dl = solution[2]
            Z = R_el + R_ct / (1 + 2j * R_ct * C_dl * 3.14 * f)

        if self.model == "RQ":
            R_el = solution[0]
            R_ct = solution[1]
            Q = solution[2]
            alpha = solution[3]
            Z = R_el + R_ct / (1 + R_ct * Q * (2j * 3.14 * f) ** alpha)

        if self.model == "DoubleRC":
            R_el = solution[0]
            R1 = solution[1]
            C1 = solution[2]
            R2 = solution[3]
            C2 = solution[4]

            Z = R_el + R1 / (1 + R1 * C1 * 1j * 2 * 3.14 * f) + R2 / (1 + R2 * C2 * 1j * 2 * 3.14 * f)

        if self.model == "DoubleRQ":
            R_el = solution[0]
            R1 = solution[1]
            Q1 = solution[2]
            alpha1 = solution[3]
            R2 = solution[4]
            Q2 = solution[5]
            alpha2 = solution[6]
            omega1 = np.power(1j * 2 * 3.14 * f, alpha1)
            omega2 = np.power(1j * 2 * 3.14 * f, alpha2)
            Z = R_el + R1 / (1 + R1 * Q1 * omega1) + R2 / (
                        1 + R2 * Q2 * omega2)

        if self.model == "Fouquet":
            R_el = solution[0]
            R_ct = solution[1]
            Q = solution[2]
            alpha = solution[3]
            R_d = solution[4]
            tau_d = solution[5]
            w = 2 * 3.14 * f

            Z_CPE = 1 / (Q * (1j * w) ** alpha)
            Z_W = R_d * np.tanh(np.sqrt(1j * w * tau_d)) / np.sqrt(1j * w * tau_d)

            Z = R_el + Z_CPE * (R_ct + Z_W) / (Z_CPE + R_ct + Z_W)

        if self.model == "FiniteDiffusion":
            R_el = solution[0]
            R_d = solution[1]
            tau_d = solution[2]
            w = 2 * 3.14 * f
            Z_W = R_d * np.tanh(np.sqrt(1j * w * tau_d)) / np.sqrt(1j * w * tau_d)

            Z = R_el + Z_W

        R_num = Z.real
        X_num = -Z.imag

        return R_num, X_num

    def calculate_R_X_numpy(self, solution, freqs, model):
        # Return two numpy vectors containing the real and imaginary parts calculated at every frequency
        if model == "Linear":
            R_el, R_ct, C_dl = solution

            Z = (R_el +
                 R_ct / (1 + R_ct * C_dl * (1j * 2 * np.pi * freqs)))

        if model == "RQ":
            R_el, R1, Q1, alpha1 = solution

            Z = (R_el +
                 R1 / (1 + R1 * Q1 * (1j * 2 * np.pi * freqs) ** alpha1))

        if model == "DoubleRC":
            R_el, R1, C1, R2, C2 = solution

            Z = (R_el +
                 R1 / (1 + R1 * C1 * 1j * 2 * np.pi * freqs) +
                 R2 / (1 + R2 * C2 * 1j * 2 * np.pi * freqs))

        if model == "DoubleRQ":
            R_el, R1, Q1, alpha1, R2, Q2, alpha2 = solution

            Z = (R_el +
                 R1 / (1 + R1 * Q1 * (1j * 2 * np.pi * freqs) ** alpha1) +
                 R2 / (1 + R2 * Q2 * (1j * 2 * np.pi * freqs) ** alpha2))

        R_num = Z.real
        X_num = -Z.imag

        return R_num, X_num

    def residuals_impedance(self, solution, x, y, z):
        if self.fitness_function_type == "normal":
            residuals = 0
            R_max = max(y)
            X_max = max(z)

            for i in range(len(x)):
                f = x[i]
                #print(solution)
                R_num, X_num = self.calculate_R_X(solution, f)

                R_exp = y[i]
                X_exp = z[i]
                residuals += math.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2)

            residuals /= len(x)
            return residuals
        if self.fitness_function_type == "numpy":
            freqs = np.array(x)
            R_exp = np.array(y)
            X_exp = np.array(z)

            R_num, X_num = self.calculate_R_X_numpy(solution, freqs, self.model)

            R_max = np.max(y)
            X_max = np.max(z)

            residuals = np.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2)
            residuals = np.sum(residuals) / len(x)
            return residuals

    def residuals_impedance_array(self, solution, x, y, z):
        if self.fitness_function_type == "normal":
            residuals = []
            R_max = max(y)
            X_max = max(z)

            for i in range(len(x)):
                f = x[i]
                #print(solution)
                R_num, X_num = self.calculate_R_X(solution, f)

                R_exp = y[i]
                X_exp = z[i]
                residuals.append(math.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2))

            return residuals
        if self.fitness_function_type == "numpy":
            freqs = np.array(x)
            R_exp = np.array(y)
            X_exp = np.array(z)

            R_num, X_num = self.calculate_R_X_numpy(solution, freqs, self.model)

            R_max = np.max(y)
            X_max = np.max(z)

            residuals = np.sqrt(((R_num - R_exp) / R_max) ** 2 + ((X_num - X_exp) / X_max) ** 2)
            # residuals = np.sqrt(((R_num - R_exp)/(R_exp**2 + X_exp**2)) ** 2 + ((X_num - X_exp)/(R_exp**2 + X_exp**2)) ** 2)

            return residuals


### LOAD
class LOADFitting:
    def __init__(self):
        pass

    def calculate_fit_load(self, datasets, search_space_dict, experiment_dict, mode, model,
                           optimization_algorithm, num_generations, sol_per_pop, parent_selection_type,
                           percent_parents_mating,
                           crossover_type, crossover_probability, mutation_type, mutation_probability, keep_elitism,
                           fitness_obj, IsInitialPop, fitness_coeff, fitness_function_type):

        self.datasets = datasets
        self.range = 20
        self.fitting_point_number = 20
        dataset_type = "LOAD"

        # Items to be returned
        fitting_datasets = []
        fitting_parameters = []
        new_dict = {}
        time_list = []

        if model == "NotLimited":
            fitting_parameters.append(
                ["Experiment", "In (A/cm^2)", "a (V/dec)", "I0 (A/cm^2)", "E0 (V)", "R (Ohm.cm^2)"])

        if model == "Limited":
            fitting_parameters.append(
                # Not dec if log (log=natural logarithm, base e and log10=logarithm base 10)
                ["Experiment", "In (A/cm^2)", "a (V/dec)", "I0 (A/cm^2)", "E0 (V)", "R (Ohm.cm^2)", "b (V/dec)",
                 "Ilim (A/cm^2)"])

        for dataset, exp_num in zip(self.datasets, experiment_dict.keys()):
            self.current_dataset = dataset
            start_time = time.perf_counter()  # Get the start time
            parameter_space = makeParameterSpace(dataset, search_space_dict, model)
            starting_parameters = []
            if optimization_algorithm == "GA":
                solution, solution_fitness = self.fit_load_ga(dataset, model, num_generations, sol_per_pop,
                                                              parent_selection_type, percent_parents_mating,
                                                              crossover_type, crossover_probability,
                                                              mutation_type, mutation_probability, keep_elitism,
                                                              fitness_obj, IsInitialPop, fitness_function_type)

            elif optimization_algorithm == "bfgs":
                solution, solution_fitness = fit_bfgs(dataset, self.residuals_load, parameter_space, model,
                                                      dataset_type, starting_parameters)

            elif optimization_algorithm == "lm":
                solution, solution_fitness = fit_lm(dataset, self.residuals_load_array,
                                                     parameter_space, model, dataset_type, starting_parameters)

            end_time = time.perf_counter()  # Get the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time

            appendFittingParameters(fitting_parameters, model, solution, exp_num)

            current = np.array(dataset[0])
            fitting_point_spacement = int(len(current) / self.fitting_point_number)
            current = current[::fitting_point_spacement]

            eta = self.calculate_eta_numpy(solution, current, model)

            # Make a global dataset for all the fitting datasets
            fitting_datasets.append((current, eta))
            new_dict[exp_num] = 1 / solution_fitness
            time_list.append(elapsed_time)

            '''print("solution fitness:", 1 / solution_fitness)
            print("solution:", parameters)
            print("time", elapsed_time)'''

        return fitting_datasets, fitting_parameters, new_dict, time_list

    def fit_load_ga(self, dataset, model, num_generations, sol_per_pop, parent_selection_type, percent_parents_mating,
                    crossover_type, crossover_probability, mutation_type, mutation_probability, keep_elitism,
                    fitness_obj, IsInitialPop, fitness_function_type):
        self.current_dataset = dataset
        self.model = model

        # Range of possible values for each gene in this order :
        num_genes = 0
        if model == "NotLimited":
            # In, A, I0, E0, R
            gene_space = [{'low': 0.00001, 'high': 0.1}, {'low': 0.000001, 'high': 2},
                          {'low': 0.0000001, 'high': 50}, {'low': 0.1, 'high': 1.2},
                          {'low': 0.00001, 'high': 10}]  # Range of possible values for each gene
            num_genes = 5  # Number of genes in each individual

        if model == "Limited":
            # In, A, I0, E0, R, B, Ilim
            gene_space = [{'low': 0.00001, 'high': 0.1}, {'low': 0.000001, 'high': 2},
                          {'low': 0.0000001, 'high': 50}, {'low': 0.1, 'high': 1.2},
                          {'low': 0.00001, 'high': 10}, {'low': 0.0001, 'high': 50},
                          {'low': 2, 'high': 3000}]  # Range of possible values for each gene
            num_genes = 7  # Number of genes in each individual

        if fitness_obj != 0:
            fitness_str = "reach_" + str(1 / fitness_obj)
        else:
            fitness_str = None

        if keep_elitism == None:
            keep_elitism = 1
        else:
            keep_elitism = keep_elitism

        # Create the GA instance
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=round(percent_parents_mating * sol_per_pop),
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            # initial_population=initial_population,
            parent_selection_type=parent_selection_type,
            crossover_probability=crossover_probability,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            keep_elitism=keep_elitism,
            mutation_probability=mutation_probability,
            stop_criteria=fitness_str,
        )

        # Run the GA
        ga_instance.run()

        # Get the results
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        solution_list = list(solution)
        return solution_list, solution_fitness

    def fitness_func(self, ga_instance, solution, solution_idx):
        '''S = 0
        fitting_point_spacement = int(len(self.current_dataset[0])/self.fitting_point_number)

        #Way faster because the for loop has less data to go through
        for k in range(self.fitting_point_number):
            if k==0 or k==1:
                k *= fitting_point_spacement
                eta = self.calculate_eta(self.current_dataset[0][k], solution, self.model)
                S += 100*abs((self.current_dataset[1][k] - eta) / self.current_dataset[1][k])

            else:
                k *= fitting_point_spacement
                eta = self.calculate_eta(self.current_dataset[0][k], solution, self.model)
                S += abs((self.current_dataset[1][k] - eta) / self.current_dataset[1][k])'''

        fitting_point_spacement = int(len(self.current_dataset[0]) / self.fitting_point_number)
        current = np.array(self.current_dataset[0])
        current = current[::fitting_point_spacement]
        eta_exp = np.array(self.current_dataset[1])
        eta_exp = eta_exp[::fitting_point_spacement]
        eta_num = self.calculate_eta_numpy(solution, current, self.model)
        residuals = np.abs((eta_exp - eta_num) / eta_exp)
        residuals[:2] *= 100
        S = np.sum(residuals) / len(eta_exp)

        '''
        for k in range(len(self.current_dataset[0])):
            if k == 0 or k == fitting_point_spacement:
                eta = self.calculate_eta(self.current_dataset[0][k], solution)
                S += 100*abs((self.current_dataset[1][k] - eta) / self.current_dataset[1][k])
            elif k % fitting_point_spacement == 0:
                eta = self.calculate_eta(self.current_dataset[0][k], solution)
                S += abs((self.current_dataset[1][k] - eta)/self.current_dataset[1][k])
        '''

        fitness = 1 / (np.abs(S))

        return fitness

    def residuals_load(self, solution, x, y):
        fitting_point_spacement = int(len(self.current_dataset[0]) / self.fitting_point_number)
        current = np.array(x)
        current = current[::fitting_point_spacement]
        eta_exp = np.array(y)
        eta_exp = eta_exp[::fitting_point_spacement]
        eta_num = self.calculate_eta_numpy(solution, current, self.model)
        residuals = np.abs((eta_exp - eta_num) / eta_exp)
        residuals[:2] *= 100
        residuals = np.sum(residuals) / len(eta_exp)
        return residuals

    def residuals_load_array(self, solution, x, y):
        fitting_point_spacement = int(len(self.current_dataset[0]) / self.fitting_point_number)
        current = np.array(x)
        current = current[::fitting_point_spacement]
        eta_exp = np.array(y)
        eta_exp = eta_exp[::fitting_point_spacement]
        eta_num = self.calculate_eta_numpy(solution, current, self.model)
        residuals = np.abs((eta_exp - eta_num) / eta_exp)
        residuals[:2] *= 100

        return residuals

    def calculate_eta_numpy(self, parameters, current, model):
        if model == "NotLimited":
            In, A, I0, E0, R = parameters
            return E0 - A * np.log((abs(current) + In) / I0) - R * (current + In)

        if model == "Limited":
            In, A, I0, E0, R, B, Ilim = parameters
            if np.any(abs(current) + In) > Ilim:
                return 100
            else:
                return E0 - A * np.log((abs(current) + In) / I0) - R * (current + In) + B * np.log(
                    1 - (abs(current) + In) / Ilim)

    def calculate_eta(self, parameters, current, model):
        if model == "NotLimited":
            In, A, I0, E0, R = parameters
            return E0 - A * math.log((abs(current) + In) / I0) - R * (current + In)

        if model == "Limited":
            In, A, I0, E0, R, B, Ilim = parameters
            if abs(current) + In > Ilim:
                return 100
            else:
                return E0 - A * math.log((abs(current) + In) / I0) - R * (current + In) + B * math.log(
                    1 - (abs(current) + In) / Ilim)

