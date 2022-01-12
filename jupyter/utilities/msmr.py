# Imports and Setup

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.optimize import fmin_slsqp
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as sf
from scipy.stats import rv_histogram

def load_experiment_data(filepath, constant_current, timestep, sf_window_length, interpolated_voltage_range, charge=True):

    """
    Loads low-current experimental data and calculates the differential capacity and differential 
    voltage using a Savitzy-Golay filter

    Parameters:

    filepath: (str) Filepath to the slow-scan data experimental data
    constant_current: (float) The constant-current value in which the experiment is using in Amps
    timestep: (float or int) Time step in between data collection in seconds
    sf_window_length: (int) Window length for the Savitzy Golay Filter (must be an odd number)
    interpolated_voltage_range: (tuple) Interpolated voltage range to be selected for optimizing
    charge: (boolean) default = True. If False, the voltage and currents will be reversed so that
    it looks similarly to the capacity-voltage profile of the charge step.

    Returns:

    voltage: Cell voltage
    current: Cell current
    capacity: Cell capacity
    dudq: Differential voltage
    data_cap_interp: Interpolated cell capacity that corresponds with the interpolating voltage window
    data_dudq_interp: Interpolated differential voltage that corresponds with the interpolating voltage window

    """

    data = pd.read_csv(filepath)
    capacity = np.array(data['Capacity(Ah)'])
    time = np.array(data['StepTime(s)'])
    if charge == True:
        voltage = np.array(data['Voltage(V)'])
        current = np.array(data['Current(A)'])
    elif charge == False:
        voltage = np.flip(np.array(data['Voltage(V)']))
        current = np.flip(np.array(data['Current(A)']))
    
    # Calculating the derivative using a Savitzy-Golay Filter
    dudt = sf(voltage, window_length = sf_window_length, deriv = 1, delta = timestep, polyorder = 3) 
    dudq = dudt / (constant_current/3600)
    
    # Interpolation
    f_data_cap_interp = interp1d(voltage, capacity)
    f_data_dudq_interp = interp1d(voltage, dudq)
    
    data_cap_interp = f_data_cap_interp(interpolated_voltage_range)
    data_dudq_interp = f_data_dudq_interp(interpolated_voltage_range)
    
    return voltage, current, capacity, dudq, data_cap_interp, data_dudq_interp

def individual_reactions(U, U0, Xj, w, T):
    """
    Solves the individual reaction within an electrode. The Xj parameter can be substituted by
    a capacity parameter (Qj) instead, and can be rescaled back to the thermodynamic factor by
    that same capacity factor.

    Parameters:

    U: (float, 1-D array) Potential to be calculated
    U0: (float) Standard electrode potential for reaction j
    Xj: (float) Maximum fractional occupancy for reaction j (intensive) or maximum capacity of reaction j (extensive)
    w: (float) Thermodynamic factor for reaction j
    T: (float) temperature

    Returns:
    xj: (1-D array) Fractional occupancy (or capacity if extensive)
    dxjdu: (1-D array) Differential capacity


    """
    R, F = 8.314, 96485 # Gas constant, Faraday's constant
    f = F/(R*T)
    xj = Xj/(1 + np.exp(f*(U-U0)/w))
    
    try:
        dxjdu = (-Xj/w)*((f*np.exp(f*(U-U0)/w))/(1+np.exp(f*(U-U0)/w))**2)
    except OverflowError: 
        dxjdu = 0 # Approximates the value as zero in the case that an Overflow Error occurs
    
    return xj, dxjdu

def electrode_response(parameter_matrix, T, min_volt, max_volt, number_of_rxns, points=1000):
    """
    Wraps the individual solver and creates the cumulative OCV and dx/dU curve or dQ/dU. 
    The parameter matrix holds the Uo, Xj or Qj, and wj term for each of the 
    individual reactions, respectively.

    parameter_matrix: (1-D Array) Array of parameters for the MSMR model, that follow the repeating order,  
    standard electrode potential (U), lithium content or capacity (Xj or Qj), and thermodynamic factor
    (omega). parameter_matrix should be equal to the number_of_rxns * 3.
    T: (float) Temperature
    min_volt: (float) Minimum voltage for the MSMR model to solve
    max_volt: (float) Maximum voltage for the MSMR model to solve
    number_of_rxns: (int) Number of reactions
    points: (int) Number of points per volt

    """
    
    # Initialize the matrix with the first entry
    voltage = np.linspace(min_volt, max_volt, int((max_volt-min_volt)*points)+1)
    host_xj, host_dxjdu = individual_reactions(U=voltage, 
                                               U0=parameter_matrix[0], 
                                               Xj=parameter_matrix[1], 
                                               w=parameter_matrix[2], 
                                               T=T)
    
    # Add additional rows into the matrix for each separate reaction
    for i in range(1, number_of_rxns):
        row = int(i)
        xj_n, dxjdu_n = individual_reactions(U=voltage, 
                                             U0=parameter_matrix[int(0+i*3)], 
                                             Xj=parameter_matrix[int(1+i*3)], 
                                             w=parameter_matrix[int(2+i*3)], 
                                             T=T)

        host_xj = np.vstack((host_xj, xj_n))
        host_dxjdu = np.vstack((host_dxjdu, dxjdu_n))
    
    host_xj_sum = np.sum(host_xj, axis = 0)
    host_dxjdu_sum = np.sum(host_dxjdu, axis = 0)
    
    return voltage, host_xj_sum, host_dxjdu_sum

def whole_cell(parameter_matrix,
               temp, nor_pos, nor_neg, 
               pos_volt_range, neg_volt_range,
               pos_lower_li_limit, neg_lower_li_limit, 
               n_p, p_capacity, usable_cap, Qj_or_Xj,
               all_output = False):
    
    """ 
    Uses the MSMR model to generate a whole cell response and yields capacity, voltage, and differential
    voltage data. If prompted, this will also output results for the two individual electrodes in conjunction
    with the whole-cell response.

    Parameters

    parameter_matrix: (N,) A 1-D array of all parameter values of Uj, Qj or Xj, and wj for both electrodes
    temp: (int or float) Temperature
    nor_pos: (int) Number of reactions in the positive electrode. Helps determine how many parameters in
             the parameter_matrix are those of the positive electrde.
    nor_neg: (int) Number of reactions in the negative electrode. Helps determine how many parameters in
             the parameter_matrix are those of the negative electrde.
    pos_volt_range: (tuple) The voltage range to calculate the MSMR results for the positive electrode.
    neg_volt_range: (tuple) The voltage range to calculate the MSMR results for the negative electrode.
    pos_lower_li_limit: (float) The lower capacity bound of the positive electrode (assuming partial utilization)
    neg_lower_li_limit: (float) The lower capacity bound of the negative electrode (assuming partial utilization)
    n_p: (float) The ratio of the negative electrode capacity to the positive electrode capacity in whole cells.
         Only necessary if Qj_or_Xj == Xj
    p_capacity: (float) Positive electrode capacity. Only necessary if Qj_or_Xj == Xj
    usable_cap: (float) Usable capacity, rated capacity, or capacity available within a certain voltage window.
    Qj_or_Xj: (str) Must be "Qj" or "Xj" and determines if the MSMR model is computed with intensive (Xj) or
              extensive (Qj) properties.
    all_output: (boolean) If True, returns capacity, V, and dV/dQ for whole cell and both electrodes. If False,
                returns capacity, V, and dV/dQ for just whole cell response.

    Returns

    capacity_range: Calculated capacity values (within the usable capacity) that correspond with the following outputs 
    whole_cell_volt: Calculated voltage values
    whole_cell_dqdu: Calculated differential capacity values
    whole_cell_dudq: Calculated differential voltage values

    p_capacity_range: Capacity range that the positive electrode is operating through
    n_capacity_range: Capacity range that the negative electrode is operating through
    pos_volt_interp: Positive Electrode Voltages
    neg_volt_interp: Negative Electrode Voltages
    pos_dqdu_interp: Positive Electrode Differential Capacity
    neg_dqdu_interp: Negative Electrode Differential Capacity

    """

    int_points = 500
    
    pos_matrix = parameter_matrix[0:3*nor_pos]
    neg_matrix = parameter_matrix[3*nor_pos:]
    
    # Unpacking Variables
    p_min_volt, p_max_volt = pos_volt_range
    n_min_volt, n_max_volt = neg_volt_range
    
    if Qj_or_Xj == 'Xj':
        # Generating the individual electrode responses, where p = positive
        pv, px, pdxdu = electrode_response(pos_matrix, temp, p_min_volt, p_max_volt, nor_pos)
        nv, nx, ndxdu = electrode_response(neg_matrix, temp, n_min_volt, n_max_volt, nor_neg)
        
        # Applying N/P ratio to convert to the usable range of the electrodes into a nominal capacity
        pq = (px)*p_capacity
        nq = (nx)*p_capacity*n_p
        
        # Converting from dxdu to dQdu (for purposes of comparing with real data)
        pdqdu = pdxdu*p_capacity
        ndqdu = ndxdu*p_capacity*n_p

        # Interpolating capacities for proper Coulomb counting between p and n electrodes. 
        # Takes the lower ends of the two ranges, and adds usable capacities to get the same Q on both sides
        p_capacity_range = np.linspace(pos_lower_li_limit*p_capacity, (pos_lower_li_limit*p_capacity)+usable_cap, int_points)
        n_capacity_range = np.linspace(neg_lower_li_limit*p_capacity*n_p, (neg_lower_li_limit*p_capacity*n_p)+usable_cap, int_points)
        capacity_range = p_capacity_range - p_capacity_range.min()

    elif Qj_or_Xj == 'Qj':
        # Generating the individual electrode responses, where p = positive
        pv, pq, pdqdu = electrode_response(pos_matrix, temp, p_min_volt, p_max_volt, nor_pos)
        nv, nq, ndqdu = electrode_response(neg_matrix, temp, n_min_volt, n_max_volt, nor_neg)
        
        # Interpolating capacities for proper Coulomb counting between p and n electrodes. 
        # Takes the lower ends of the two ranges, and adds usable capacities to get the same Q on both sides
        p_capacity_range = np.linspace(pos_lower_li_limit, pos_lower_li_limit+usable_cap, int_points)
        n_capacity_range = np.linspace(neg_lower_li_limit, neg_lower_li_limit+usable_cap, int_points)
        capacity_range = p_capacity_range - p_capacity_range.min()

    else:
        raise ValueError('Missing input for Qj_or_Xj')
    
    # Interpolating the positive and negative electrode data to ensure the capacity data are evenly spaced
    f_pos_cap_interp = interp1d(pq, pv, fill_value='extrapolate')
    f_pos_dx_interp = interp1d(pq, pdqdu, fill_value='extrapolate')
    f_neg_cap_interp = interp1d(nq, nv, fill_value='extrapolate')
    f_neg_dx_interp = interp1d(nq, ndqdu, fill_value='extrapolate')

    pos_volt_interp = f_pos_cap_interp(p_capacity_range)
    pos_dqdu_interp = f_pos_dx_interp(p_capacity_range)
    neg_volt_interp = f_neg_cap_interp(n_capacity_range)
    neg_dqdu_interp = f_neg_dx_interp(n_capacity_range)
    
    whole_cell_volt = np.flip(pos_volt_interp) - neg_volt_interp
    whole_cell_dqdu = np.flip(pos_dqdu_interp) + neg_dqdu_interp
    whole_cell_dudq = 1/np.flip(pos_dqdu_interp) + 1/neg_dqdu_interp
    
    if all_output == False:
        return capacity_range, whole_cell_volt, whole_cell_dqdu, whole_cell_dudq
    elif all_output == True:
        return ((capacity_range, p_capacity_range, n_capacity_range), 
                (whole_cell_volt, pos_volt_interp, neg_volt_interp), 
                (whole_cell_dqdu, pos_dqdu_interp, neg_dqdu_interp),
                (whole_cell_dudq, 1/pos_dqdu_interp, 1/neg_dqdu_interp))

def verbrugge_whole_cell_opt(parameter_matrix,
                             volt_range, cap_data, dudq_data,
                             full_volt_min, full_volt_max,
                             temp, nor_pos, nor_neg, 
                             pos_volt_range, neg_volt_range,
                             pos_lower_li_limit, neg_lower_li_limit, 
                             n_p, p_capacity, usable_cap, Qj_or_Xj,
                             error, pos_U0s, neg_U0s,
                             fixed_potential, fixed_li_limit,
                             cap_weight, dudq_weight,
                             all_output = False):

    """
    Cost function used to optimize the MSMR model against experimental OCV and dU/dQ through minimizing the
    mean absolute error

    Parameters:

    parameter_matrix: (N,) A 1-D array of all parameter values of Uj, Qj or Xj, and wj for both parameters
    volt_range: (N,) Range of voltage to perform the optimization over
    cap_data: (N,) Experimental capacity data over the voltage range ot perform optimization over
    dudq_data: (N,) Experimental differential data over the voltage range ot perform optimization over
    full_volt_min: (float) Minimum voltage exhibited in the experimental data
    full_volt_max: (float) Maximum voltage exhibited in the experimental data
    temp: (int or float) Temperature
    nor_pos: (int) Number of reactions in the positive electrode. Helps determine how many parameters in
             the parameter_matrix are those of the positive electrde.
    nor_neg: (int) Number of reactions in the negative electrode. Helps determine how many parameters in
             the parameter_matrix are those of the negative electrde.
    pos_volt_range: (tuple) The voltage range to calculate the MSMR results for the positive electrode.
    neg_volt_range: (tuple) The voltage range to calculate the MSMR results for the negative electrode.
    pos_lower_li_limit: (float) The lower site (x) or capacity bound of the positive electrode (assuming partial utilization)
    neg_lower_li_limit: (float) The lower site (x) or capacity bound of the negative electrode (assuming partial utilization)
    n_p: (float) The ratio of the negative electrode capacity to the positive electrode capacity in whole cells.
         Only necessary if Qj_or_Xj == Xj
    p_capacity: (float) Positive electrode capacity. Only necessary if Qj_or_Xj == Xj
    usable_cap: (float) Usable capacity, rated capacity, or capacity available within a certain voltage window.
    Qj_or_Xj: (str) Must be "Qj" or "Xj" and determines if the MSMR model is computed with intensive (Xj) or
              extensive (Qj) properties.
    error: (str) Type of error to minimize.
    pos_U0s: (1-D array) Array of the standard electrode potentials for the positive electrode in the case one wants
             to keep these parameters constant throughout the fitting
    neg_U0s: (1-D array) Array of the standard electrode potentials for the negative electrode in the case one wants
             to keep these parameters constant throughout the fitting
    fixed_potential: (boolean) If True, standard electrode potentials remain constant throughout optimization
    fixed_li_limit: (boolean) If True, lower Li limits will remain constant throughout optimization.
    cap_weight: (float) Weight of the capacity vs open-circuit voltage data in optimization
    dudq_weight: (float) Weight of the open-circuit voltage data vs differential voltage data in optimization
    all_output: (boolean) If True, returns capacity, V, and dV/dQ for whole cell and both electrodes. If False,
                returns capacity, V, and dV/dQ for just whole cell response.

    Returns:

    residual (float): Value being minimized through optimization

    """
    
    # In fixed voltages, the input parameter matrix will not have have U0 parameters, so we must add them in
    # to get a full parameter matrix that goes into solving the MSMR model.
    if fixed_potential == True:
        input_matrix = np.zeros((nor_pos+nor_neg)*3)
        for i in range(0,nor_pos):
            idx = int(i*3)
            input_matrix[idx] = pos_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int(i*2)]
            input_matrix[idx+2] = parameter_matrix[int(i*2 + 1)]
        for i in range(0,nor_neg):
            idx = int((i+nor_pos)*3)
            input_matrix[idx] = neg_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int((i+nor_pos)*2)]
            input_matrix[idx+2] = parameter_matrix[int((i+nor_pos)*2+1)]

        # 
        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_model, volt_model, dqdu_model, dudq_model = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                   pos_volt_range, neg_volt_range, 
                                                                   pos_lower_li_limit, neg_lower_li_limit, 
                                                                   n_p, p_capacity, usable_cap, Qj_or_Xj)
        
        # Separate function to solve for the residuals
        residual = guess_data_interpolation(cap_data, dudq_data, volt_range,
                                            cap_model, volt_model, dudq_model,
                                            cap_weight, dudq_weight, error)

    elif fixed_potential == False:
        input_matrix = parameter_matrix[0:(nor_pos+nor_neg)*3]

        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_model, volt_model, dqdu_model, dudq_model = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                   pos_volt_range, neg_volt_range, 
                                                                   pos_lower_li_limit, neg_lower_li_limit, 
                                                                   n_p, p_capacity, usable_cap, Qj_or_Xj)
        
        residual = guess_data_interpolation(cap_data, dudq_data, volt_range,
                                            cap_model, volt_model, dudq_model,
                                            cap_weight, dudq_weight, error)
    else:
        pass

    return residual

def guess_data_interpolation(cap_data, dudq_data, volt_range,
                             cap_model, volt_model, dudq_model,
                             cap_weight, dudq_weight, error):

    """
    Takes input experimental data and solved MSMR data, and ensures that they are compared across the same 
    x values and calculates the residuals between model and experimental data.

    Parameters:
    cap_data: (1-D array) Experimental capacity data
    dudq_data: (1-D array) Experimental differential capacity data
    volt_range: (1-D array) Experimental voltage data
    cap_model: (1-D array) MSMR solved capacity data
    volt_model: (1-D array) MSMR solved voltage data
    dudq_model: (1-D array) MSMR solved differential voltage data
    cap_weight: (float)
    dudq_weight: (float)
    error: (str)

    """
    
    f_cap_interp = interp1d(volt_model, cap_model, fill_value='extrapolate')
    f_dudq_interp = interp1d(volt_model, dudq_model, fill_value='extrapolate')

    cap_model_interp = f_cap_interp(volt_range)
    dudq_model_interp = f_dudq_interp(volt_range)

    if error == 'MAE':
        err_cap = cap_data - cap_model_interp
        err_dudq = dudq_data - (-dudq_model_interp)
        err_total = cap_weight*abs(err_cap/np.average(cap_data)) + dudq_weight*abs(err_dudq/np.average(dudq_data))
        try:
            residual = sum(err_total/len(err_total))
            return residual
        except ZeroDivisionError:
            residual = np.inf()
            return residual

    elif error == 'RMSE':
        err_cap = np.sqrt(np.sum((cap_data - cap_model_interp)**2)/len(cap_data))
        err_dudq = np.sqrt(np.sum((dudq_data - (-dudq_model_interp))**2)/len(dudq_data))
        err_total = cap_weight*(err_cap/np.average(cap_data)) + dudq_weight*(err_dudq/np.average(dudq_data))
        return err_total
    else:
        raise ValueError('Error Type missing')
    
def rmse(data_y, guess_y, guess_x, data_x, x_interp):

    """
    Calculates root mean squared error between two datasets by ensuring that they are calculated along the same x-values
    
    data_x: (1-D array) Experimental data of independent variable
    data_y: (1-D array) Experimental data of dependent variable
    guess_x: (1-D array) Model data of independent variable
    guess_y: (1-D array) Model data of dependent variable
    x_interp: (1-D array) Points along the x-axis to compare the experimental and model values to


    """

    guess_interp = interp1d(guess_x, guess_y, fill_value = 'extrapolate') # Gets interpolation ranges for guess x and guess y
    guess_interp_y = guess_interp(x_interp) # Gives you interpolated values of the guess y in the same as data's x
    
    data_interp = interp1d(data_x, data_y,fill_value = 'extrapolate')
    data_interp_y = data_interp(x_interp)
    
    if np.isnan(guess_interp_y).any() == True:
        guess_interp_y = guess_interp_y[~np.isnan(guess_interp_y)]
        data_interp_y = data_interp_y[~np.isnan(guess_interp_y)]
    elif np.isnan(data_interp_y).any() == True:
        guess_interp_y = guess_interp_y[~np.isnan(data_interp_y)]
        data_interp_y = data_interp_y[~np.isnan(data_interp_y)]

    error = np.sqrt(np.sum((data_interp_y - guess_interp_y)**2)/len(data_interp_y))

    return error

def mae(data_y, guess_y, guess_x, data_x, x_interp):
    """
    Calculates mean absolute error between two datasets by ensuring that they are calculated along the same x-values
    
    data_x: (1-D array) Experimental data of independent variable
    data_y: (1-D array) Experimental data of dependent variable
    guess_x: (1-D array) Model data of independent variable
    guess_y: (1-D array) Model data of dependent variable
    x_interp: (1-D array) Points along the x-axis to compare the experimental and model values to

    """

    guess_interp = interp1d(guess_x, guess_y, fill_value='extrapolate') # Gets interpolation ranges for guess x and guess y
    guess_interp_y = guess_interp(x_interp) # Gives you interpolated values of the guess y in the same as data's x
    
    data_interp = interp1d(data_x, data_y, fill_value = 'extrapolate')
    data_interp_y = data_interp(x_interp)
    
    if np.isnan(guess_interp_y).any() == True:
        guess_interp_y = guess_interp_y[~np.isnan(guess_interp_y)]
        data_interp_y = data_interp_y[~np.isnan(guess_interp_y)]
    elif np.isnan(data_interp_y).any() == True:
        guess_interp_y = guess_interp_y[~np.isnan(data_interp_y)]
        data_interp_y = data_interp_y[~np.isnan(data_interp_y)]
    
    error = (np.sum(abs(data_interp_y - guess_interp_y))/len(data_interp_y))

    return error

def percent_residuals(data_y, guess_y, guess_x, data_x, x_interp):
    """
    Calculates the residuals (in %) between two datasets by ensuring that they are calculated along the same x-values
    
    data_x: (1-D array) Experimental data of independent variable
    data_y: (1-D array) Experimental data of dependent variable
    guess_x: (1-D array) Model data of independent variable
    guess_y: (1-D array) Model data of dependent variable
    x_interp: (1-D array) Points along the x-axis to compare the experimental and model values to

    """

    guess_interp = interp1d(guess_x, guess_y, fill_value='extrapolate') # Gets interpolation ranges for guess x and guess y
    guess_interp_y = guess_interp(x_interp) # Gives you interpolated values of the guess y in the same as data's x
    
    data_interp = interp1d(data_x, data_y, fill_value = 'extrapolate')
    data_interp_y = data_interp(x_interp)
    
    return 100*((data_interp_y - guess_interp_y)/data_interp_y)

def pos_li_constraint(parameter_matrix,
                      volt_range, cap_data, dudq_data,
                      full_volt_min, full_volt_max,
                      temp, nor_pos, nor_neg, 
                      pos_volt_range, neg_volt_range,
                      pos_lower_li_limit, neg_lower_li_limit,
                      n_p, p_capacity, usable_cap, Qj_or_Xj,
                      error, pos_U0s=None, neg_U0s=None,
                      fixed_potential=None, fixed_li_limit=None,
                      cap_weight=None, dudq_weight=None,
                      all_output=False):
    
    """
    Imposes a constraint that in the intensive form, all Xj parameters must sum to unity
    and in the extensive form, that all the Qj parameters must sum to the the total insertion capacity (Q)
    for the positive electrode

    """
    if fixed_potential == True:
        if Qj_or_Xj == 'Xj':
            return 1 - parameter_matrix[0:2*nor_pos][0::2].sum()
        elif Qj_or_Xj == 'Qj':
            return p_capacity - parameter_matrix[0:2*nor_pos][0::2].sum()

    elif fixed_potential == False:
        if Qj_or_Xj == 'Xj':
            return 1 - parameter_matrix[0:3*nor_pos][1::3].sum()
        elif Qj_or_Xj == 'Qj':
            return p_capacity - parameter_matrix[0:3*nor_pos][1::3].sum()   
    else:
        pass

def neg_li_constraint(parameter_matrix,
                      volt_range, cap_data, dudq_data,
                      full_volt_min, full_volt_max,
                      temp, nor_pos, nor_neg, 
                      pos_volt_range, neg_volt_range,
                      pos_lower_li_limit, neg_lower_li_limit,
                      n_p, p_capacity, usable_cap, Qj_or_Xj,
                      error, pos_U0s=None, neg_U0s=None,
                      fixed_potential=None, fixed_li_limit=None,
                      cap_weight=None, dudq_weight=None,
                      all_output=False):
    """
    Imposes a constraint that in the intensive form, all Xj parameters must sum to unity
    and in the extensive form, that all the Qj parameters must sum to the the total insertion capacity (Q)
    for the negative electrode
     
    """
    if fixed_potential == True:
        if Qj_or_Xj == 'Xj':
            return 1 - parameter_matrix[2*nor_pos:(2*nor_pos + 2*nor_neg)][0::2].sum()
        elif Qj_or_Xj == 'Qj':
            return p_capacity*n_p - parameter_matrix[2*nor_pos:(2*nor_pos + 2*nor_neg)][0::2].sum()
        else:
            raise ValueError('Input Qj_or_Xj is missing')

    elif fixed_potential == False:
        if Qj_or_Xj == 'Xj':
            return 1 - parameter_matrix[3*nor_pos:][1::3].sum()
        elif Qj_or_Xj == 'Qj':
            return p_capacity*n_p - parameter_matrix[3*nor_pos:-2][1::3].sum()        
        else:
            raise ValueError('Input Qj_or_Xj is missing')
    else:
        pass

def lower_v_constraint(parameter_matrix,
                       volt_range, cap_data, dudq_data,
                       full_volt_min, full_volt_max,
                       temp, nor_pos, nor_neg, 
                       pos_volt_range, neg_volt_range,
                       pos_lower_li_limit, neg_lower_li_limit,
                       n_p, p_capacity, usable_cap, Qj_or_Xj,
                       error, pos_U0s=None, neg_U0s=None,
                       fixed_potential=None, fixed_li_limit=None,
                       cap_weight=None, dudq_weight=None,
                       all_output=False):
    """
    Imposes a constraint that the voltages at delta Q (usable capacity) = 0 must be equal between
    the model and experiment.
     
    """
    if fixed_potential == True:

        input_matrix = np.zeros((nor_pos+nor_neg)*3)
        for i in range(0,nor_pos):
            idx = int(i*3)
            input_matrix[idx] = pos_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int(i*2)]
            input_matrix[idx+2] = parameter_matrix[int(i*2 + 1)]

        for i in range(0,nor_neg):
            idx = int((i+nor_pos)*3)
            input_matrix[idx] = neg_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int((i+nor_pos)*2)]
            input_matrix[idx+2] = parameter_matrix[int((i+nor_pos)*2+1)]

        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_guess, ocv_guess, dqdu_guess, dudq_guess = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                  pos_volt_range, neg_volt_range, 
                                                                  pos_lower_li_limit, neg_lower_li_limit, 
                                                                  n_p, p_capacity, usable_cap, Qj_or_Xj)

    elif fixed_potential == False:
        input_matrix = parameter_matrix[0:(nor_pos+nor_neg)*3]

        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_guess, ocv_guess, dqdu_guess, dudq_guess = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                  pos_volt_range, neg_volt_range, 
                                                                  pos_lower_li_limit, neg_lower_li_limit, 
                                                                  n_p, p_capacity, usable_cap, Qj_or_Xj)

    return full_volt_min - ocv_guess.min()

def upper_v_constraint(parameter_matrix,
                       volt_range, cap_data, dudq_data,
                       full_volt_min, full_volt_max,
                       temp, nor_pos, nor_neg, 
                       pos_volt_range, neg_volt_range,
                       pos_lower_li_limit, neg_lower_li_limit,
                       n_p, p_capacity, usable_cap, Qj_or_Xj,
                       error, pos_U0s=None, neg_U0s=None,
                       fixed_potential=None, fixed_li_limit=None,
                       cap_weight=None, dudq_weight=None,
                       all_output=False):
    """
    Imposes a constraint that the voltages at delta Q (usable capacity) = max must be equal between
    the model and experiment.
     
    """
    if fixed_potential == True:

        input_matrix = np.zeros((nor_pos+nor_neg)*3)
        for i in range(0,nor_pos):
            idx = int(i*3)
            input_matrix[idx] = pos_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int(i*2)]
            input_matrix[idx+2] = parameter_matrix[int(i*2 + 1)]

        for i in range(0,nor_neg):
            idx = int((i+nor_pos)*3)
            input_matrix[idx] = neg_U0s[i]
            input_matrix[idx+1] = parameter_matrix[int((i+nor_pos)*2)]
            input_matrix[idx+2] = parameter_matrix[int((i+nor_pos)*2+1)]

        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_guess, ocv_guess, dqdu_guess, dudq_guess = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                  pos_volt_range, neg_volt_range, 
                                                                  pos_lower_li_limit, neg_lower_li_limit, 
                                                                  n_p, p_capacity, usable_cap, Qj_or_Xj)

    elif fixed_potential == False:
        input_matrix = parameter_matrix[0:(nor_pos+nor_neg)*3]

        if fixed_li_limit == True:
            pass
        elif fixed_li_limit == False:
            pos_lower_li_limit = parameter_matrix[-2]
            neg_lower_li_limit = parameter_matrix[-1]

        cap_guess, ocv_guess, dqdu_guess, dudq_guess = whole_cell(input_matrix, temp, nor_pos, nor_neg, 
                                                                  pos_volt_range, neg_volt_range, 
                                                                  pos_lower_li_limit, neg_lower_li_limit, 
                                                                  n_p, p_capacity, usable_cap, Qj_or_Xj)


    return full_volt_max - ocv_guess.max()

def bootstrapping_fit(voltage_data, capacity_data, dudq_data, 
                      full_voltage_data, full_capacity_data, full_dudq_data,
                      initial_guess, pos_lower_li_limit,
                      nor_pos, nor_neg,
                      neg_lower_li_limit, cap_weight, dudq_weight,
                      N_samples, bootstrap_iterations, bounds, folderpath):

    """
    Code to run bootstrapping on a dataset by fitting a select number of points dataset picked with random 
    sampling with replacement. The plots, parameters, and errors for all the iterations are returned.

    voltage_data: Processed voltage data within a select range
    capacity_data: Processed capacity data within a select voltage range
    dudq_data: Processed dudq data within a select voltage range
    full_voltage_data: Voltage data for the whole dataset
    full_capacity_data: Capacity data for the whole dataset
    full_dudq_data: dUdQ data for the whole dataset
    initial_guess: Initial guess of parameters
    pos_lower_li_limit: Lower lithiation limit of the positive electrode
    neg_lower_li_limit: Lower lithiation limit of the negative electrode
    cap_weight: Weight of the capacity fit in the fitting process
    dudq_weight: Weight of the dUdQ fit in the fitting process
    N_samples: Number of samples to be randomly selected with replacement
    bootstrap_iterations: Number of bootstrap iterations to run
    bounds: Bounds for all the parameters in the fit
    folderpath: Directory to store data


    """

    for k in range(1,bootstrap_iterations+1):
        bootstrap_voltage, bootstrap_capacity, bootstrap_dudq = np.zeros(N_samples), np.zeros(N_samples), np.zeros(N_samples)

        for i in range(0, N_samples):
            idx_generator = np.arange(0,len(voltage_data))
            idx = np.random.choice(idx_generator, replace=True)
            bootstrap_voltage[i] = voltage_data[idx]
            bootstrap_capacity[i] = capacity_data[idx]
            bootstrap_dudq[i] = dudq_data[idx]

            sorted_voltage_order = bootstrap_voltage.argsort()
            sorted_voltage = bootstrap_voltage[sorted_voltage_order[::1]]
            sorted_capacity = bootstrap_capacity[sorted_voltage_order[::1]]
            sorted_dudq = bootstrap_dudq[sorted_voltage_order[::1]]

        yvolt_bs=fmin_slsqp(func=verbrugge_whole_cell_opt, x0=initial_guess, 
                            args=(sorted_voltage,             # interpolated voltage range
                                  sorted_capacity,       # interpolated capacity data
                                  sorted_dudq,     # interpolated dudq data
                                  full_voltage_data.min(),   # Minimum voltage
                                  full_voltage_data.max(),   # Maximum voltage
                                  298,                       # temperature
                                  nor_pos, nor_neg,          # number of pos rxns, number of neg rxns
                                  (3.4,5), (0,1),            # positive voltage range, negative voltange range
                                  pos_lower_li_limit,
                                  neg_lower_li_limit,        # positive | neg lower X or Q limit
                                  None,                      # N|P Ratio
                                  None,                      # positive electrode full capacity
                                  full_capacity_data.max(),  # usable/cyclable capacity within the voltage limits
                                  'Qj', 'MAE',              # Qj or Xj
                                  None,                      # list of pos_U0s
                                  None,                      # list of neg_U0s
                                  False,                     # fixed voltage
                                  True,                      # fixed xj_limit
                                  cap_weight, dudq_weight),                 # capacity vs dudq weight on error function
                            eqcons=[lower_v_constraint, upper_v_constraint],
                            bounds=bounds, iter=500, full_output=False)

        guess_full_bs = yvolt_bs[0:(3*(nor_pos+nor_neg))]

        cap_bs, v_bs, dqdu_bs, dudq_bs = whole_cell(guess_full_bs, nor_pos = nor_pos, nor_neg = nor_neg,
                                                    pos_lower_li_limit = pos_lower_li_limit, neg_lower_li_limit=neg_lower_li_limit,
                                                    temp = 298, pos_volt_range = (3.4, 5), 
                                                    neg_volt_range = (0, 1), p_capacity=None, n_p=None,
                                                    usable_cap = full_capacity_data.max(), Qj_or_Xj='Qj')
        
        capacity_range = np.linspace(0.0, full_capacity_data.max(), 251)
        fit_bs_mae = mae(full_voltage_data, v_bs, cap_bs, full_capacity_data, capacity_range)


        deriv_list = [(v_bs, dudq_bs)]
    
        newvrange = np.linspace(voltage_data.min(), voltage_data.max(), 1000)
        for item in deriv_list:
            if item[0].max() > 4.15:
                v_range = newvrange
            else:
                v_range = np.linspace(voltage_data.min(), item[0].max(), 1000)
        fit_bs_mae_dudq = mae(full_dudq_data, -(item[1]), item[0], full_voltage_data, v_range)

        fig = plt.figure(figsize = (8,4))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax.plot(full_capacity_data, full_voltage_data, label='Data')
        ax.plot(cap_bs, v_bs)
        ax2.plot(full_voltage_data, full_dudq_data, label='Data')
        ax2.plot(v_bs, -dudq_bs)
              
        ax.set_xlabel('Capacity (Ahr)')
        ax.set_ylabel('Potential (V)')
        ax.text(.05, 2.52, 'MAE=%1.3f V' %fit_bs_mae)

        ax2.set_xlabel('Potential (V)')
        ax2.set_ylabel('dU/dQ (V $Ahr^{-1}$')
        ax2.set_xlim(3.4, 4.25)
        ax2.set_ylim(0, 1)
        ax2.text(3.85, 0.02, 'MAE=%1.4f V/Ahr' %fit_bs_mae_dudq)

        plt.tight_layout()

        try:
            fig.savefig(folderpath+'/graphs/graph_of_fits_{}.png'.format(int(k)), dpi=100)
        except FileNotFoundError:
            os.mkdir(folderpath+'/graphs/')
            fig.savefig(folderpath+'/graphs/graph_of_fits_{}.png'.format(int(k)), dpi=100)
        fit_bs_data = np.append(guess_full_bs, (fit_bs_mae, fit_bs_mae_dudq))
        bootstrap_data = np.vstack((np.round(sorted_voltage,4), np.round(sorted_capacity, 4), np.round(sorted_dudq , 4)))
        try:    
            np.savetxt(folderpath+'/params_and_errors/params_and_errors_of_fit_{}.csv'.format(int(k)), fit_bs_data)
            np.savetxt(folderpath+'/params_and_errors/random_sampled_data_{}.csv'.format(int(k)), bootstrap_data, delimiter=',')
        except FileNotFoundError:
            os.mkdir(folderpath+'/params_and_errors/')
            np.savetxt(folderpath+'/params_and_errors/params_and_errors_of_fit_{}.csv'.format(int(k)), fit_bs_data)
            np.savetxt(folderpath+'/params_and_errors/random_sampled_data_{}.csv'.format(int(k)), bootstrap_data, delimiter=',')