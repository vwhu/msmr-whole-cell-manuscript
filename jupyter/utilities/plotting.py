import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def all_electrode_curves(v, cap, dqdu, ax, ax2, electrode_response=None, solid_line=True, error_bar_value=None, 
                         text_x_loc=None, error_bar_x_vloc=None, error_bar_x_hloc=None):
    """
    Takes a dictionary of capacity and dqdu data for calculated individual reactions and plot them
    all along the same voltage (v) axis. If a whole electrode response is given, this function can be used to
    offset the dqdu response and give a scale bar instead.

    Parameters:
    v: Array of Voltages used to calculate all the MSMR results
    cap: Dictionary of capacity or xj values
    dqdu: Dictionary of differential capacity or dxj/du values.

    """


    if electrode_response is not None:
        for i in cap.keys():
            if solid_line == True:
                ax.plot(v, cap[i], label = 'j = {}'.format(i))
            elif solid_line == False:
                ax.plot(v, cap[i], label = 'j = {}'.format(i), ls='--')

        if solid_line == True:
            ax.plot(electrode_response[0], electrode_response[1], label = 'Whole Pos Response', color='k')
        elif solid_line == False:
            ax.plot(electrode_response[0], electrode_response[1], label = 'Whole Pos Response', color='k', ls= '--')

        ax.set_ylabel('Capacity, Q (Ahr)')
        ax.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')

        for i in dqdu.keys():
            if solid_line == True:
                ax2.plot(v, -dqdu[i], label = 'j = {}'.format(i))
            elif solid_line == False:
                ax2.plot(v, -dqdu[i], label = 'j = {}'.format(i), ls = '--')

        dqdu_offset = abs(electrode_response[2]).max()
        text_y_loc = (-electrode_response[2]+dqdu_offset).max() - (error_bar_value/2) * 1.25

        if solid_line == True:
            ax2.plot(electrode_response[0], -electrode_response[2] + dqdu_offset, label = 'Whole Pos Response', color='k')
        elif solid_line == False:
            ax2.plot(electrode_response[0], -electrode_response[2] + dqdu_offset, label = 'Whole Pos Response', color='k', ls='--')

        ax2.errorbar(error_bar_x_vloc, (-electrode_response[2]+dqdu_offset).max() - error_bar_value/2, error_bar_value/2, None, color='k') # vertical lines
        ax2.errorbar(error_bar_x_hloc, (-electrode_response[2]+dqdu_offset).max() - error_bar_value, None, 0.02, color='k') # horizontal lines
        ax2.errorbar(error_bar_x_hloc, (-electrode_response[2]+dqdu_offset).max(), None, 0.02, color='k') # horizontal lines
        ax2.text(x=text_x_loc, y=text_y_loc, s='{} Ahr/V'.format(error_bar_value))

        ax2.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
        ax2.set_ylabel('dQ/dU (Ahr/V)', labelpad = 10)
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        return ax, ax2

    elif electrode_response == None:
        for i in cap.keys():
            if solid_line == True:
                ax.plot(v, cap[i], label = 'j = {}'.format(i))
            elif solid_line == False:
                ax.plot(v, cap[i], label = 'j = {}'.format(i), ls='--')

        ax.set_ylabel('Capacity, Q (Ahr)')
        ax.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')

        for i in dqdu.keys():
            if solid_line == True:
                ax2.plot(v, -dqdu[i], label = 'j = {}'.format(i))
            elif solid_line == False:
                ax2.plot(v, -dqdu[i], label = 'j = {}'.format(i), ls = '--')

        ax2.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
        ax2.set_ylabel('dQ/dU (Ahr/V)', labelpad = 10)


        return ax, ax2

def plot_parameters_bootstrap(bootstrap_params, electrode, lines=False):
    """
    Takes all bootstrapped parameters and the parameters from the evenly-spaced fit (fit_params) 
    and plots the histograms of each parameter.

    Parameters:
    bootstrap_params: All the bootstrap parameters
    electrode: 'pos' or 'neg'
    lines: If True, will draw a solid vertical line at the median of the histogram, and dashed lines
           at the 5th and 95th percentile of the values, denoting the limits of the 90% confidence
           interval.

    """
    fig = plt.figure(figsize = (9, 15))
    nrow = 6
    ncol = 3
    U0_modes, Qj_modes, Wj_modes = [], [], []

    ax1, ax2, ax3 = fig.add_subplot(nrow,ncol,1) , fig.add_subplot(nrow,ncol,2) , fig.add_subplot(nrow,ncol,3)
    ax4, ax5, ax6 = fig.add_subplot(nrow,ncol,4) , fig.add_subplot(nrow,ncol,5) , fig.add_subplot(nrow,ncol,6)
    ax7, ax8, ax9 = fig.add_subplot(nrow,ncol,7) , fig.add_subplot(nrow,ncol,8) , fig.add_subplot(nrow,ncol,9)
    ax10, ax11, ax12 = fig.add_subplot(nrow,ncol,10) , fig.add_subplot(nrow,ncol,11) , fig.add_subplot(nrow,ncol,12)
    ax13, ax14, ax15 = fig.add_subplot(nrow,ncol,13) , fig.add_subplot(nrow,ncol,14) , fig.add_subplot(nrow,ncol,15)
    ax16, ax17, ax18 = fig.add_subplot(nrow,ncol,16) , fig.add_subplot(nrow,ncol,17) , fig.add_subplot(nrow,ncol,18)

    U0_plots = [ax1, ax4, ax7, ax10, ax13, ax16]
    Qj_plots = [ax2, ax5, ax8, ax11, ax14, ax17]
    Wj_plots = [ax3, ax6, ax9, ax12, ax15, ax18]

    ax1.set_title('$U^{0}$')
    ax2.set_title('$Q_{j,max}$')
    ax3.set_title('$\omega$')

    if electrode == 'pos':
        count_U0 = 0
        count_Qj = 1
        count_Wj = 2
    elif electrode == 'neg':
        count_U0 = 18
        count_Qj = 19
        count_Wj = 20
    else:
        raise ValueError

    for ax in U0_plots:
        bin_count, bin_values, patch = ax.hist(bootstrap_params[:,count_U0], bins = 50)
        U0_modes.append(bin_values[np.argmax(bin_count)])
        if lines==True:
            lower_percentile = np.percentile(bootstrap_params[:,count_U0], 5)
            upper_percentile = np.percentile(bootstrap_params[:,count_U0], 95)
            ax.plot((lower_percentile, lower_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.plot((upper_percentile, upper_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.axvline(x=np.median(bootstrap_params[:,count_U0]), color='k')
            ax.set_ylim(0, bin_count.max()*1.1)
        ax.set_xlabel('U0')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        count_U0 += 3
        
    for ax in Qj_plots:
        bin_count, bin_values, patch = ax.hist(bootstrap_params[:,count_Qj], bins = 50)
        Qj_modes.append(bin_values[np.argmax(bin_count)])
        if lines==True:
            lower_percentile = np.percentile(bootstrap_params[:,count_Qj], 5)
            upper_percentile = np.percentile(bootstrap_params[:,count_Qj], 95)
            ax.plot((lower_percentile, lower_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.plot((upper_percentile, upper_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.axvline(x=np.median(bootstrap_params[:,count_Qj]), color='k')
            ax.set_ylim(0, bin_count.max()*1.1)
        ax.set_xlabel('Qj (Ahr)')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        count_Qj += 3
        
    for ax in Wj_plots:
        bin_count, bin_values, patch = ax.hist(bootstrap_params[:,count_Wj], bins = 50)
        Wj_modes.append(bin_values[np.argmax(bin_count)])
        if lines==True:
            lower_percentile = np.percentile(bootstrap_params[:,count_Wj], 5)
            upper_percentile = np.percentile(bootstrap_params[:,count_Wj], 95)
            ax.plot((lower_percentile, lower_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.plot((upper_percentile, upper_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
            ax.axvline(x=np.median(bootstrap_params[:,count_Wj]), color='k')
            ax.set_ylim(0, bin_count.max()*1.1)
        ax.set_xlabel('Wj')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        count_Wj += 3
    
    modes_list = np.zeros(len(U0_modes)*3)
    for i in range(0,len(U0_modes)):
        modes_list[i*3], modes_list[i*3 + 1], modes_list[i*3 + 2]  = U0_modes[i], Qj_modes[i], Wj_modes[i]
    
    plt.tight_layout()

    return fig, modes_list

def individual_electrode_analysis(model_results, pos_whole, neg_whole):
    """
    Takes the MSMR results and first generates one plot of the whole cell response and the underlying negative
    and positive electrode responses that sum to the whole. A second figure plots the capacity, dqdu, and dudq
    of each electrode, with the solid line showing which parts of the electrode are being utilized and the
    dashed components unutilized. Results for all three must be in the extensive forms.

    Parameters:
    model_results: Output of the whole_cell() function with "all_output" set to True.
    pos_whole: Output of the electrode_response() function for the positive electrode. Must be solved extensively
    neg_whole: Output of the electrode_response() function for the negative electrode. Must be solved extensively

    """
    capacities, voltages, dqdus, dudqs = model_results[0], model_results[1], model_results[2], model_results[3]
    pv, pq, pd = pos_whole
    nv, nq, nd = neg_whole
    fig1 = plt.figure(figsize = (8, 4))

    ax = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)

    ax.plot(capacities[0], voltages[0], label='Whole Cell', color='k')
    ax.plot(np.flip(capacities[0]), voltages[1], label='Positive', color='r')
    ax.plot(capacities[0], voltages[2], label='Negative', ls='--', color='b')
    ax.set_xlabel('Capacity (Ahr)')
    ax.set_ylabel('Voltage')

    ax2.plot(voltages[0], -dudqs[0], label='Whole Cell', color='k')
    ax2.plot(np.flip(voltages[0]), -dudqs[1], label='Positive', color='r')
    ax2.plot(voltages[0], -dudqs[2], label='Negative', ls='--', color='b')
    ax2.set_xlabel('Voltage')
    ax2.set_ylabel('dU/dQ (V/Ahr)')

    ax.set_ylim()
    ax2.set_xlim(3.4, 4.2)
    ax2.set_ylim(0,1.5)
    fig1.tight_layout()
    
    fig2 = plt.figure(figsize = (12, 8))

    axe1, axe2, axe3 = fig2.add_subplot(231), fig2.add_subplot(232), fig2.add_subplot(233)
    axe4, axe5, axe6 = fig2.add_subplot(234), fig2.add_subplot(235), fig2.add_subplot(236)

    cap_ax, dqdu_ax, dudq_ax = [axe1, axe4], [axe2, axe5], [axe3, axe6]

    axe1.plot(pq, pv, ':', color = 'r')
    axe1.plot(capacities[1], voltages[1], color = 'r', label='Positive Electrode')

    axe2.plot(pv, -pd, ':', color='r')
    axe2.plot(voltages[1], -dqdus[1], color='r')

    axe3.plot(pv, -1/pd, ':', color='r')
    axe3.plot(voltages[1], -dudqs[1], color='r')

    axe4.plot(nq, nv, ':', color = 'b')
    axe4.plot(capacities[2], voltages[2], color = 'b', label='Negative Electrode')

    axe5.plot(nv, -nd, ':', color='b')
    axe5.plot(voltages[2], -dqdus[2], color='b')

    axe6.plot(nv, -1/nd, ':', color='b')
    axe6.plot(voltages[2], -dudqs[2], color='b')

    for axe in cap_ax:
        axe.set_xlabel('Capacity (Ahr)')
        axe.set_ylabel('Voltages (V)')
        axe.legend()

    for axe in dqdu_ax:
        axe.set_xlabel('Voltage')
        axe.set_ylabel('dQdU (V/Ah)')

    for axe in dudq_ax:
        axe.set_xlabel('Voltage')
        axe.set_ylabel('dUdQ (V/Ah)')
        axe.set_ylim(0,1)

    axe5.set_xlim(0,0.4)
    axe6.set_xlim(0,0.4)
    fig2.tight_layout()
    
    return fig1, fig2