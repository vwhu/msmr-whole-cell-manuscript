import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import utilities.msmr as msmr

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def all_electrode_curves(vrange, params, T, nor, ax, ax2, electrode_response=True, solid_line=True, error_bar_value=None, 
                          text_x_loc=None, error_bar_x_vloc=None, error_bar_x_hloc=None):
    """
    Takes a dictionary of capacity and dqdu data for calculated individual reactions and plot them
    all along the same voltage (v) axis. If a whole electrode response is given, this function can be used to
    offset the dqdu response and give a scale bar instead.

    Parameters:
    vrange: Array of voltages used to calculate all the MSMR results
    params: 1D array of electrode parameters
    T: temperature
    nor: number of reactions in electrode
    ax: Axis for Potential vs Capacity or X plot
    ax2: Axis for Potential vs dQdU or dXdU plot

    """
    # Plotting each individual reaction first
    if electrode_response == True:
        whole_response = msmr.electrode_response(params, T, vrange.min(), vrange.max(), nor)
        dqdu_offset = abs(whole_response[2]).max()
        text_y_loc = (-whole_response[2]+dqdu_offset).max() - (error_bar_value/2) * 1.25
        
        for i in range(0,nor):
            cap, dqdu = msmr.individual_reactions(vrange, params[0+(i*3)], params[1+(i*3)], params[2+(i*3)], T)
            if electrode_response == True:
                if solid_line == True:
                    ax.plot(vrange, cap, label = 'j = {}'.format(i))
                    ax.plot(whole_response[0], whole_response[1], label = 'Whole Response', color='k')
                    ax2.plot(vrange, -dqdu, label = 'j = {}'.format(i))
                    ax2.plot(whole_response[0], -whole_response[2] + dqdu_offset, label = 'Whole Response', color='k')

                elif solid_line == False:
                    ax.plot(vrange, cap, label = 'j = {}'.format(i), ls='--')
                    ax.plot(whole_response[0], whole_response[1], label = 'Whole Response', color='k', ls= '--')
                    ax2.plot(vrange, -dqdu, label = 'j = {}'.format(i), ls = '--')
                    ax2.plot(whole_response[0], -whole_response[2] + dqdu_offset, label = 'Whole Response', color='k', ls='--')

        ax.set_ylabel('Capacity, Q (Ahr)')
        ax.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
                
        ax2.errorbar(error_bar_x_vloc, (-whole_response[2]+dqdu_offset).max() - error_bar_value/2, error_bar_value/2, None, color='k') # vertical lines
        ax2.errorbar(error_bar_x_hloc, (-whole_response[2]+dqdu_offset).max() - error_bar_value, None, 0.02, color='k') # horizontal lines
        ax2.errorbar(error_bar_x_hloc, (-whole_response[2]+dqdu_offset).max(), None, 0.02, color='k') # horizontal lines
        ax2.text(x=text_x_loc, y=text_y_loc, s='{} Ahr/V'.format(error_bar_value))

        ax2.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
        ax2.set_ylabel('dQ/dU (Ahr/V)', labelpad = 10)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        return ax, ax2

    elif electrode_response == False:
        for i in range(0,nor):
            cap, dqdu = msmr.individual_reactions(vrange, params[0+(i*3)], params[1+(i*3)], params[2+(i*3)], T)
            if solid_line == True:
                ax.plot(vrange, cap, label = 'j = {}'.format(i))
                ax2.plot(v, -dqdu[i], label = 'j = {}'.format(i))
            elif solid_line == False:
                ax.plot(vrange, cap, label = 'j = {}'.format(i), ls='--')
                ax2.plot(vrange, -dqdu, label = 'j = {}'.format(i), ls = '--')

        ax.set_ylabel('Capacity, Q (Ahr)')
        ax.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
        ax2.set_xlabel('Potential vs Li/Li$^{+}$, U (V)')
        ax2.set_ylabel('dQ/dU (Ahr/V)', labelpad = 10)
        return ax, ax2

def plot_all_replicates(cap_lists, voltage_lists, dvdq_lists, labels, cycle_num):
    fig = plt.figure(figsize = (6,6))
    ax, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)
    for i in (ax, ax2):
        i.set_xlabel('Capacity (Ahr)')
        i.set_ylabel('Potential (V)')
        i.set_xlim(-0.05, 1.5)
        i.set_ylim(2.4, 4.25)
    for i in (ax3, ax4):
        i.set_xlim(3.4, 4.2)
        i.set_ylim(0,1)
        i.set_xlabel('Potential (V)')
        i.set_ylabel('dV/dQ (V/Ahr)')
    for i in range(0,len(cap_lists[0])):
        ax.plot(cap_lists[0][i], voltage_lists[0][i], label=labels[i])
        ax2.plot(cap_lists[1][i], voltage_lists[1][i], label=labels[i])    
        ax3.plot(voltage_lists[0][i], dvdq_lists[0][i], label=labels[i])
        ax4.plot(voltage_lists[1][i], dvdq_lists[1][i], label=labels[i])
    ax.legend(loc=4)
    ax.set_title('{} Cycles - Charge'.format(cycle_num))
    ax2.set_title('{} Cycles - Discharge'.format(cycle_num))
    plt.tight_layout()

def plot_parameters_bootstrap(bootstrap_params, lines=False):
    """
    Takes all bootstrapped parameters and the parameters from the evenly-spaced fit (fit_params) 
    and plots the histograms of each parameter.

    Parameters:
    bootstrap_params: All the bootstrap parameters
    lines: If True, will draw a solid vertical line at the median of the histogram, and dashed lines
           at the 5th and 95th percentile of the values, denoting the limits of the 90% confidence
           interval.

    """
    fig = plt.figure(figsize = (19, 15))
    main_fig_width = 0.16
    nrow = 6
    ncol = 6
    gs = gridspec.GridSpec(nrow, ncol+1, width_ratios=[main_fig_width, main_fig_width, main_fig_width,
                                                       1-(main_fig_width*6),
                                                       main_fig_width, main_fig_width, main_fig_width])

    U0_pos = [fig.add_subplot(gs[i,0]) for i in range(0,nrow)]
    Qtot_pos = [fig.add_subplot(gs[i,1]) for i in range(0,nrow)]
    wj_pos = [fig.add_subplot(gs[i,2]) for i in range(0,nrow)]
    U0_neg = [fig.add_subplot(gs[i,4]) for i in range(0,nrow)]
    Qtot_neg = [fig.add_subplot(gs[i,5]) for i in range(0,nrow)]
    wj_neg = [fig.add_subplot(gs[i,6]) for i in range(0,nrow)]

    list_of_subplots = [U0_pos, Qtot_pos, wj_pos, U0_neg, Qtot_neg, wj_neg]
    counts_list = [0, 1, 2, 18, 19, 20]
    labels_list = [r'$U_{pos}^{0}$ V vs Li/Li$^{+}$', r'$Q_{j,tot,pos}$', r'$\omega_{j,pos}$',
                   r'$U_{neg}^{0}$ V vs Li/Li$^{+}$', r'$Q_{j,tot,neg}$', r'$\omega_{j,neg}$']
    lines=True
    for idx, param in enumerate(list_of_subplots):
        count = counts_list[idx]
        for ax in param:
            bin_count, bin_values, patch = ax.hist(bootstrap_params[:,count], bins=50)
            if lines==True:
                lower_percentile = np.percentile(bootstrap_params[:,count], 5)
                upper_percentile = np.percentile(bootstrap_params[:,count], 95)
                ax.plot((lower_percentile, lower_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
                ax.plot((upper_percentile, upper_percentile), (2.25*bin_count.max()/3, 1000), color='k', ls=':')
                ax.axvline(x=np.median(bootstrap_params[:,count]), color='k')
                ax.set_ylim(0, bin_count.max()*1.1)
            ax.set_xlabel(labels_list[idx], fontsize=12)
            count += 3

    wj_pos[0].set_title('(a)', loc='right', fontsize=36, pad=18)
    wj_neg[0].set_title('(b)', loc='right', fontsize=36, pad=18)
    #ax5.suptitle('(b)', fontsize=18, loc='right')

    plt.tight_layout()

    return fig

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