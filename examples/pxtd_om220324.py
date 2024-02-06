import plasma_analysis.tdstreak as td 
import matplotlib.pyplot as plt
import numpy as np
import os.path as pt
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


directory = '/Users/tuckerevans/Documents/MIT/HEDP/multiionSeries/MultiIon-22A/PTD/data_files/'
files = os.listdir(directory) 
#files = [files[0]]

shot_num_colors = {
    "103739": 'red', 
    "103740": 'red', 
    "103741": 'red', 
    "103743": 'green', 
    "103744": 'green', 
    "103745": 'green', 
    "103746": 'goldenrod', 
    "103747": 'goldenrod', 
    "103748": 'goldenrod', 
    "103749": 'teal', 
    "103750": 'teal', 
    "103751": 'teal'
}

combo_line_fig, combo_line_ax = plt.subplots(1,2)
peak_centers0 = []
peak_centers1 = []
DTpeaks = []
D3Hepeaks = []
shotnums = []

DTwidths = []
D3Hewidths = []
DDwidths = []



for filename in files: 
    print('=============================') 
    print(f'Analyzing {filename}...') 
    print('=============================') 
    plt.figure()
    filtered_image, shot_num = td.show_pxtd_image(directory + filename)
    
    savename = 'ptd'+str(shot_num) 
    savedir = './'+savename + '/'
    plotdir = savedir + 'plots/'
    lineoutdir = savedir + 'lineouts/'
    shotnums.append(float(shot_num))
    

    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
        os.mkdir(plotdir)
        os.mkdir(lineoutdir)

    plt.savefig(plotdir + savename+ '_fullImage.png') 
    plt.close()

    #showing the image and getting the lineouts/timing
    lineouts = td.pxtd_2ch_lineouts(directory+filename)
    bg_lineout = td.pxtd_lineout(directory+filename, 235)
    notch_correction = []
    with open('ptd_notch_correction.txt', 'r') as notch_file:
        for element in notch_file:
            entries = element.split(',')
            notch_correction.append(float(entries[1]))
    notch_correction = np.array(notch_correction)
    corrected_lineouts = []
    for lineout in lineouts:
        corrected_lineouts.append((lineout - bg_lineout)/notch_correction) 
    time_axis, centers, fid_lineout = td.get_pxtd_fid_timing(directory + filename)

    # wiener deconvolutions
    output_time0, output_decon0 = td.wiener_decon_lineout(time_axis, corrected_lineouts[0], noise_left = 40, noise_right = 120, signal_left = 40, signal_right= 480)
    output_time1, output_decon1 = td.wiener_decon_lineout(time_axis, corrected_lineouts[1],noise_left = 40, noise_right = 120, signal_left = 40, signal_right= 480)

    # plotting all of the lineouts
    fig, ax = plt.subplots()
    #ax.plot(time_axis, fid_lineout, c ='g') 
    ax.plot(time_axis, lineouts[0], c = 'k', linestyle = ':') 
    ax.plot(time_axis, lineouts[1], c = 'k', linestyle = '--') 
    ax.plot(output_time0, output_decon0, c = 'r', linestyle = ':')
    ax.plot(output_time1, output_decon1, c = 'r', linestyle = '--')
    plt.grid()
    plt.legend(['lineout 0', 'lineout 1', 'decon 0', 'decon 1'])
    plt.title(f'{shot_num}')
    plt.savefig(plotdir+savename+'_lineouts.png') 
    plt.close()

    # plotting all of the normalized lineouts
    #ax.plot(time_axis, fid_lineout, c ='g') 
    fig, ax = plt.subplots()
    ax.plot(time_axis, lineouts[0]/np.max(lineouts[0]), c = 'k', linestyle = ':') 
    ax.plot(time_axis, lineouts[1]/np.max(lineouts[1]), c = 'k', linestyle = '--') 
    ax.plot(output_time0, output_decon0/np.max(output_decon0), c = 'r', linestyle = ':')
    ax.plot(output_time1, output_decon1/np.max(output_decon1), c = 'r', linestyle = '--')
    plt.grid()
    plt.legend(['fiducial', 'lineout 0', 'lineout 1', 'decon 0', 'decon 1'])
    plt.savefig(plotdir+savename+'_normLineouts.png') 

    print(time_axis.shape)
    print(fid_lineout.shape) 
    timed_lineouts = np.vstack([time_axis, fid_lineout, lineouts[0], lineouts[1], corrected_lineouts[0], corrected_lineouts[1]])
    timed_decons0 = np.vstack([output_time0, output_decon0]) 
    timed_decons1 = np.vstack([output_time1, output_decon1]) 
    print(len(lineouts[0]))
    print(timed_lineouts.shape)

    np.savetxt(lineoutdir+savename + '_lineouts.txt', np.transpose(timed_lineouts), delimiter = ", ")
    np.savetxt(lineoutdir+savename + '_decons0.txt', np.transpose(timed_decons0), delimiter = ", ")
    np.savetxt(lineoutdir+savename + '_decons1.txt', np.transpose(timed_decons1), delimiter = ", ")

    #finding all of the peaks in each lineout:

    peaktimes0 = []
    peaktimes1 = []

    peaks0, properties0 = find_peaks(output_decon0/np.max(output_decon0), height = .08, distance = 50)
    for peak in peaks0:
        peaktimes0.append(output_time0[peak]) 
    
    peaks1, properties1 = find_peaks(output_decon1/np.max(output_decon1), height = .08, distance = 50)
    for peak in peaks1:
        peaktimes1.append(output_time1[peak]) 
    #plt.vlines(peaktimes0, ymin = 0, ymax = np.max(output_decon0), colors = 'goldenrod')
    #plt.vlines(peaktimes1, ymin = 1, ymax = np.max(output_decon1),colors = 'goldenrod')
    print(f'channel 0 peak times: {peaktimes0}') 
    print(f'channel 1 peak times: {peaktimes1}') 
    
    # finding indices of the output deconvolution 50% rise time and then
    # plotting all of these on one plot for a given channel to get a sense
    # of the change for different gas fills

    rise_ind0 = 0
    rise_ind1 = 0

    element = 0
    while element <= np.max(output_decon0)*.5:
        rise_ind0 += 1
        element =  output_decon0[rise_ind0]

    element = 0
    while element <= np.max(output_decon1)*.5:
        rise_ind1 += 1
        element =  output_decon1[rise_ind1]
        
    edge_time0 = output_time0[rise_ind0]
    edge_time1 = output_time1[rise_ind1]


    #FITTING THE PEAK WITH TWO GAUSSIANS

    
    print('Peak Fitting w/ Gaussians')
    window = 15

    fit_peaks0 = []
    for peak in peaks0:
        peak_ta = []
        peak_data = []

        for element in range(peak-window, peak+ window):
            peak_ta.append(output_time0[element]) 
            peak_data.append(output_decon0[element])

        peak_opt, peak_cov = curve_fit(td.Gauss, peak_ta, peak_data/np.max(output_decon0), p0 = [1, 100, peak_ta[int(np.floor(window/2))]], method = 'trf')
        ax.plot(peak_ta, td.Gauss(peak_ta, peak_opt[0], peak_opt[1], peak_opt[2]), c = 'goldenrod')


        print(peak_opt[2])
        #print(peak_opt[5]+peak_ta[0])
        print(peak_cov)
        fit_peaks0.append(peak_opt[2]) 

    fit_peaks1 = []    
    for peak in peaks1:
        peak_ta = []
        peak_data = []

        for element in range(peak-window, peak+ window):
            peak_ta.append(output_time1[element]) 
            peak_data.append(output_decon1[element])

        peak_opt, peak_cov = curve_fit(td.Gauss, peak_ta, peak_data/np.max(output_decon1), p0 = [1, 100, peak_ta[int(np.floor(window/2))]], method = 'trf', maxfev = 2400)
        ax.plot(peak_ta, td.Gauss(peak_ta, peak_opt[0], peak_opt[1], peak_opt[2]), c = 'goldenrod')


        print(peak_opt[2])
        #print(peak_opt[5]+peak_ta[0])
        print(peak_cov)
        fit_peaks1.append(peak_opt[2]) 

    combo_line_ax[0].plot(output_time0 - fit_peaks0[0], output_decon0/np.max(output_decon0), c = shot_num_colors[shot_num]) 
    combo_line_ax[1].plot(output_time1 - fit_peaks0[0], output_decon1/np.max(output_decon1), c = shot_num_colors[shot_num]) 

    DTpeaks.append(fit_peaks0[0])
    D3Hepeaks.append(fit_peaks1[0])




    
        
plt.figure()
plt.scatter(shotnums, np.array(DTpeaks)-np.array(D3Hepeaks)) 
plt.xlabel('shot number')
plt.ylabel('non-TOF DT/D3He BT sep')

        

    #combo_line_ax[0].set_xlim([-500, 1600]) 
    #combo_line_ax[1].set_xlim([-500, 1600])

    

    
    
plt.show()


