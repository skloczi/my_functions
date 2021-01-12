from scipy.fft import fft, ifft
import numpy as np
import math
from scipy import signal

def rm_damaged_signal(signal_array, threshold):
    """Removes damaged signals from an array of recordings which absolute
    amplitude average is below given threshold.

    Parameters
    ----------
    signal_array : array_like
        Signal data in a matrix form (rows - signal data, columns - sensor)
    threshold : float
        Threshold of the absolute average in reference to the max average below
        which the signals are rejected from analysis and replace with "nan"

    Returns
    -------
    Matrix with removed signals below given threshold. "Damaged" signal replaced
    with "nan"
    """
    sig_len = signal_array.shape[0]
    nb_sens = signal_array.shape[1]

    # creating empty matrices for calculation
    absolute_signal_matrix = np.zeros((sig_len,nb_sens))
    mean_signal_vector = np.zeros(nb_sens)
    good_signal = np.zeros((sig_len,nb_sens))

    for column in range(nb_sens):
        absolute_signal = abs(signal_array[:,column])
        mean_signal = sum(absolute_signal)/sig_len
        absolute_signal_matrix[:,column] = absolute_signal
        mean_signal_vector[column] =  mean_signal

    # max value of mean to normalize mean
    max_ampl = max(mean_signal_vector)
    # mean normalization
    for column in range(nb_sens):
        nor_mean = mean_signal_vector[column]/max_ampl
        # removing whole signals where first recording is damaged
        if mean_signal_vector[0]/max_ampl < threshold:
            good_signal[:,column] = float("nan")
        # removing recordings form sensors which average is below treshold
        elif nor_mean < threshold:
            good_signal[:,column] = float("nan")
        # copying the good data in the good_signal matrix
        else:
            good_signal[:,column] = signal_array[:,column].copy()

    return good_signal

def deconvo(s1, s2, WL):
    """Deconvolution of two signals with water-level regularization technique

    Parameters
    ----------
    s1 :    array_like
        Reference time signal
    s2 :    array_like
        Deconvolved time signal
    WL	:    float
        Water level - minmun spectrum level in % compared to the max of the specturm

    Returns
    -------
    Deconvolved signal in time

    Method by :
    ----------
    CLAYTON, R.W., and WIGGINS, R.A. (1976), Source shape estimation
    and deconvolution of teleseismic bodywaves, The
    Geophysical Journal of the Royal Astronomy Society, 47, 151-177.
    """
    n = max(len(s1), len(s2))
    S2=fft(s2,n)
    S2a=abs(S2)
    S2new = np.zeros(len(S2),dtype = "complex_")
    WL2 = max(S2a) * WL / 100

    for i in range(len(S2new)):
        if S2a[i] == 0:
            S2new[i] = WL2
        elif S2a[i] < WL2: #and np.isclose(S2a[i],0):
            S2new[i] = WL2 * S2[i]/S2a[i]
        elif S2a[i] >= WL2:
            S2new[i] = S2[i]

    S1 = fft(s1, n)
    CC = S1 / S2new
    cc = np.real(ifft(CC))
    return cc

def interfer_deconvo(s1, s2, wl, r, dstack, Fs, t_ax = False):
    """

    Parameters:
    ----------
    s1  :   array_like
        Signal data of the reference signal
    s2  :   array_like
        Signal data of the deconvolved signal
    wl  :   float
        Water-level in %. The minmun spectrum level in % compared to the max of the specturm
    r   :   int/float
        Factor of signal resampling for better peak picking (signal * r)
    dstack  :   float
        Duration of the deconvolved signal to stack (in [sec])
    Fs  :   int/float
        Sampling frequency of the signal
    t_ax    :   bool, optional
        Define if the time axis for the interferogram is calculated

    Returns
    -------
    sig_deco_r : array_like
        The deconvolved signal centered within -dstack:dstack time range
    t   :   array_like
        Time axis for the deconvolved signal
    """

    sig_deco = deconvo(s1, s2, wl)
    nt = len(sig_deco); # deconvolved signal length
    Fs_r = Fs * r

    # flipping the signal
    sig_deco = np.concatenate((sig_deco[(math.floor(nt/2)):],sig_deco[:(math.floor(nt/2))]))

    # signal resampling -> resample(sig, number of samples) --> sig * p/q
    sig_deco = signal.resample(sig_deco, nt*r);
    nt2 = len(sig_deco);   # length of the resampled signal

    # deconvolved and resampled signal - taken just the dstack*2 length
    sig_deco_r = sig_deco[(math.floor(nt2/2+1)-int(dstack*Fs_r)):(math.floor(nt2/2+1)+1+int(dstack*Fs_r))]

    # time axis for the interferogram
    if t_ax == True:
        t = np.linspace(-dstack, dstack, len(sig_deco_r))
        return sig_deco_r, t
    else:
        return sig_deco_r
