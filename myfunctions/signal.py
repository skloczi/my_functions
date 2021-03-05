import numpy as np
import math
from scipy import signal, fftpack
from scipy.signal import butter, sosfilt, zpk2sos, iirfilter
import matplotlib.pyplot as plt

## Damaged signals removal
def rm_damaged_signal(signal_array, threshold, rm_first = False, first = 0):
    """Removes damaged signals from an array of recordings which absolute
    amplitude average is below given threshold.

    Parameters
    ----------
    signal_array    :   array_like
        Signal data in a matrix form (rows - signal data, columns - sensor)
    threshold   :   float
        Threshold of the absolute average in reference to the max average below
        which the signals are rejected from analysis and replace with "nan"
    rm_first    :   bool, optional
        If True, if first recoridng in the line is damaged, the whole line is
        replaced with "nan" - i.e. not taken under consideration in futher analysis
    first   :   int, optional
        Defining which sensor is taken under consideration as first in the line.
        Important when rm_first = True.

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

        ## if removing all the recoding in case of damage 1st in the linspace
        if rm_first == True:
            # removing whole signals where first recording is damaged
            if mean_signal_vector[first]/max_ampl < threshold:
                good_signal[:,column] = float("nan")
            # removing recordings form sensors which average is below treshold
            elif nor_mean < threshold:
                good_signal[:,column] = float("nan")
            # copying the good data in the good_signal matrix
            else:
                good_signal[:,column] = signal_array[:,column].copy()
        else:
            # removing recordings form sensors which average is below treshold
            if nor_mean < threshold:
                good_signal[:,column] = float("nan")
            # copying the good data in the good_signal matrix
            else:
                good_signal[:,column] = signal_array[:,column].copy()
    return good_signal


## Deconvolution
def deconvo(s1, s2, WL, freq_domain = False):
    """Deconvolution of two signals with water-level regularization technique

    Parameters
    ----------
    s1 :    array_like
        Deconvolved time signal
    s2 :    array_like
        Reference time signal
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
    S2=fftpack.fft(s2,n)
    S2a=abs(S2)
    S2new = np.zeros(len(S2),dtype = "complex")
    WL2 = max(S2a) * WL / 100

    for i in range(len(S2new)):
        if S2a[i] == 0:
            S2new[i] = WL2
        elif S2a[i] < WL2: #and np.isclose(S2a[i],0):
            S2new[i] = WL2 * S2[i]/S2a[i]
        elif S2a[i] >= WL2:
            S2new[i] = S2[i]

    S1 = fftpack.fft(s1, n)
    CC = S1 / S2new
    if freq_domain == True:
        return CC
    else:
        cc = np.real(fftpack.ifft(CC))
        return cc

def freq_deconvo(s1, s2, WL):
    S2a=abs(s2)
    s1 = abs(s1)
    S2new = np.zeros(np.shape(s2),dtype = "complex")
    if s2.ndim == 1:
        WL2 = max(S2a) * WL / 100
        for i in range(np.shape(S2new)[0]):
            if S2a[i] == 0:
                S2new[i] = WL2
            elif S2a[i] < WL2: #and np.isclose(S2a[i],0):
                S2new[i] = WL2 * s2[i]/S2a[i]
            elif S2a[i] >= WL2:
                S2new[i] = s2[i]
    else:
        for kk in range(np.shape(s2)[1]):
            WL2 = max(S2a[:,kk]) * WL / 100
            for i in range(np.shape(S2new)[0]):
                if S2a[i,kk] == 0:
                    S2new[i,kk] = WL2
                elif S2a[i,kk] < WL2: #and np.isclose(S2a[i],0):
                    S2new[i,kk] = WL2 * s2[i,kk]/S2a[i,kk]
                elif S2a[i,kk] >= WL2:
                    S2new[i,kk] = s2[i,kk]
    CC = s1 / S2new
    return CC

## Seismic interferometry by deconvolution
def interfer_deconvo(s1, s2, wl, r, dstack, Fs, t_ax = False):
    """Interferometry by deconvolution with water-level.

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
    if not math.isnan(s1[0]):
        sig_deco = deconvo(s1, s2, wl)
        nt = len(sig_deco); # deconvolved signal length
        Fs_r = Fs * r   # resampled sampling frequency

        # flipping the signal
        sig_deco = np.concatenate((sig_deco[(math.floor(nt/2)):],sig_deco[:(math.floor(nt/2))]))

        # signal resampling -> resample(sig, number of samples) --> sig * p/q
        sig_deco = signal.resample(sig_deco, nt*r);
        nt2 = len(sig_deco);   # length of the resampled signal

        # deconvolved and resampled signal - taken just the dstack*2 length
        sig_deco_r = sig_deco[(math.floor(nt2/2)-int(dstack*Fs_r)):(math.floor(nt2/2)+1+int(dstack*Fs_r))]

        # time axis for the interferogram
        if t_ax == True:
            t = np.linspace(-dstack, dstack, len(sig_deco_r))
            return sig_deco_r, t
        else:
            return sig_deco_r

## Damaged ssensors removal
def rm_damaged_sensors(signal_array, damaged_sensors):
    """Function to remove damaged sensors from the array (with a systemic
    malfunction). Setting the damaged signals to "NaN"

    Parameters
    ----------
    signal_array    :   array_like
        Matrix of a sensors array. Rows - signal data, columns - sensors
    damaged_sensors :   array_like
        A vector containing damaged sensors numbers in the array like (fitst
        sensor is 1, not 0)

    Returns
    -------
    cleared_array   :   array_like
        Matrix of a sensors array with damaged signals data set to "NaN".
    """
    for sensor in damaged_sensors:
        signal_array[:,sensor-1] = float("nan")
    return signal_array

def stacked_fft(path, file_list, line, sensor_number):

    count = 1
    non_nans = np.zeros(sensor_number)

    # loop over each file/recording in the line
    for file in file_list:
        data = np.load(path+file)
        n = data.shape[0]
        print(file, f'the iteration number is {count}')
        # checking how many sensors are not nans -> used later for averaging the spectrum
        non_nans += np.all(~np.isnan(data), axis=0)
        # creation of empty matrices/vectors
        abs_fft_matrix = np.empty((n, sensor_number))# matrix for abs FFT for each sensor in each file - reset afer loading a  new file

        # loop over each sensor in the line for one file/recording
        for sensor in range(sensor_number):
            data[np.isnan(data)] = 0
            fourier_transform = fftpack.rfft(data[:,sensor])
            abs_fft_matrix[:, sensor] = abs(fourier_transform)

        # stacking the results
        if count == 1:
            abs_fft_stacked = abs_fft_matrix
            all_one_sensor = abs_fft_matrix[:,0]
        else:
            abs_fft_stacked += abs_fft_matrix
            all_one_sensor = np.append(all_one_sensor, abs_fft_matrix[:,0])

        count += 1
    abs_fft_stacked = abs_fft_stacked/non_nans  # averaging the stacked ffts
    return non_nans, abs_fft_stacked


## Cross-correlation with the max lag
def nextpow2(x):
    if x == 0:
        y = 0
    else:
        y = math.ceil(math.log2(x))
    return y


def xcorr(x, y, maxlag):
    m = max(len(x), len(y))
    mx1 = min(maxlag, m - 1)
    ceilLog2 = nextpow2(2 * m - 1)
    m2 = 2 ** ceilLog2

    X = fftpack.fft(x, m2)
    Y = fftpack.fft(y, m2)
    c1 = np.real(fftpack.ifft(X * np.conj(Y)))
    index1 = np.arange(1, mx1+1, 1) + (m2 - mx1 -1)
    index2 = np.arange(1, mx1+2, 1) - 1
    c = np.hstack((c1[index1], c1[index2]))
    return c


## Butter filter
def butter_bandpass(data, freqmin, freqmax, fs, corners=4, zerophase=False):
    nyq = 0.5 * fs
    low = freqmin / nyq
    high = freqmax / nyq
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, nyq)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def butter_lowpass(data, cutoff, fs, corners=4, zerophase=False):
    nyq = 0.5 * fs
    f = cutoff / nyq
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

def butter_highpass(data, cutoff, fs, corners=4, zerophase=False):
    nyq = 0.5 * fs
    f = cutoff / nyq
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

# moving window fft
def moving_window_fft(data, win_len, step, fs, plot = False, freq_axis = False):
    """Function to calculate an average of the moving window FFT. Window used is
    a Hanning window.

    Parameters
    ----------
    data    :   array_like
        Signal data in time domain
    win_len :   int
        Hanning window lenght as a number of data points taken under consideration
    step    :   int
        Step of a moving Window
    fs      :   int
        Sampling frequency of the analyzed signal
    plot    : bool
        If True, a plot of the operation happening will be shown

    Returns
    -------
    An average of the moving window Fourier transform of the signal. average
    is calcualted by dividing of the sum by the number of considered windows.
    """
    hann_win = np.hanning(win_len +1)[:-1]
    freqaxis =  fftpack.rfftfreq(win_len, d=1./fs)
    number = 0
    for n in range(0,len(data)-win_len,step):
        temp_data = data[n:(n+win_len)]
        temp_window = temp_data * hann_win
        temp_fft = fftpack.rfft(temp_window)
        temp_abs_fft = abs(temp_fft)
        if number == 0:
            abs_fft_tot_X = temp_abs_fft
        else:
            abs_fft_tot_X += temp_abs_fft
        number += 1

    if plot == True:
        fig, ax = plt.subplots(figsize = (15,7))
        ax.loglog(freqaxis,abs_fft_tot_X/number, label = "X")
        ax.set_xlabel("Frequency [Hz]", fontsize = 20)
        ax.set_ylabel("Amplitude", fontsize = 20)
        ax.set_title("Moving window FFT", fontsize = 20)
        ax.legend()
        plt.show()

    if freq_axis == True:
        return abs_fft_tot_X/number, freqaxis
    else:
        return abs_fft_tot_X/number



def moving_window_hv(data_horizontal, data_vertical):

    fft_vertical = fftpack.rfft(data_vertical)
    fft_horizontal = fftpack.rfft(data_horizontal)

    hv = fft_horizontal/fft_vertical
    abs_hv = abs(hv)

    return abs_hv


def kohmachi(signal,freq_array,smooth_coeff):
    x = signal
    f = freq_array
    b = smooth_coeff

    if round(b) != b:
        b = round(b)   # round non integers
    if np.remainder(b,2) == 1:  # if b is odd
        b = b-1  # make it even
    if b < 2:
        b = 2  # "cup" b value
    if b > 100:
        b = 100  # "cap" b value



    l = len(x)
    y = np.zeros(l)

    f_shifted = f/(1+1e-4);
    for i in range(l):
        if (i == 0) or (i == l-1):
            continue  # skip first and last indices for now
        fc = f[i]
        z = f_shifted / fc;

        w = ((np.sin(b * np.log10(z)) / b) / np.log10(z)) ** 4;
        w[np.isnan(w)] = 0;
        y[i] = np.dot(w, x) / sum(w);

    y[0] = y[1]  # calculate first and last indices
    y[-1] = y[-2]

    return y
