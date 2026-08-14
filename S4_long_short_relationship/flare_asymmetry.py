import numpy as np

def calculate_asymmetric_features(signal):
    
    """
    Parameters:
        signal (array): Input time series.
 
    Returns:
        dict: A dictionary containing the calculated features.
    """

    positive_peaks = signal[signal > 0]
    negative_peaks = signal[signal < 0]

    if len(positive_peaks) > 0 and len(negative_peaks) > 0:
        fluctuation_pos = np.mean(positive_peaks**2)
        fluctuation_neg = np.mean(negative_peaks**2)
        fluctuation_ratio = np.log(fluctuation_pos/fluctuation_neg)

    else:
        fluctuation_ratio = np.nan
        
    return fluctuation_ratio