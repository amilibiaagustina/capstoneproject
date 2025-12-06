import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from classes.LFPData import LFPData


class BandAnalysis:
    """
    Frequency band power analysis for LFP data
    
    Calculates band power continuously across the entire signal,
    then provides methods to split by phase when needed.
    
    Attributes
    ----------
    lfp_data : LFPData
        Preprocessed LFP data object
    freq_bands : dict
        Frequency bands: {'Theta': (4, 8), ...}
    window_sec : float
        Window size in seconds
    results : dict
        Continuous results: {band: {'left': [...], 'right': [...], 'times': [...]}}
    """
    
    def __init__(self, lfp_data: LFPData, 
                 freq_bands: Dict[str, Tuple[float, float]] = None,
                 window_sec: float = 5.0):
        """
        Initialize band analysis
        
        Parameters
        ----------
        lfp_data : LFPData
            Preprocessed LFP data object
        freq_bands : dict, optional
            Custom frequency bands. Default: Theta, Alpha, Beta
        window_sec : float
            Window size in seconds
        """
        if not lfp_data.is_preprocessed:
            raise ValueError("LFP data must be preprocessed first")
        
        self.lfp_data = lfp_data
        self.window_sec = window_sec
        
        # Default frequency bands
        if freq_bands is None:
            self.freq_bands = {
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30)
            }
        else:
            self.freq_bands = freq_bands
        
        self.results = {}
    
    def calculate_band_power_continuous(self, freq_band: Tuple[float, float],
                                       signal: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Calculate band power continuously across entire signal
        
        Parameters
        ----------
        freq_band : tuple
            (low_freq, high_freq)
        signal : np.ndarray
            LFP signal (left or right hemisphere)
        
        Returns
        -------
        powers : list
            Band power for each window
        times : list
            Center time of each window
        """
        window_samples = int(self.window_sec * self.lfp_data.fs)
        n_windows = len(signal) // window_samples
        
        powers = []
        times = []
        
        for w in range(n_windows):
            window_start = w * window_samples
            window_end = window_start + window_samples
            segment = signal[window_start:window_end]
            
            # Compute PSD with Welch
            freqs, psd = sig.welch(segment, fs=self.lfp_data.fs,
                                   nperseg=min(len(segment), 512))
            
            # Integrate power in frequency band
            band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            power = np.trapz(psd[band_mask], freqs[band_mask])
            
            powers.append(power)
            
            # Time at center of window
            time_center = (window_start + window_end) / 2 / self.lfp_data.fs
            times.append(time_center)
        
        return powers, times
    
    def calculate_all_bands(self):
        """
        Calculate power for all frequency bands continuously
        
        Results stored in self.results:
        {
            'Theta': {
                'left': [power values],
                'right': [power values],
                'times': [time points]
            },
            ...
        }
        """
        print(f"\n{'='*60}")
        print(f"Calculating continuous band power for {len(self.freq_bands)} bands...")
        print(f"{'='*60}")
        
        for band_name, freq_range in self.freq_bands.items():
            print(f"  Processing {band_name} ({freq_range[0]}-{freq_range[1]} Hz)...")
            
            # Calculate for left hemisphere
            powers_left, times_left = self.calculate_band_power_continuous(
                freq_range, self.lfp_data.lfp_left
            )
            
            # Calculate for right hemisphere
            powers_right, times_right = self.calculate_band_power_continuous(
                freq_range, self.lfp_data.lfp_right
            )
            
            # Store results
            self.results[band_name] = {
                'left': np.array(powers_left),
                'right': np.array(powers_right),
                'times': np.array(times_left)  # Same for both hemispheres
            }
        
        print(f"\n✓ All bands calculated")
        print(f"  Total windows: {len(self.results[list(self.freq_bands.keys())[0]]['times'])}")
        print(f"{'='*60}\n")
        
        return self
    
    def get_band_continuous(self, band_name: str, hemisphere: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get continuous band power and times
        
        Parameters
        ----------
        band_name : str
            Band name ('Theta', 'Alpha', 'Beta')
        hemisphere : str
            'left' or 'right'
        
        Returns
        -------
        powers : np.ndarray
            Power values
        times : np.ndarray
            Time points
        """
        if band_name not in self.results:
            raise ValueError(f"Band '{band_name}' not calculated yet")
        
        powers = self.results[band_name][hemisphere]
        times = self.results[band_name]['times']
        
        return powers, times
    
    def get_band_by_phase(self, band_name: str, hemisphere: str = 'left') -> Dict[str, np.ndarray]:
        """
        Split band power by experimental phases
        
        Parameters
        ----------
        band_name : str
            Band name
        hemisphere : str
            'left' or 'right'
        
        Returns
        -------
        dict : {phase_name: power_array}
        
        Examples
        --------
        >>> theta_by_phase = analysis.get_band_by_phase('Theta', 'left')
        >>> baseline_power = theta_by_phase['Baseline']
        """
        if self.lfp_data.phases is None:
            raise ValueError("Phases not defined in LFPData")
        
        powers, times = self.get_band_continuous(band_name, hemisphere)
        
        # Split by phase
        phase_data = {}
        for phase_name, (start_time, end_time) in self.lfp_data.phases.items():
            mask = (times >= start_time) & (times < end_time)
            phase_data[phase_name] = powers[mask]
        
        return phase_data
    
    def align_with_suds(self, band_name: str, hemisphere: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        """
        Align band power with SUDS scores (nearest neighbor)
        
        Parameters
        ----------
        band_name : str
            Band name
        hemisphere : str
            'left' or 'right'
        
        Returns
        -------
        powers : np.ndarray
            Band power values
        suds : np.ndarray
            Aligned SUDS scores
        """
        powers, times = self.get_band_continuous(band_name, hemisphere)
        
        # Get SUDS data
        suds_df = self.lfp_data.suds_df
        suds_times = suds_df['Time_seconds'].values
        suds_values = suds_df['Entry'].values
        
        # Find nearest SUDS for each window
        suds_aligned = []
        for time_point in times:
            distances = np.abs(suds_times - time_point)
            nearest_idx = np.argmin(distances)
            suds_aligned.append(suds_values[nearest_idx])
        
        return powers, np.array(suds_aligned)
    
    def print_summary(self):
        """Print summary of continuous results"""
        if not self.results:
            print("No results yet. Run calculate_all_bands() first.")
            return
        
        print(f"\n{'='*60}")
        print("BAND ANALYSIS SUMMARY (Continuous)")
        print(f"{'='*60}")
        
        n_windows = len(self.results[list(self.freq_bands.keys())[0]]['times'])
        duration = self.results[list(self.freq_bands.keys())[0]]['times'][-1]
        
        print(f"\nTotal windows: {n_windows}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
        print(f"Window size: {self.window_sec}s")
        
        for band_name in self.freq_bands.keys():
            mean_left = np.mean(self.results[band_name]['left'])
            mean_right = np.mean(self.results[band_name]['right'])
            
            print(f"\n{band_name}:")
            print(f"  Left:  mean={mean_left:8.2f} µV²  "
                  f"range=[{np.min(self.results[band_name]['left']):.2f}, "
                  f"{np.max(self.results[band_name]['left']):.2f}]")
            print(f"  Right: mean={mean_right:8.2f} µV²  "
                  f"range=[{np.min(self.results[band_name]['right']):.2f}, "
                  f"{np.max(self.results[band_name]['right']):.2f}]")
        
        print(f"{'='*60}\n")
    
    def print_summary_by_phase(self):
        """Print summary split by phases"""
        if not self.results:
            print("No results yet. Run calculate_all_bands() first.")
            return
        
        if self.lfp_data.phases is None:
            print("Phases not defined.")
            return
        
        print(f"\n{'='*60}")
        print("BAND ANALYSIS SUMMARY (By Phase)")
        print(f"{'='*60}")
        
        for band_name in self.freq_bands.keys():
            print(f"\n{band_name}:")
            
            for hemi in ['left', 'right']:
                print(f"  {hemi.capitalize()}:")
                phase_data = self.get_band_by_phase(band_name, hemi)
                
                for phase_name, powers in phase_data.items():
                    if len(powers) > 0:
                        print(f"    {phase_name:12s}: {len(powers):3d} windows | "
                              f"mean={np.mean(powers):6.2f} µV² | "
                              f"std={np.std(powers):5.2f}")
        
        print(f"{'='*60}\n")