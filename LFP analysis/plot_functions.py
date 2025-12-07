# functions.py
"""
Plotting functions for LFP analysis
All functions receive LFPData and/or BandAnalysis objects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sig
from scipy.stats import gaussian_kde, mannwhitneyu, pearsonr, spearmanr
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
#  RAW LFP SIGNAL WITH ZOOM


def plot_raw_lfp(patient, time_start: float = 0, time_end: float = None,
                 figsize: Tuple = (14, 6)):
    """
    Plot raw LFP signals with optional zoom
    """
    
    time_lfp = np.arange(len(patient.lfp_left)) / patient.fs
    
    if time_end is None:
        time_end = time_lfp[-1]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Left hemisphere
    axes[0].plot(time_lfp, patient.lfp_left, linewidth=0.5, color='steelblue')
    axes[0].set_ylabel('Amplitude (µV)', fontsize=14)
    axes[0].set_title(f'LFP Left Hemisphere - Demeaned', fontsize=16, fontweight='bold')
    axes[0].set_ylim([-30, 30])
    axes[0].set_xlim(time_start, time_end)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Add phase markers if defined
    if patient.phases is not None:
        for phase_name, (start, end) in patient.phases.items():
            if start >= time_start and start <= time_end:
                axes[0].axvline(start, color='red', linestyle='--', alpha=0.5)
                axes[0].text(start, 25, phase_name, fontsize=12)
    
    # Right hemisphere
    axes[1].plot(time_lfp, patient.lfp_right, linewidth=0.5, color='crimson')
    axes[1].set_ylabel('Amplitude (µV)', fontsize=14)
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_title(f'LFP Right Hemisphere - Demeaned', fontsize=16, fontweight='bold')
    axes[1].set_ylim([-30, 30])
    axes[1].set_xlim(time_start, time_end)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=12)
    
    # Add phase markers
    if patient.phases is not None:
        for phase_name, (start, end) in patient.phases.items():
            if start >= time_start and start <= time_end:
                axes[1].axvline(start, color='red', linestyle='--', alpha=0.5)
                axes[1].text(start, 25, phase_name, fontsize=12)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# SPECTROGRAMS


def plot_spectrograms(patient, time_start: float = 0, time_end: float = None,
                     window_sec: float = 5, freq_max: float = 30,
                     figsize: Tuple = (14, 8)):
    """
    Plot spectrograms for both hemispheres
    """
    if not patient.is_preprocessed:
        raise ValueError("Patient data must be preprocessed first")
    
    # Compute spectrograms
    window = int(window_sec * patient.fs)
    noverlap = int(0.75 * window)
    nfft = max(256, 2**int(np.ceil(np.log2(window))))
    
    # Left hemisphere
    freqs_left, times_left, Sxx_left = sig.spectrogram(
        patient.lfp_left, fs=patient.fs, window='hamming',
        nperseg=window, noverlap=noverlap, nfft=nfft
    )
    Sxx_left_db = 10 * np.log10(np.abs(Sxx_left) + 1e-10)
    
    # Right hemisphere
    freqs_right, times_right, Sxx_right = sig.spectrogram(
        patient.lfp_right, fs=patient.fs, window='hamming',
        nperseg=window, noverlap=noverlap, nfft=nfft
    )
    Sxx_right_db = 10 * np.log10(np.abs(Sxx_right) + 1e-10)
    
    if time_end is None:
        time_end = times_left[-1]
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Left hemisphere
    im1 = axes[0].imshow(
        Sxx_left_db, aspect='auto', origin='lower',
        extent=[times_left[0], times_left[-1], freqs_left[0], freqs_left[-1]],
        cmap='plasma', vmin=-10, vmax=30
    )
    axes[0].set_ylabel('Frequency (Hz)', fontsize=14)
    axes[0].set_title('LFP Left Spectrogram', fontsize=16, fontweight='bold')
    axes[0].set_ylim(0, freq_max)
    axes[0].set_xlim(time_start, time_end)
    axes[0].tick_params(axis='both', labelsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Power (dB)', fontsize=12)
    cbar1.ax.tick_params(labelsize=12)
    
    # Add phase markers
    if patient.phases is not None:
        for phase_name, (start, end) in patient.phases.items():
            if start >= time_start and start <= time_end:
                axes[0].axvline(start, color='white', linestyle='--', alpha=0.7, linewidth=2)
    
    # Right hemisphere
    im2 = axes[1].imshow(
        Sxx_right_db, aspect='auto', origin='lower',
        extent=[times_right[0], times_right[-1], freqs_right[0], freqs_right[-1]],
        cmap='plasma', vmin=-10, vmax=30
    )
    axes[1].set_ylabel('Frequency (Hz)', fontsize=14)
    axes[1].set_xlabel('Time (seconds)', fontsize=14)
    axes[1].set_title('LFP Right Spectrogram', fontsize=16, fontweight='bold')
    axes[1].set_ylim(0, freq_max)
    axes[1].set_xlim(time_start, time_end)
    axes[1].tick_params(axis='both', labelsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Power (dB)', fontsize=12)
    cbar2.ax.tick_params(labelsize=12)
    
    # Add phase markers
    if patient.phases is not None:
        for phase_name, (start, end) in patient.phases.items():
            if start >= time_start and start <= time_end:
                axes[1].axvline(start, color='white', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    plt.show()


# ============================================================
#  BAND POWER CONTINUOUS 

def plot_band_continuous(analysis, band_name: str = 'Theta',
                        highlight_phases: bool = True, time_start: float = 0,
                        time_end: float = None, figsize: Tuple = (16, 6)):
    """
    Plot band power continuously over time (simple version without SUDS)
    """
    if band_name not in analysis.results:
        raise ValueError(f"Band '{band_name}' not calculated yet")
    
    # Get continuous data
    band_left, times = analysis.get_band_continuous(band_name, 'left')
    band_right, _ = analysis.get_band_continuous(band_name, 'right')
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Highlight phases if requested
    if highlight_phases and analysis.lfp_data.phases is not None:
        phase_colors = {
            'Baseline': 'lightblue',
            'Exposure': 'lightcoral',
            'Compulsions': 'yellow',
            'Relief': 'lightgreen'
        }
        
        for phase_name, (start, end) in analysis.lfp_data.phases.items():
            for ax in axes:
                ax.axvspan(start, end, alpha=0.2, 
                          color=phase_colors.get(phase_name, 'gray'),
                          label=phase_name)
    
    # Left hemisphere
    axes[0].plot(times, band_left, 'o-', markersize=2, linewidth=1,
                color='steelblue', label=f'{band_name} power')
    axes[0].set_ylabel(f'{band_name} power (µV²)', fontsize=14)
    axes[0].set_xlim(time_start, time_end if time_end is not None else times[-1])
    axes[0].set_title(f'Left hemisphere - {band_name} band power', fontsize=16, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Add legend only once for phases
    if highlight_phases and analysis.lfp_data.phases is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    # Right hemisphere
    axes[1].plot(times, band_right, 'o-', markersize=2, linewidth=1,
                color='crimson', label=f'{band_name} power')
    axes[1].set_ylabel(f'{band_name} power (µV²)', fontsize=14)
    axes[1].set_xlabel('Time (seconds)', fontsize=14)
    axes[1].set_xlim(time_start, time_end if time_end is not None else times[-1])
    axes[1].set_title(f'Right hemisphere - {band_name} band power', fontsize=16, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.show()


# ============================================================
#  PSD BY PHASE

def plot_psd_by_phase(patient, band_to_highlight: str = 'Theta', 
                      freq_max: float = 30, figsize: Tuple = (10, 10)):
    """
    Plot Power Spectral Density for each phase
    """
    if not patient.is_preprocessed:
        raise ValueError("Patient data must be preprocessed first")
    
    if patient.phases is None:
        raise ValueError("Phases must be defined")
    
    # Get frequency band range
    band_ranges = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    freq_band = band_ranges.get(band_to_highlight, (4, 8))
    
    # Phase colors
    colors = {
        'Baseline': 'darkslateblue',
        'Exposure': 'firebrick',
        'Compulsions': 'goldenrod',
        'Relief': 'olivedrab'
    }
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Left hemisphere
    for phase_name, (start_time, end_time) in patient.phases.items():
        start_idx = int(start_time * patient.fs)
        end_idx = int(end_time * patient.fs)
        phase_signal = patient.lfp_left[start_idx:end_idx]
        
        # Compute PSD
        freqs, psd = sig.welch(phase_signal, fs=patient.fs, nperseg=2**10)
        
        # Plot
        axes[0].plot(freqs, 10*np.log10(psd), linewidth=2,
                    color=colors[phase_name], label=phase_name, alpha=0.8)
    
    # Highlight band
    axes[0].axvspan(freq_band[0], freq_band[1], alpha=0.2, color='blue',
                   label=f'{band_to_highlight} band')
    
    axes[0].set_xlabel('Frequency (Hz)', fontsize=14)
    axes[0].set_ylabel('Power (dB)', fontsize=14)
    axes[0].set_title('Left hemisphere - PSD by phase', fontsize=16, fontweight='bold')
    axes[0].set_xlim(0, freq_max)
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Right hemisphere
    for phase_name, (start_time, end_time) in patient.phases.items():
        start_idx = int(start_time * patient.fs)
        end_idx = int(end_time * patient.fs)
        phase_signal = patient.lfp_right[start_idx:end_idx]
        
        # Compute PSD
        freqs, psd = sig.welch(phase_signal, fs=patient.fs, nperseg=2**10)
        
        # Plot
        axes[1].plot(freqs, 10*np.log10(psd), linewidth=2,
                    color=colors[phase_name], label=phase_name, alpha=0.8)
    
    # Highlight band
    axes[1].axvspan(freq_band[0], freq_band[1], alpha=0.2, color='blue',
                   label=f'{band_to_highlight} band')
    
    axes[1].set_xlabel('Frequency (Hz)', fontsize=14)
    axes[1].set_ylabel('Power (dB)', fontsize=14)
    axes[1].set_title('Right hemisphere - PSD by phase', fontsize=16, fontweight='bold')
    axes[1].set_xlim(0, freq_max)
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.show()


# ============================================================
#  BAND POWER DISTRIBUTIONS BY PHASE

def plot_band_distributions(analysis, band_name: str = 'Theta',
                           xlim: Tuple = (0, 20), figsize: Tuple = (16, 6)):
    """
    Plot band power distributions by phase with KDE and Mann-Whitney tests
    """
    if band_name not in analysis.results:
        raise ValueError(f"Band '{band_name}' not calculated yet")
    
    # Get data by phase
    band_left = analysis.get_band_by_phase(band_name, 'left')
    band_right = analysis.get_band_by_phase(band_name, 'right')
    
    phases_list = list(band_left.keys())
    colors = ['darkslateblue', 'firebrick', 'goldenrod', 'olivedrab']
    
    # Compute Mann-Whitney tests
    comp_pairs = [
        ('Baseline', 'Relief'),
        ('Exposure', 'Compulsions'),
        ('Exposure', 'Relief'),
        ('Compulsions', 'Relief')
    ]
    
    comparisons_left = []
    comparisons_right = []
    
    for phase1, phase2 in comp_pairs:
        # Left
        if len(band_left[phase1]) > 0 and len(band_left[phase2]) > 0:
            u_stat_l, p_val_l = mannwhitneyu(band_left[phase1], band_left[phase2], 
                                            alternative='two-sided')
            if p_val_l < 0.001:
                sig_l = '***'
            elif p_val_l < 0.01:
                sig_l = '**'
            elif p_val_l < 0.05:
                sig_l = '*'
            else:
                sig_l = 'n.s.'
            comparisons_left.append((phase1, phase2, u_stat_l, p_val_l, sig_l))
        
        # Right
        if len(band_right[phase1]) > 0 and len(band_right[phase2]) > 0:
            u_stat_r, p_val_r = mannwhitneyu(band_right[phase1], band_right[phase2],
                                            alternative='two-sided')
            if p_val_r < 0.001:
                sig_r = '***'
            elif p_val_r < 0.01:
                sig_r = '**'
            elif p_val_r < 0.05:
                sig_r = '*'
            else:
                sig_r = 'n.s.'
            comparisons_right.append((phase1, phase2, u_stat_r, p_val_r, sig_r))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for ax, band_data, title, comparisons in zip(
        axes,
        [band_left, band_right],
        ['Left hemisphere', 'Right hemisphere'],
        [comparisons_left, comparisons_right]
    ):
        for phase, color in zip(phases_list, colors):
            data = band_data[phase]
            
            if len(data) == 0:
                continue
            
            # KDE
            kde = gaussian_kde(data)
            x_range = np.linspace(xlim[0], xlim[1], 300)
            density = kde(x_range)
            
            # Stats
            median_val = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            # Plot curve
            ax.plot(x_range, density, color=color, linewidth=2.5,
                   label=f'{phase} (M={median_val:.1f}, IQR={iqr:.1f})', alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.2, color=color)
            
            # Median line
            ax.axvline(median_val, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Text box with significant comparisons
        textstr = 'Mann-Whitney U tests:\n'
        sig_found = False
        for phase1, phase2, u_stat, p_val, sig in comparisons:
            if sig != 'n.s.':
                textstr += f'{phase1} vs {phase2}: p={p_val:.4f} {sig}\n'
                sig_found = True
        
        if not sig_found:
            textstr += 'No significant differences\n(all p > 0.05)'
        
        # Add box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, textstr.strip(), transform=ax.transAxes,
               fontsize=12, verticalalignment='top', horizontalalignment='right',
               bbox=props, family='monospace')
        
        ax.set_xlabel(f'{band_name} power (µV²)', fontsize=14)
        ax.set_ylabel('Probability density', fontsize=14)
        ax.set_title(f'{title}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='center right')
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(xlim)
        ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.show()


# ============================================================
#  BAND POWER WITH SUDS ALIGNMENT (TIMESERIES)


def plot_band_suds_timeseries(analysis, band_name: str = 'Theta',
                              figsize: Tuple = (16, 8)):
    """
    Plot band power over time with aligned SUDS scores
    """
    if band_name not in analysis.results:
        raise ValueError(f"Band '{band_name}' not calculated yet")
    
    # Get continuous data
    band_left, times = analysis.get_band_continuous(band_name, 'left')
    band_right, _ = analysis.get_band_continuous(band_name, 'right')
    
    # Align with SUDS
    _, suds_aligned = analysis.align_with_suds(band_name, 'left')
    
    # Get actual SUDS for overlay
    suds_df = analysis.lfp_data.suds_df
    suds_times = suds_df['Time_seconds'].values
    suds_values = suds_df['Entry'].values
    
    # Helper function for ylim without outliers
    def get_ylim_without_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        data_clean = data[data <= upper_bound]
        return np.max(data_clean) * 1.1
    
    ylim_left = get_ylim_without_outliers(band_left)
    ylim_right = get_ylim_without_outliers(band_right)
    ylim_max = max(ylim_left, ylim_right)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Left hemisphere
    axes[0].plot(times, band_left, 'o-', markersize=3, linewidth=0.5,
                alpha=0.6, color='steelblue', label=f'{band_name} power')
    
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(times, suds_aligned, 's', markersize=4,
                 color='seagreen', alpha=0.5, label='SUDS (aligned)')
    ax0_twin.set_ylabel('SUDS score', fontsize=14, color='seagreen')
    ax0_twin.tick_params(axis='y', labelcolor='seagreen', labelsize=12)
    ax0_twin.set_ylim(2, 8)
    
    # Actual SUDS points
    ax0_twin.plot(suds_times, suds_values, 'd', color='seagreen', markersize=10,
                 markeredgecolor='black', markeredgewidth=1, label='SUDS (actual)')
    
    axes[0].set_ylabel(f'{band_name} power (µV²)', fontsize=14)
    axes[0].set_title(f'Left hemisphere - {band_name} power with SUDS alignment',
                     fontsize=16, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=12)
    ax0_twin.legend(loc='upper right', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, ylim_max)
    axes[0].tick_params(axis='both', labelsize=12)
    
    # Right hemisphere
    axes[1].plot(times, band_right, 'o-', markersize=3, linewidth=0.5,
                alpha=0.6, color='crimson', label=f'{band_name} power')
    
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(times, suds_aligned, 's', markersize=4,
                 color='seagreen', alpha=0.5, label='SUDS (aligned)')
    ax1_twin.set_ylabel('SUDS score', fontsize=14, color='seagreen')
    ax1_twin.tick_params(axis='y', labelcolor='seagreen', labelsize=12)
    ax1_twin.set_ylim(2, 8)
    
    # Actual SUDS points
    ax1_twin.plot(suds_times, suds_values, 'd', markersize=10,
                 markeredgecolor='black', markeredgewidth=1, 
                 color='seagreen', label='SUDS (actual)')
    
    axes[1].set_ylabel(f'{band_name} power (µV²)', fontsize=14)
    axes[1].set_xlabel('Time (seconds)', fontsize=14)
    axes[1].set_title(f'Right hemisphere - {band_name} power with SUDS alignment',
                     fontsize=16, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=12)
    ax1_twin.legend(loc='upper right', fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, ylim_max)
    axes[1].tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# BAND VS SUDS CORRELATION


def plot_suds_correlation(analysis, band_name: str = 'Theta',
                         remove_outliers: bool = True,
                         figsize: Tuple = (14, 6)):
    """
    Plot correlation between band power and SUDS scores
    """
    if band_name not in analysis.results:
        raise ValueError(f"Band '{band_name}' not calculated yet")
    
    # Get data aligned with SUDS
    band_left, suds_left = analysis.align_with_suds(band_name, 'left')
    band_right, suds_right = analysis.align_with_suds(band_name, 'right')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for ax, band_data, suds_data, title in zip(
        axes,
        [band_left, band_right],
        [suds_left, suds_right],
        ['Left hemisphere', 'Right hemisphere']
    ):
        # Remove outliers if requested
        if remove_outliers:
            q1 = np.percentile(band_data, 25)
            q3 = np.percentile(band_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mask = (band_data >= lower_bound) & (band_data <= upper_bound)
            band_clean = band_data[mask]
            suds_clean = suds_data[mask]
            
            removed = len(band_data) - len(band_clean)
            title_suffix = f'(Outliers removed: {removed}, n={len(band_clean)})'
        else:
            band_clean = band_data
            suds_clean = suds_data
            title_suffix = f'(n={len(band_clean)})'
        
        # Scatter plot
        ax.scatter(suds_clean, band_clean, alpha=0.5, s=20, color='steelblue')
        
        # Calculate correlations
        r_pearson, p_pearson = pearsonr(suds_clean, band_clean)
        r_spearman, p_spearman = spearmanr(suds_clean, band_clean)
        
        # Regression line
        z = np.polyfit(suds_clean, band_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(suds_clean.min(), suds_clean.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
        
        # Significance stars
        if p_spearman < 0.001:
            sig = '***'
        elif p_spearman < 0.01:
            sig = '**'
        elif p_spearman < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        
        # Text box with stats
        textstr = f'Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}\n'
        textstr += f'Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f} {sig}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', bbox=props)
        
        ax.set_xlabel('SUDS score', fontsize=14)
        ax.set_ylabel(f'{band_name} power (µV²)', fontsize=14)
        ax.set_title(f'{title}\n{title_suffix}', fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(2.5, 7.5)
        ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.show()

# ============================================================
#  INTERACTIVE PLOT WITH SUDS AND SPECTROGRAMS
def plot_suds_and_spectrograms(patient, window_sec: float = 2.0):
    """
    Interactive plot with SUDS and spectrograms using Plotly
    
    Parameters
    ----------
    patient : LFPData
        LFPData object with preprocessed data
    window_sec : float
        Window size for spectrogram in seconds
    """
    
    # Get SUDS data
    suds_df = patient.suds_df
    events_df = patient.events_df
    
    # Get phase markers
    task_start_time = patient.phase_markers['task_start']
    exposure_time = patient.phase_markers['exposure']
    compulsions_time = patient.phase_markers['compulsions']
    relief_time = patient.phase_markers['relief']
    
    # Compute spectrograms
    window = int(window_sec * patient.fs)
    noverlap = int(0.75 * window)
    nfft = max(256, 2**int(np.ceil(np.log2(window))))
    
    freqs_left, times_left, Sxx_left = sig.spectrogram(
        patient.lfp_left, fs=patient.fs, window='hamming',
        nperseg=window, noverlap=noverlap, nfft=nfft
    )
    Sxx_left_db = 10 * np.log10(np.abs(Sxx_left) + 1e-10)
    
    freqs_right, times_right, Sxx_right = sig.spectrogram(
        patient.lfp_right, fs=patient.fs, window='hamming',
        nperseg=window, noverlap=noverlap, nfft=nfft
    )
    Sxx_right_db = 10 * np.log10(np.abs(Sxx_right) + 1e-10)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.3, 0.35, 0.35],
        subplot_titles=("SUDS Over Time", "LFP Left Spectrogram", "LFP Right Spectrogram")
    )
    
    # Row 1: SUDS
    fig.add_trace(
        go.Scatter(
            x=suds_df['Time_seconds'],
            y=suds_df['Entry'],
            mode='lines+markers',
            name='SUDS',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
        ),
        row=1, col=1
    )
    
    # Clinical notes
    if events_df is not None and len(events_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=events_df['Time_seconds'],
                y=[suds_df['Entry'].max() + 0.5] * len(events_df),
                mode='markers',
                name='Clinical Notes',
                marker=dict(size=10, color='#C78E37', symbol='circle'),
                text=events_df['Entry'],
                hovertemplate='<b>Note:</b> %{text}<br>Time: %{x:.1f}s<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Row 2: Left spectrogram
    fig.add_trace(
        go.Heatmap(
            z=Sxx_left_db,
            x=times_left,
            y=freqs_left,
            colorscale='Plasma',
            zmin=-10, zmax=30,
            colorbar=dict(title='Power (dB)', len=0.25, y=0.5),
            name='LFP Left',
        ),
        row=2, col=1
    )
    
    # Row 3: Right spectrogram
    fig.add_trace(
        go.Heatmap(
            z=Sxx_right_db,
            x=times_right,
            y=freqs_right,
            colorscale='Plasma',
            zmin=-10, zmax=30,
            colorbar=dict(title='Power (dB)', len=0.25, y=0.15),
            name='LFP Right'
        ),
        row=3, col=1
    )
    
    # Add phase markers - Row 1 with annotations at bottom
    fig.add_vline(x=task_start_time, line_width=3, line_dash='dash', line_color='green',
                  annotation_text='Task Start', annotation_position='bottom', row=1, col=1)
    fig.add_vline(x=exposure_time, line_width=3, line_dash='dash', line_color='red',
                  annotation_text='Exposure', annotation_position='bottom', row=1, col=1)
    fig.add_vline(x=compulsions_time, line_width=3, line_dash='dash', line_color='orange',
                  annotation_text='Compulsions', annotation_position='bottom', row=1, col=1)
    fig.add_vline(x=relief_time, line_width=3, line_dash='dash', line_color='blue',
                  annotation_text='Relief', annotation_position='bottom', row=1, col=1)
    
    # Rows 2 and 3 without annotations
    for time_val, color in [(task_start_time, 'green'), (exposure_time, 'red'), 
                            (compulsions_time, 'orange'), (relief_time, 'blue')]:
        fig.add_vline(x=time_val, line_width=3, line_dash='dash', line_color=color, row=2, col=1)
        fig.add_vline(x=time_val, line_width=3, line_dash='dash', line_color=color, row=3, col=1)
    
    # Update axes
    fig.update_yaxes(title_text='SUDS Score', row=1, col=1)
    fig.update_yaxes(title_text='Frequency (Hz)', range=[0, 30], row=2, col=1)
    fig.update_yaxes(title_text='Frequency (Hz)', range=[0, 30], row=3, col=1)
    fig.update_xaxes(title_text='Time (seconds)', row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title=dict(
            text=f'SUDS and LFP Spectrograms - Patient {patient.patient_id} ({patient.session_date})',
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial', color='black')
        ),
        template='plotly_white',
        showlegend=False
    )
    
    fig.show()