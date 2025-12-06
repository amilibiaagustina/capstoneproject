import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

class LFPData:
    """Class to handle LFP data and related operations.
    
    Attributes:
        patient_id (str): The identifier for the patient.
        session_date (str): The date of the recording session.
        lfp_left (np.ndarray): The LFP data from the left hemisphere.
        lfp_right (np.ndarray): The LFP data from the right hemisphere.
        time_lfp (np.ndarray): The time vector for the LFP data.
        task_df (pd.DataFrame): The DataFrame containing task-related data.
        suds_df (pd.DataFrame): The DataFrame containing SUDS ratings.
        fs (int): The sampling frequency of the LFP data.
        phases (Dict[str, np.ndarray]): A dictionary to store phase information.
        phase_markers (Dict[str, float]): A dictionary to store phase marker times.
        artifact_start_sec (float): Seconds to exclude from the beginning of the recording.
        artifact_end_sec (float): Seconds to exclude from the end of the recording.
        is_preprocessed (bool): Flag indicating if the data has been preprocessed.
    
    """

    def __init__(self, patient_id: str, session_date: str, fs: int=250):
        """Initialize the LFPData object with patient ID, session date, and sampling frequency.

            fs (int): The sampling frequency of the LFP data. Default is 250 Hz

        """
        self.patient_id = patient_id
        self.session_date = session_date
        self.fs = fs
        self.is_preprocessed = False

        #data with load_data function
        self.lfp_left = None
        self.lfp_right = None
        self.task_df = None
        self.time_lfp = None
        self.suds_df = None
        self.events_df = None
        self.phase_markers = None
        self.artifact_start_sec = 0
        self.artifact_end_sec = 0

        # Define phases in define_phases function
        self.phases = {}

    def load_data(self, csv_lfp_path: str, csv_task_path: str):
        """Load LFP data and task DataFrame for the patient and session.
        This method should be implemented to load actual data.
        """

        # Load lfp data
        lfp_df= pd.read_csv(csv_lfp_path)
        varNames = lfp_df.columns.tolist()

        # check different naming patterns
        if any("TD_Aic_ONE_THREE_LEFT" in v for v in varNames):
            self.lfp_left  = lfp_df["TD_Aic_ONE_THREE_LEFT"].values
            self.lfp_right = lfp_df["TD_Aic_ONE_THREE_RIGHT"].values
            print("Using ONE_THREE channels")
        elif any("TD_Aic_ZERO_TWO_LEFT" in v for v in varNames):
            self.lfp_left  = lfp_df["TD_Aic_ZERO_TWO_LEFT"].values
            self.lfp_right = lfp_df["TD_Aic_ZERO_TWO_RIGHT"].values
            print("Using ZERO_TWO channels")
        elif any("TD_Other_ZERO_TWO_LEFT" in v for v in varNames):
            self.lfp_left  = lfp_df["TD_Other_ZERO_TWO_LEFT"].values
            self.lfp_right = lfp_df["TD_Other_ZERO_TWO_RIGHT"].values
            print("Using TD_Other_ZERO_TWO channels")
        elif any("TD_Other_ONE_THREE_LEFT" in v for v in varNames):
            self.lfp_left  = lfp_df["TD_Other_ONE_THREE_LEFT"].values
            self.lfp_right = lfp_df["TD_Other_ONE_THREE_RIGHT"].values
            print("Using TD_Other_ONE_THREE channels")
        else:
            raise ValueError("Could not find LFP channels. Available variables: " + ", ".join(varNames))
        time_lfp = np.arange(len(self.lfp_left)) / self.fs

        #load task data
        self.task_df = pd.read_csv(csv_task_path)
        
        print(f" Data loaded for patient {self.patient_id} on {self.session_date}.")
        print (f" LFP duration: {len(self.lfp_left)/self.fs:.1f} seconds.")
        print (f" Task entries: {len(self.task_df)}.")
    

        return self #for method chaining

    def interpolate_nans(self, plot: bool = False):
        """
        Interpolate NaN values in LFP signals
        
        Parameters
        ----------
        plot : bool
            If True, plot NaN regions before interpolation
        """
        if self.lfp_left is None or self.lfp_right is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Find NaN indices
        nan_idx_left = np.where(np.isnan(self.lfp_left))[0]
        nan_idx_right = np.where(np.isnan(self.lfp_right))[0]
        
        # Report NaNs
        n_nan_left = len(nan_idx_left)
        n_nan_right = len(nan_idx_right)
        n_nan_both = len(np.intersect1d(nan_idx_left, nan_idx_right))
        
        print(f"✓ NaN detection:")
        print(f"  Left:  {n_nan_left:,} ({100*n_nan_left/len(self.lfp_left):.2f}%)")
        print(f"  Right: {n_nan_right:,} ({100*n_nan_right/len(self.lfp_right):.2f}%)")
        print(f"  Both:  {n_nan_both:,}")
        
        # Plot if requested
        if plot and (n_nan_left > 0 or n_nan_right > 0):
            self._plot_nan_regions(nan_idx_left, nan_idx_right)
        
        # Interpolate
        self.lfp_left = pd.Series(self.lfp_left).interpolate(method='linear').values
        self.lfp_right = pd.Series(self.lfp_right).interpolate(method='linear').values
        
        # Verify
        remaining_left = np.sum(np.isnan(self.lfp_left))
        remaining_right = np.sum(np.isnan(self.lfp_right))
        
        print(f"✓ Interpolation complete:")
        print(f"  Remaining NaNs - Left: {remaining_left}, Right: {remaining_right}")
        
        return self


    def _plot_nan_regions(self, nan_idx_left, nan_idx_right):
        """
        Plot signals with NaN regions highlighted (internal method)
        """
        import matplotlib.pyplot as plt
        
        # Find continuous NaN segments
        segments_left = self._find_nan_segments(nan_idx_left)
        segments_right = self._find_nan_segments(nan_idx_right)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
        time_full = np.arange(len(self.lfp_left)) / self.fs / 60  # minutes
        
        # Left hemisphere
        axes[0].plot(time_full, self.lfp_left, color='steelblue', lw=0.5, label='LFP signal')
        for start, end in segments_left:
            axes[0].axvspan(start/self.fs/60, end/self.fs/60, color='red', alpha=0.3, label='NaN' if start == segments_left[0][0] else '')
        axes[0].set_title("LFP Left – NaN Regions", fontsize=15)
        axes[0].set_ylabel("Amplitude (µV)", fontsize=14)
        axes[0].legend(loc='upper right', fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Right hemisphere
        axes[1].plot(time_full, self.lfp_right, color='green', lw=0.5, label='LFP signal')
        for start, end in segments_right:
            axes[1].axvspan(start/self.fs/60, end/self.fs/60, color='red', alpha=0.3, label='NaN' if start == segments_right[0][0] else '')
        axes[1].set_title("LFP Right – NaN Regions", fontsize=15)
        axes[1].set_xlabel("Time (minutes)", fontsize=14)
        axes[1].set_ylabel("Amplitude (µV)", fontsize=14)
        axes[1].legend(loc='upper right', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        fig.suptitle(f"Patient {self.patient_id} - NaN Regions", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def _find_nan_segments(nan_indices):
        """
        Find continuous segments of NaN indices
        
        Parameters
        ----------
        nan_indices : np.ndarray
            Array of NaN indices
            
        Returns
        -------
        list of tuples
            Each tuple is (start_idx, end_idx)
        """
        if len(nan_indices) == 0:
            return []
        
        segments = []
        start = nan_indices[0]
        prev = nan_indices[0]
        
        for idx in nan_indices[1:]:
            if idx != prev + 1:
                segments.append((start, prev))
                start = idx
            prev = idx
        
        segments.append((start, prev))
        return segments


    def demean(self):
        """Remove the mean from LFP signals."""
        if self.lfp_left is None or self.lfp_right is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        LFP_Left_dm = self.lfp_left - np.nanmean(self.lfp_left)
        LFP_Right_dm = self.lfp_right - np.nanmean(self.lfp_right)#numpy.nanmean() is a function in the NumPy library used to compute the arithmetic mean of elements in an array, specifically designed to handle and ignore NaN (Not a Number) values.

        print("✓ Mean removed from LFP signals.")
        
        return self
    

    def create_suds_df(self):
        """Create a DataFrame for SUDS ratings from the task DataFrame."""
        if self.task_df is None:
            raise ValueError("Task DataFrame not loaded. Call load_data() first.")
        
        self.suds_df = self.task_df[self.task_df['Event']=='SUDS'].copy()
        # convert Entry column to numeric for SUDS rows
        self.suds_df['Entry'] = pd.to_numeric(self.suds_df['Entry'], errors='coerce')

        # remove any rows where Entry is NaN (non-numeric entries that got coerced)
        self.suds_df = self.suds_df.dropna(subset=['Entry'])

        print(f"SUDS values: {self.suds_df['Entry'].values}")
        print(f"SUDS data type: {self.suds_df['Entry'].dtype}")
            

        return self

    def create_events_df(self):
        """Create a DataFrame for task events from the task DataFrame."""
        if self.task_df is None:
            raise ValueError("Task DataFrame not loaded. Call load_data() first.")
        
        self.events_df = self.task_df[self.task_df['Event']=='Note'].copy()

        print(f"Task events: {self.events_df['Entry'].values}")
            
        return self
        
    def define_phases(self):
        """Define task phases based on time intervals.
        
        Parameters:
            phase_definitions (Dict[str, Tuple[float, float]]): A dictionary where keys are phase names and values are tuples of (start_time, end_time) in seconds.
        """
        if self.task_df is None:
            raise ValueError("Task DataFrame not loaded. Call load_data() first.")
        
        # markers
        task_start = self.task_df[self.task_df['Entry']=='Task Start']['Time_seconds'].values[0]
        exposure = self.task_df[self.task_df['Entry']=='exposure']['Time_seconds'].values[0]
        compulsions = self.task_df[self.task_df['Entry']=='compulsions']['Time_seconds'].values[0]
        relief = self.task_df[self.task_df['Entry']=='relief']['Time_seconds'].values[0]

        # Save markers
        self.phase_markers = {
                'task_start': task_start,
                'exposure': exposure,
                'compulsions': compulsions,
                'relief': relief
            }
        
        total_duration = len(self.lfp_left) / self.fs
            
        # phase definitions with artifact exclusion
        artifact_start = getattr(self, 'artifact_start_sec', 0)
        artifact_end = getattr(self, 'artifact_end_sec', 0)
            
        self.phases = {
                'Baseline': (task_start + artifact_start, exposure),
                'Exposure': (exposure, compulsions),
                'Compulsions': (compulsions, relief),
                'Relief': (relief, total_duration - artifact_end)
            }
            
        print(f"✓ Phases defined:")
        for phase, (start, end) in self.phases.items():
                print(f"  {phase}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
     
        
        return self

    def clean_task_data(self):
        # keep first 7 columns if present (use .iloc to avoid InvalidIndexError)
        if self.task_df.shape[1] >= 7:
            self.task_df = self.task_df.iloc[:, :7].copy()
        else:
            self.task_df = self.task_df.copy()

        # normalize column names (strip leading/trailing spaces)
        self.task_df.columns = self.task_df.columns.str.strip()
        self.task_df.head()
        # Clean all string cells in the DataFrame
        # Replace non-breaking spaces (\xa0) with normal spaces
        # and remove leading/trailing spaces
        self.task_df = self.task_df.map(
            lambda x: str(x).replace('\xa0', ' ').strip() if isinstance(x, str) else x
        )

        self.task_df['Timer'] = self.task_df['Timer'].str.replace('.',':')
        self.task_df['Time_seconds'] = self.task_df['Timer'].apply(timer_to_seconds)

        self.create_events_df()
        self.create_suds_df()

        
        return self
    

            
    def preprocess(self, plot_nans: bool = False, artifact_start_sec: float=0, artifact_end_sec: float=0):
        """Preprocess LFP data by interpolating NaNs and removing the mean.
        
        Parameters:
            artifact_start_sec (float): seconds to exclude from beggining
            artifact_end_sec (float): seconds to exclude from end
            plot_nans (bool): If True, plot NaN regions before interpolation.
        """
        self.interpolate_nans(plot=plot_nans)
        self.demean()
        self.clean_task_data()
        self.artifact_start_sec = artifact_start_sec
        self.artifact_end_sec = artifact_end_sec
        self.lfp_left = self.lfp_left[int(artifact_start_sec*self.fs): len(self.lfp_left)-int(artifact_end_sec*self.fs)]
        self.lfp_right = self.lfp_right[int(artifact_start_sec*self.fs): len(self.lfp_right)-int(artifact_end_sec*self.fs)]
        self.define_phases()
        self.is_preprocessed = True
        print("✓ Preprocessing complete.")
        return self

    
            
def timer_to_seconds(timer_str):
        m, s, ms = timer_str.strip().split(':')
        return int(m)*60 + int(s) + int(ms)/100

        





        