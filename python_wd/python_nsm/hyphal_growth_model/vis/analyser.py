# vis/analyser.py

# Imports
import matplotlib.pyplot as plt # Plotting library
from core.mycel import Mycel # Mycell class to extract simulation state
from dataclasses import dataclass, field # For defining simple data containers

@dataclass
class SimulationStats:
    """
    Container for time-series statistics of a Mycel sim.
    Fields accumulate metrics at each update call.
    """
    times: list = field(default_factory=list) # Sim time points
    tip_counts: list = field(default_factory=list) # No. active tips
    total_sections: list = field(default_factory=list) # Total segments in network
    avg_lengths: list = field(default_factory=list) # Avg. segment length
    avg_ages: list = field(default_factory=list) # Avg. segment age

    def update(self, mycel: Mycel):
        """
        Record current sim metrics.
        Args:
            mycel (Mycel): the sim instance to sample.
        """
        tips = mycel.get_tips() # List of current active tips
        all_sections = mycel.get_all_segments() # List of all segments

        self.times.append(mycel.time) # Append current sim time
        self.tip_counts.append(len(tips)) # Append count of active tips
        self.total_sections.append(len(all_sections)) # Append total no. segments
        self.avg_lengths.append( # Compute and append avg. segment length
            sum(s.length for s in all_sections) / len(all_sections)
        )
        self.avg_ages.append( # Compute and append avg. segment age
            sum(s.age for s in all_sections) / len(all_sections)
        )

def plot_stats(stats: SimulationStats, save_path=None):
    """
    Plot time-series charts for sim stats.
    Args:
        stats (SimulationStats): collected metrics.
        save_path (str, optional): File path to save figure. 
                                   If none, display interactively.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True) # Create a figure w/ 3 stacjed subplots sharing the x-axis (time)

    # Top plot: tip counts and total sections over time
    axs[0].plot(stats.times, stats.tip_counts, label="Tips", color="red") # Tip count
    axs[0].plot(stats.times, stats.total_sections, label="Total Sections", color="green") # Total segment length
    axs[0].legend() # Show legend
    axs[0].set_ylabel("Count") # Y-axis label

    # Middle plot: avg. segment length over time
    axs[1].plot(stats.times, stats.avg_lengths, label="Avg Length") # Avg. lengths of segments
    axs[1].set_ylabel("Length") # Y-axis label

    # Bottom plot: avg. segment age over time
    axs[2].plot(stats.times, stats.avg_ages, label="Avg Age", color="purple") # Plot avg. age
    axs[2].set_ylabel("Age") # Y-axis label
    axs[2].set_xlabel("Time") # X-axis label on bottom subplot

    plt.tight_layout() # Improve layout to prevent overlap

    if save_path:
        plt.savefig(save_path) # Save figure to file and close to free memory
        plt.close()
    else:
        plt.show() # Display plots in a window
