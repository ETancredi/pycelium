# vis/analyser.py

# Imports
import matplotlib.pyplot as plt
from core.mycel import Mycel
from dataclasses import dataclass, field

@dataclass
class SimulationStats:
    times: list = field(default_factory=list)
    tip_counts: list = field(default_factory=list)
    total_sections: list = field(default_factory=list)
    avg_lengths: list = field(default_factory=list)
    avg_ages: list = field(default_factory=list)

    def update(self, mycel: Mycel):
        tips = mycel.get_tips()
        all_sections = mycel.get_all_segments()

        self.times.append(mycel.time)
        self.tip_counts.append(len(tips))
        self.total_sections.append(len(all_sections))
        self.avg_lengths.append(
            sum(s.length for s in all_sections) / len(all_sections)
        )
        self.avg_ages.append(
            sum(s.age for s in all_sections) / len(all_sections)
        )

def plot_stats(stats: SimulationStats, save_path=None):
    """Plot stats over time. Save to disk if save_path is provided."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(stats.times, stats.tip_counts, label="Tips", color="red")
    axs[0].plot(stats.times, stats.total_sections, label="Total Sections", color="green")
    axs[0].legend()
    axs[0].set_ylabel("Count")

    axs[1].plot(stats.times, stats.avg_lengths, label="Avg Length")
    axs[1].set_ylabel("Length")

    axs[2].plot(stats.times, stats.avg_ages, label="Avg Age", color="purple")
    axs[2].set_ylabel("Age")
    axs[2].set_xlabel("Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
