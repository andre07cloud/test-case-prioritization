import matplotlib.pyplot as plt


def plot_boxplot(data, num_tests):
    output_path = "data/boxplot_result"
    # Create the box plot
    plt.figure(figsize=(10, 8))
    plt.boxplot(data, labels=["Random selection", "Optimized selection"])
    plt.title(f'Box Plot of {num_tests} Test Case Prioritization')
    plt.ylabel('Average Diversity')
    plt.xlabel('Selector')
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.savefig(f'{output_path}/{num_tests}_av_diversity_boxplot.png')
    plt.show()