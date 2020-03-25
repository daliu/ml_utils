import statistics


def calculate_stats(lst, print_results=False):
    """
    Args:
        lst (list): numeric values
        print_results (bool): whether to print to console

    Returns:
        (dict): basic statistical results
    """
    avg = stats.mean(lst)
    median = stats.median(lst)
    stdev = stats.stdev(lst)
    stats_summary = {"avg": avg,
                     "median": median,
                     "std_dev": std_dev}

    if print_results:
        print(stats_summary)

    return stats_summary
