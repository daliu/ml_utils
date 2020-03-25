
from ml_utils.eda_utils import calculate_stats

def test_calculate_stats():
    """
    """
    stats_dict = calculate_stats([1,2,3])
    assert stats_dict.get("avg") == 2
    assert stats_dict.get("median") == 2
    assert stats_dict.get("stdev") == 1.0
