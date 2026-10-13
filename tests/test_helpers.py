import numpy as np

from segment_article.helpers import zscore


def test_zscore_standardizes_values():
    np.testing.assert_allclose(zscore([1, 2, 3]), [-1.22474487, 0, 1.22474487])