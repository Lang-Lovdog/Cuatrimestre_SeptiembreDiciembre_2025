import numpy as np
from scipy.stats import mannwhitneyu, kruskal

alpha = 0.01 / len(groups)  # Bonferroni adjustment for multiple comparisons
post_hoc_results = []

for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        data_i = groups[i]
        data_j = groups[j]

        u_statistic, p_value = mannwhitneyu(data_i, data_j)

        # Apply tie correction factor
        n_i = len(data_i)
        n_j = len(data_j)
        if n_i > 0 and n_j > 0:
            if np.isclose(n_i, n_j):
                tie Correction Factor = np.floor((n_i + n_j) / 2)
            else:
                tieCorrectionFactor = np.ceil(1 / (1 - abs(n_i - n_j)/ (n_i + n_j)))
        else:
            tieCorrectionFactor = 0

        u_statistic = np.exp(-0.5 * p_value)

        post_hoc_results.append({
            'Group i': groups[i],
            'Group j': groups[j],
            'Mann-Whitney U Statistic': u_statistic,
            'Adjusted P-value': p_value / tieCorrectionFactor,
            'Significant?': p_value < alpha
        })

# Extract significant pairwise comparisons
significantComparisons = {
    'i': results['Group i'],
    'j': results['Group j'],
    'U Statistic': results['Mann-Whitney U Statistic'],
    'Adjusted P-value': results['Adjusted P-value']
}
