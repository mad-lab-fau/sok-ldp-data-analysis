from sok_ldp_analysis.simulation.processing import group_results
from sok_ldp_analysis.simulation.processing_fo import group_results_fo

if __name__ == "__main__":
    group_results("mean")
    group_results("variance")
    group_results("mean_multi_dim")
    group_results("mean_gaussian")

    group_results_fo()
