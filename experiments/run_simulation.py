import argparse
import pathlib

from sok_ldp_analysis.simulation.simulation_fo import simulate_fo
from sok_ldp_analysis.simulation.simulation_mean import simulate_mean
from sok_ldp_analysis.simulation.simulation_mean_gaussian import simulate_mean_gaussian
from sok_ldp_analysis.simulation.simulation_mean_multi import simulate_mean_multi_dim
from sok_ldp_analysis.simulation.simulation_variance import simulate_variance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("type", type=str, default="mean", help="The type of simulation to run")
    parser.add_argument("--output", type=str, default="results/simulation", help="The output path")
    parser.add_argument("--mpi", action="store_true", help="Use MPI for simulations")
    parser.add_argument("--i", type=int, default=None, help="The index of the simulation")
    args = parser.parse_args()

    use_mpi = args.mpi

    if use_mpi:
        print("Using MPI for simulations")
    else:
        print("Using multiprocessing for simulations")

    output_path = pathlib.Path(args.output)

    sim_type = args.type

    if sim_type == "mean":
        # Run 1D mean simulation
        simulate_mean(output_path / "mean", use_mpi=use_mpi, i=args.i)

    if sim_type == "mean_multi_dim":
        # Run multi-dimensional mean simulation
        simulate_mean_multi_dim(output_path / "mean_multi_dim", use_mpi=use_mpi, i=args.i)

    if sim_type == "mean_gaussian":
        # Run 1D mean Gaussian simulation
        simulate_mean_gaussian(output_path / "mean_gaussian", use_mpi=use_mpi, i=args.i)

    if sim_type == "fo":
        # Run frequency oracle simulation
        simulate_fo(output_path / "fo", use_mpi=use_mpi, i=args.i)

    if sim_type == "variance":
        # Run variance simulation
        simulate_variance(output_path / "variance", use_mpi=use_mpi, i=args.i)
