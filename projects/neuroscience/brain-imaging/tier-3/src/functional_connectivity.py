"""
Functional connectivity analysis module.

Implements resting-state and task-based fMRI connectivity analysis.
"""

import logging
from typing import Optional

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import connectome, datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectivityAnalyzer:
    """Analyze functional connectivity from fMRI data."""

    def __init__(self):
        """Initialize connectivity analyzer."""
        self.atlases = self._load_atlases()

    def _load_atlases(self) -> dict:
        """Load commonly used brain atlases."""
        atlases = {}

        try:
            # Schaefer 2018 atlas (400 regions)
            atlases["schaefer2018_400"] = datasets.fetch_atlas_schaefer_2018(
                n_rois=400, yeo_networks=7
            )
        except Exception as e:
            logger.warning(f"Could not load Schaefer atlas: {e}")

        try:
            # AAL atlas
            atlases["aal"] = datasets.fetch_atlas_aal()
        except Exception as e:
            logger.warning(f"Could not load AAL atlas: {e}")

        try:
            # Harvard-Oxford cortical atlas
            atlases["harvard_oxford"] = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm"
            )
        except Exception as e:
            logger.warning(f"Could not load Harvard-Oxford atlas: {e}")

        return atlases

    def extract_time_series(
        self,
        fmri_file: str,
        atlas: str = "schaefer2018_400",
        confounds: Optional[str] = None,
        standardize: bool = True,
        detrend: bool = True,
        low_pass: Optional[float] = 0.1,
        high_pass: Optional[float] = 0.01,
        t_r: float = 2.0,
    ) -> np.ndarray:
        """
        Extract regional time series from fMRI data.

        Args:
            fmri_file: Path to preprocessed fMRI file
            atlas: Atlas name
            confounds: Path to confounds file (TSV)
            standardize: Whether to standardize signals
            detrend: Whether to detrend signals
            low_pass: Low-pass filter cutoff (Hz)
            high_pass: High-pass filter cutoff (Hz)
            t_r: Repetition time (seconds)

        Returns:
            Time series array (timepoints × regions)
        """
        logger.info(f"Extracting time series using {atlas} atlas...")

        # Get atlas
        if atlas not in self.atlases:
            raise ValueError(f"Unknown atlas: {atlas}")

        atlas_data = self.atlases[atlas]

        # Create masker
        masker = NiftiLabelsMasker(
            labels_img=atlas_data["maps"],
            standardize=standardize,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            memory="nilearn_cache",
            verbose=1,
        )

        # Load confounds if provided
        confounds_data = None
        if confounds:
            confounds_df = pd.read_csv(confounds, sep="\t")
            # Use common confounds
            confound_cols = [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "csf",
                "white_matter",
            ]
            available_cols = [c for c in confound_cols if c in confounds_df.columns]
            if available_cols:
                confounds_data = confounds_df[available_cols].values

        # Extract time series
        time_series = masker.fit_transform(fmri_file, confounds=confounds_data)

        logger.info(
            f"Extracted time series: {time_series.shape[0]} timepoints, "
            f"{time_series.shape[1]} regions"
        )

        return time_series

    def compute_connectivity(
        self, time_series: np.ndarray, method: str = "correlation", fisher_z: bool = True
    ) -> np.ndarray:
        """
        Compute functional connectivity matrix.

        Args:
            time_series: Time series array (timepoints × regions)
            method: Connectivity method ('correlation', 'partial_correlation', 'precision')
            fisher_z: Apply Fisher z-transformation to correlations

        Returns:
            Connectivity matrix (regions × regions)
        """
        logger.info(f"Computing {method} connectivity...")

        if method == "correlation":
            conn_matrix = np.corrcoef(time_series.T)
            if fisher_z:
                # Fisher z-transformation
                conn_matrix = np.arctanh(conn_matrix)
        elif method == "partial_correlation":
            conn_measure = connectome.ConnectivityMeasure(kind="partial correlation")
            conn_matrix = conn_measure.fit_transform([time_series])[0]
        elif method == "precision":
            conn_measure = connectome.ConnectivityMeasure(kind="precision")
            conn_matrix = conn_measure.fit_transform([time_series])[0]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Set diagonal to zero
        np.fill_diagonal(conn_matrix, 0)

        return conn_matrix

    def detect_networks(
        self,
        connectivity_matrix: np.ndarray,
        algorithm: str = "louvain",
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Detect brain networks using community detection.

        Args:
            connectivity_matrix: Connectivity matrix
            algorithm: Community detection algorithm ('louvain', 'spectral')
            threshold: Threshold for binarizing connectivity (None = use all edges)

        Returns:
            Array of community assignments
        """
        logger.info(f"Detecting networks using {algorithm}...")

        # Threshold matrix if specified
        if threshold is not None:
            conn_matrix = connectivity_matrix.copy()
            conn_matrix[np.abs(conn_matrix) < threshold] = 0
        else:
            conn_matrix = connectivity_matrix

        # Convert to NetworkX graph
        G = nx.from_numpy_array(np.abs(conn_matrix))

        # Community detection
        if algorithm == "louvain":
            import community as community_louvain

            communities = community_louvain.best_partition(G)
            assignments = np.array([communities[i] for i in range(len(communities))])
        elif algorithm == "spectral":
            from sklearn.cluster import SpectralClustering

            n_clusters = 7  # Default to 7 Yeo networks
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
            )
            assignments = clustering.fit_predict(np.abs(conn_matrix))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(f"Detected {len(np.unique(assignments))} networks")

        return assignments

    def compute_graph_metrics(
        self, connectivity_matrix: np.ndarray, threshold: float = 0.3
    ) -> dict[str, float]:
        """
        Compute graph theory metrics.

        Args:
            connectivity_matrix: Connectivity matrix
            threshold: Threshold for binarizing connectivity

        Returns:
            Dictionary of graph metrics
        """
        logger.info("Computing graph theory metrics...")

        # Threshold and binarize
        conn_matrix = connectivity_matrix.copy()
        conn_matrix = np.abs(conn_matrix)
        conn_matrix[conn_matrix < threshold] = 0

        # Convert to NetworkX graph
        G = nx.from_numpy_array(conn_matrix)

        # Compute metrics
        metrics = {}

        # Global metrics
        if nx.is_connected(G):
            metrics["global_efficiency"] = nx.global_efficiency(G)
            metrics["average_shortest_path"] = nx.average_shortest_path_length(G)
        else:
            # For disconnected graphs, compute on largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            metrics["global_efficiency"] = nx.global_efficiency(G_cc)
            metrics["average_shortest_path"] = nx.average_shortest_path_length(G_cc)

        metrics["local_efficiency"] = nx.local_efficiency(G)
        metrics["clustering_coefficient"] = nx.average_clustering(G)
        metrics["transitivity"] = nx.transitivity(G)

        # Modularity (requires community detection)
        try:
            import community as community_louvain

            communities = community_louvain.best_partition(G)
            metrics["modularity"] = community_louvain.modularity(communities, G)
        except ImportError:
            logger.warning("python-louvain not installed, skipping modularity")

        # Small-worldness (sigma)
        # Compare to random graph
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        G_random = nx.gnm_random_graph(n_nodes, n_edges)

        clustering_real = nx.average_clustering(G)
        clustering_random = nx.average_clustering(G_random)

        if nx.is_connected(G) and nx.is_connected(G_random):
            path_real = nx.average_shortest_path_length(G)
            path_random = nx.average_shortest_path_length(G_random)

            # Small-worldness coefficient
            gamma = clustering_real / clustering_random
            lambda_val = path_real / path_random
            if lambda_val > 0:
                metrics["small_worldness"] = gamma / lambda_val
            else:
                metrics["small_worldness"] = np.nan
        else:
            metrics["small_worldness"] = np.nan

        logger.info(f"Computed {len(metrics)} graph metrics")

        return metrics

    def identify_hubs(self, connectivity_matrix: np.ndarray, threshold: float = 0.9) -> list[int]:
        """
        Identify hub regions based on node degree.

        Args:
            connectivity_matrix: Connectivity matrix
            threshold: Percentile threshold for hub identification

        Returns:
            List of hub node indices
        """
        # Compute node degrees (strength)
        degrees = np.sum(np.abs(connectivity_matrix), axis=0)

        # Identify hubs (top percentile)
        hub_threshold = np.percentile(degrees, threshold * 100)
        hubs = np.where(degrees >= hub_threshold)[0]

        logger.info(f"Identified {len(hubs)} hub regions")

        return hubs.tolist()

    def plot_connectivity_matrix(
        self,
        connectivity_matrix: np.ndarray,
        labels: Optional[list[str]] = None,
        title: str = "Functional Connectivity",
        save_path: Optional[str] = None,
    ):
        """
        Plot connectivity matrix.

        Args:
            connectivity_matrix: Connectivity matrix
            labels: Region labels
            title: Plot title
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        _fig, ax = plt.subplots(figsize=(12, 10))

        # Plot matrix
        sns.heatmap(
            connectivity_matrix,
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0,
            cbar_kws={"label": "Correlation (Fisher z)"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16)

        if labels:
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels, rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_brain_networks(
        self, networks: np.ndarray, atlas: str = "schaefer2018_400", save_path: Optional[str] = None
    ):
        """
        Plot brain networks on glass brain.

        Args:
            networks: Network assignments for each region
            atlas: Atlas name
            save_path: Path to save figure
        """
        atlas_data = self.atlases[atlas]

        # Create network image
        network_img = nib.Nifti1Image(networks, affine=atlas_data["maps"].affine)

        # Plot
        display = plotting.plot_roi(network_img, title="Brain Networks", cmap="tab10")

        if save_path:
            display.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plotting.show()

    def group_comparison(
        self,
        connectivity_matrices_group1: list[np.ndarray],
        connectivity_matrices_group2: list[np.ndarray],
        alpha: float = 0.05,
        correction: str = "fdr",
    ) -> dict:
        """
        Compare connectivity between two groups.

        Args:
            connectivity_matrices_group1: List of connectivity matrices (group 1)
            connectivity_matrices_group2: List of connectivity matrices (group 2)
            alpha: Significance level
            correction: Multiple comparison correction ('fdr', 'bonferroni', None)

        Returns:
            Dictionary with statistics
        """
        logger.info(
            f"Comparing groups: {len(connectivity_matrices_group1)} vs "
            f"{len(connectivity_matrices_group2)} subjects..."
        )

        # Stack matrices
        group1 = np.array(connectivity_matrices_group1)
        group2 = np.array(connectivity_matrices_group2)

        # Two-sample t-test at each edge
        n_regions = group1.shape[1]
        t_stats = np.zeros((n_regions, n_regions))
        p_values = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                t_stat, p_val = stats.ttest_ind(group1[:, i, j], group2[:, i, j])
                t_stats[i, j] = t_stat
                t_stats[j, i] = t_stat
                p_values[i, j] = p_val
                p_values[j, i] = p_val

        # Multiple comparison correction
        if correction == "fdr":
            from statsmodels.stats.multitest import fdrcorrection

            # Get upper triangle
            triu_indices = np.triu_indices(n_regions, k=1)
            p_flat = p_values[triu_indices]
            _, p_corrected_flat = fdrcorrection(p_flat, alpha=alpha)
            # Reconstruct matrix
            p_corrected = np.zeros_like(p_values)
            p_corrected[triu_indices] = p_corrected_flat
            p_corrected = p_corrected + p_corrected.T
        elif correction == "bonferroni":
            n_tests = (n_regions * (n_regions - 1)) // 2
            p_corrected = p_values * n_tests
        else:
            p_corrected = p_values

        # Significant edges
        significant = p_corrected < alpha

        results = {
            "t_statistics": t_stats,
            "p_values": p_values,
            "p_corrected": p_corrected,
            "significant_edges": significant,
            "n_significant": np.sum(np.triu(significant, k=1)),
        }

        logger.info(f"Found {results['n_significant']} significant edges")

        return results


def main():
    """
    Main function for connectivity analysis demo.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Functional connectivity analysis")
    parser.add_argument("--fmri", type=str, required=True, help="Path to fMRI file")
    parser.add_argument("--atlas", type=str, default="schaefer2018_400", help="Atlas name")
    parser.add_argument("--confounds", type=str, help="Path to confounds file")
    parser.add_argument(
        "--output",
        type=str,
        default="connectivity_matrix.npy",
        help="Output file for connectivity matrix",
    )
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ConnectivityAnalyzer()

    # Extract time series
    time_series = analyzer.extract_time_series(
        args.fmri, atlas=args.atlas, confounds=args.confounds
    )

    # Compute connectivity
    conn_matrix = analyzer.compute_connectivity(time_series)

    # Save
    np.save(args.output, conn_matrix)
    print(f"Saved connectivity matrix to {args.output}")

    # Compute metrics
    metrics = analyzer.compute_graph_metrics(conn_matrix)
    print("\nGraph Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Detect networks
    networks = analyzer.detect_networks(conn_matrix)
    print(f"\nDetected {len(np.unique(networks))} networks")

    # Plot
    analyzer.plot_connectivity_matrix(
        conn_matrix, title="Functional Connectivity", save_path="connectivity_matrix.png"
    )


if __name__ == "__main__":
    main()
