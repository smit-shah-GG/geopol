"""
Calibration metrics calculation using netcal library.

Calibration measures how well predicted probabilities match actual frequencies.
A perfectly calibrated model should, among all predictions with p=0.7, see 70% positive outcomes.

Key metrics:
- ECE (Expected Calibration Error): Average calibration error across bins
- MCE (Maximum Calibration Error): Worst-case calibration error
- ACE (Adaptive Calibration Error): ECE with adaptive binning

Target: ECE < 0.1 for good calibration, ECE < 0.05 for excellent calibration
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from netcal.metrics import ECE, ACE, MCE

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """
    Calculates calibration metrics for geopolitical predictions.

    Calibration measures the alignment between predicted probabilities and actual outcomes.
    Uses netcal library for industry-standard ECE, MCE, ACE calculations.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration metrics calculator.

        Args:
            n_bins: Number of bins for ECE calculation (default 10)
        """
        self.n_bins = n_bins

        # Initialize netcal metrics
        self.ece_metric = ECE(bins=n_bins)
        self.mce_metric = MCE(bins=n_bins)
        self.ace_metric = ACE()  # ACE uses adaptive binning

    def calculate_metrics(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all calibration metrics for predictions.

        Args:
            predictions: List of resolved prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict with 'ece', 'mce', 'ace', 'n_predictions', 'is_well_calibrated'

        Raises:
            ValueError: If no resolved predictions provided
        """
        # Filter resolved predictions
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            raise ValueError("No resolved predictions for calibration metrics")

        # Extract probabilities and outcomes
        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        y_true = np.array([p["outcome"] for p in resolved])
        y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in resolved])

        # netcal expects probabilities in [N, 2] format for binary classification
        # Convert to [N, 2]: [prob_negative, prob_positive]
        y_prob_binary = np.column_stack([1 - y_prob, y_prob])

        # Calculate metrics
        ece = float(self.ece_metric.measure(y_prob_binary, y_true))
        mce = float(self.mce_metric.measure(y_prob_binary, y_true))
        ace = float(self.ace_metric.measure(y_prob_binary, y_true))

        # Determine calibration quality
        # ECE < 0.05: Excellent
        # ECE < 0.10: Good
        # ECE < 0.15: Acceptable
        # ECE >= 0.15: Poor (needs recalibration)
        is_well_calibrated = ece < 0.10

        logger.info(
            f"Calibration metrics: ECE={ece:.4f}, MCE={mce:.4f}, ACE={ace:.4f} "
            f"({'GOOD' if is_well_calibrated else 'NEEDS RECALIBRATION'})"
        )

        return {
            "ece": ece,
            "mce": mce,
            "ace": ace,
            "n_predictions": len(resolved),
            "is_well_calibrated": is_well_calibrated,
        }

    def calculate_per_category_metrics(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate calibration metrics separately for each category.

        Args:
            predictions: List of prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            Dict mapping category -> metrics dict
        """
        results = {}

        for category in ["conflict", "diplomatic", "economic"]:
            category_preds = [p for p in predictions if p["category"] == category]

            if not category_preds or not any(
                p.get("outcome") is not None for p in category_preds
            ):
                results[category] = None
                continue

            try:
                metrics = self.calculate_metrics(category_preds, use_calibrated)
                results[category] = metrics
            except ValueError as e:
                logger.warning(f"Could not calculate metrics for {category}: {e}")
                results[category] = None

        return results

    def generate_reliability_diagram(
        self,
        predictions: List[Dict],
        use_calibrated: bool = True,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """
        Generate reliability diagram (calibration plot).

        Shows predicted probability vs observed frequency in each bin.
        Perfect calibration follows the diagonal line.

        Args:
            predictions: List of resolved prediction dicts
            use_calibrated: Use calibrated vs raw probabilities
            output_path: Where to save plot (defaults to ./outputs/reliability_diagram.png)
            title: Plot title (optional)

        Returns:
            Path to saved plot

        Raises:
            ValueError: If no resolved predictions
        """
        # Filter resolved
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            raise ValueError("No resolved predictions for reliability diagram")

        # Extract data
        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        y_true = np.array([p["outcome"] for p in resolved])
        y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in resolved])

        # Set output path
        if output_path is None:
            output_dir = Path("./outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / "reliability_diagram.png")

        # Create our own reliability diagram
        fig, ax = plt.subplots(figsize=(8, 8))

        # Calculate bin statistics
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]

            # Find predictions in bin
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            if i == self.n_bins - 1:  # Include upper boundary for last bin
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accs.append(np.mean(y_true[in_bin]))
                bin_counts.append(np.sum(in_bin))

        # Plot reliability curve
        if bin_centers:
            ax.plot(bin_centers, bin_accs, 'o-', label='Model', linewidth=2, markersize=8)

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)

        # Formatting
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Observed Frequency', fontsize=12)
        ax.set_title(f'Reliability Diagram {title or ""}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved reliability diagram to {output_path}")

        return output_path

    def generate_per_category_diagrams(
        self,
        predictions: List[Dict],
        use_calibrated: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate reliability diagrams for each category.

        Args:
            predictions: List of prediction dicts
            use_calibrated: Use calibrated vs raw probabilities
            output_dir: Directory for plots (defaults to ./outputs)

        Returns:
            Dict mapping category -> plot path
        """
        if output_dir is None:
            output_dir = "./outputs"

        Path(output_dir).mkdir(exist_ok=True)

        results = {}

        for category in ["conflict", "diplomatic", "economic"]:
            category_preds = [p for p in predictions if p["category"] == category]

            if not category_preds or not any(
                p.get("outcome") is not None for p in category_preds
            ):
                results[category] = None
                continue

            try:
                plot_path = self.generate_reliability_diagram(
                    category_preds,
                    use_calibrated,
                    output_path=f"{output_dir}/reliability_{category}.png",
                    title=f"{category.capitalize()} Predictions",
                )
                results[category] = plot_path
            except ValueError as e:
                logger.warning(f"Could not generate diagram for {category}: {e}")
                results[category] = None

        return results

    def calculate_bin_statistics(
        self, predictions: List[Dict], use_calibrated: bool = True
    ) -> List[Dict]:
        """
        Calculate statistics for each probability bin.

        Args:
            predictions: List of resolved prediction dicts
            use_calibrated: Use calibrated vs raw probabilities

        Returns:
            List of dicts with bin statistics:
                - bin_lower: Lower bound of bin
                - bin_upper: Upper bound of bin
                - n_predictions: Count in bin
                - avg_predicted: Average predicted probability
                - avg_observed: Observed frequency
                - calibration_error: |avg_predicted - avg_observed|
        """
        resolved = [p for p in predictions if p.get("outcome") is not None]

        if not resolved:
            return []

        # Extract data
        prob_key = "calibrated_probability" if use_calibrated else "raw_probability"

        y_true = np.array([p["outcome"] for p in resolved])
        y_prob = np.array([p.get(prob_key, p["raw_probability"]) for p in resolved])

        # Create bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)

        bin_stats = []

        for i in range(self.n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]

            # Find predictions in bin
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

            # Handle last bin (include upper boundary)
            if i == self.n_bins - 1:
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)

            n_in_bin = np.sum(in_bin)

            if n_in_bin == 0:
                continue

            # Calculate statistics
            bin_probs = y_prob[in_bin]
            bin_outcomes = y_true[in_bin]

            avg_predicted = float(np.mean(bin_probs))
            avg_observed = float(np.mean(bin_outcomes))
            calibration_error = abs(avg_predicted - avg_observed)

            bin_stats.append(
                {
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "n_predictions": int(n_in_bin),
                    "avg_predicted": avg_predicted,
                    "avg_observed": avg_observed,
                    "calibration_error": calibration_error,
                }
            )

        return bin_stats
