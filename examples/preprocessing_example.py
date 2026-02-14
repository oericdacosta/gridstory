"""
Exemplo demonstrando pipeline de pré-processamento SciPy com dados FastF1.

Este exemplo mostra como:
1. Carregar dados do FastF1
2. Sincronizar telemetria usando scipy.interpolate
3. Limpar sinais usando scipy.signal
4. Calcular features estatísticas usando scipy.stats
"""

import numpy as np
import fastf1

from src.preprocessing.interpolation import (
    synchronize_telemetry,
    synchronize_multiple_laps,
)
from src.preprocessing.signal_processing import apply_telemetry_pipeline
from src.preprocessing.feature_engineering import enrich_dataframe_with_stats


def example_telemetry_synchronization():
    """Example 1: Synchronize telemetry for driver comparison."""
    print("=" * 80)
    print("EXAMPLE 1: Telemetry Synchronization")
    print("=" * 80)

    # Load a session (using cache to avoid repeated downloads)
    session = fastf1.get_session(2024, "Monaco", "Q")
    session.load()

    # Get fastest laps for two drivers
    ver_lap = session.laps.pick_driver("VER").pick_fastest()
    ham_lap = session.laps.pick_driver("HAM").pick_fastest()

    # Get track length
    track_length = session.get_circuit_info().total_distance

    # Synchronize telemetry for both drivers
    ver_telemetry = ver_lap.get_telemetry()
    ham_telemetry = ham_lap.get_telemetry()

    print(f"\nOriginal VER telemetry points: {len(ver_telemetry)}")
    print(f"Original HAM telemetry points: {len(ham_telemetry)}")

    # Synchronize both to common grid
    ver_sync = synchronize_telemetry(ver_telemetry, track_length, num_points=1000)
    ham_sync = synchronize_telemetry(ham_telemetry, track_length, num_points=1000)

    print(f"\nSynchronized VER telemetry points: {len(ver_sync)}")
    print(f"Synchronized HAM telemetry points: {len(ham_sync)}")
    print("\nNow both drivers have telemetry at EXACTLY the same distances!")

    # Calculate delta
    speed_delta = ver_sync["Speed"].values - ham_sync["Speed"].values

    print(f"\nMax speed advantage (VER): {speed_delta.max():.2f} km/h")
    print(f"Max speed advantage (HAM): {abs(speed_delta.min()):.2f} km/h")
    print(f"Average speed delta: {speed_delta.mean():.2f} km/h")

    return ver_sync, ham_sync


def example_signal_processing():
    """Example 2: Clean noisy telemetry signals."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Signal Processing")
    print("=" * 80)

    # Load session
    session = fastf1.get_session(2024, "Monaco", "Q")
    session.load()

    # Get a lap
    lap = session.laps.pick_driver("VER").pick_fastest()
    telemetry = lap.get_telemetry()

    # Extract telemetry channels
    telemetry_dict = {
        "Speed": telemetry["Speed"].values,
        "Throttle": telemetry["Throttle"].values,
        "Brake": telemetry["Brake"].values,
        "RPM": telemetry["RPM"].values,
    }

    print("\nOriginal signal statistics:")
    for channel, values in telemetry_dict.items():
        print(
            f"  {channel}: mean={np.nanmean(values):.2f}, std={np.nanstd(values):.2f}"
        )

    # Apply complete processing pipeline
    processed = apply_telemetry_pipeline(
        telemetry_dict,
        noise_reduction=True,
        outlier_removal=True,
        calculate_derivatives=True,
    )

    print("\nProcessed signal statistics:")
    for channel in ["Speed", "Throttle", "Brake", "RPM"]:
        values = processed[channel]
        print(
            f"  {channel}: mean={np.nanmean(values):.2f}, std={np.nanstd(values):.2f}"
        )

    print("\nDerivatives calculated:")
    print(
        f"  Speed_derivative (acceleration): shape={processed['Speed_derivative'].shape}"
    )
    print(f"  Throttle_derivative: shape={processed['Throttle_derivative'].shape}")

    # Calculate acceleration
    if "Speed_derivative" in processed:
        accel = processed["Speed_derivative"]
        print(f"\nMax acceleration: {np.nanmax(accel):.2f}")
        print(f"Max deceleration: {np.nanmin(accel):.2f}")

    return processed


def example_statistical_features():
    """Example 3: Calculate statistical features for stint analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Statistical Feature Engineering")
    print("=" * 80)

    # Load race session
    session = fastf1.get_session(2024, "Monaco", "R")
    session.load()

    # Get all laps for a driver
    driver = "VER"
    laps = session.laps.pick_driver(driver)

    # Select relevant columns
    laps_df = laps[["LapNumber", "LapTime", "Compound", "TyreLife", "Stint"]].copy()

    print(f"\nAnalyzing {len(laps_df)} laps for {driver}")
    print(f"Stints: {laps_df['Stint'].unique()}")

    # Enrich with statistical features
    enriched = enrich_dataframe_with_stats(
        laps_df,
        value_column="LapTime",
        group_by=["Driver", "Stint", "Compound"],
        include_degradation=True,
    )

    print("\nNew columns added:")
    new_cols = [col for col in enriched.columns if col not in laps_df.columns]
    print(f"  {new_cols}")

    # Analyze outliers
    outliers = enriched[enriched["is_outlier"]]
    print(f"\nOutliers detected: {len(outliers)}")
    if len(outliers) > 0:
        print(f"Outlier laps: {outliers['LapNumber'].tolist()}")

    # Analyze degradation per stint
    print("\nDegradation analysis by stint:")
    for stint in enriched["Stint"].unique():
        stint_data = enriched[enriched["Stint"] == stint].iloc[0]
        if "degradation_slope" in enriched.columns:
            slope = stint_data["degradation_slope"]
            r2 = stint_data["degradation_r_squared"]
            compound = stint_data["Compound"]
            print(f"  Stint {stint} ({compound}): {slope:.3f}s/lap (R²={r2:.3f})")

    return enriched


def example_complete_pipeline():
    """Example 4: Complete preprocessing pipeline for ML."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete ML-Ready Pipeline")
    print("=" * 80)

    # Load session
    session = fastf1.get_session(2024, "Monaco", "Q")
    session.load()

    # Get multiple laps
    driver = "VER"
    laps = session.laps.pick_driver(driver).pick_quicklaps()

    print(f"\nProcessing {len(laps)} quick laps for {driver}")

    # Step 1: Synchronize all laps onto common grid
    track_length = session.get_circuit_info().total_distance

    try:
        synchronized_matrix = synchronize_multiple_laps(
            laps,
            track_length,
            num_points=500,
        )
        print(f"\nStep 1 - Synchronized telemetry: {len(synchronized_matrix)} laps")
    except Exception as e:
        print(f"\nStep 1 - Synchronization failed: {e}")
        synchronized_matrix = None

    # Step 2: Calculate statistical features on lap times
    laps_df = laps[["LapNumber", "LapTime", "Compound", "TyreLife"]].copy()
    enriched = enrich_dataframe_with_stats(
        laps_df,
        value_column="LapTime",
        group_by=["Compound"],
        include_degradation=False,  # Not meaningful for qualifying
    )

    print("\nStep 2 - Statistical features calculated:")
    print(
        f"  Z-scores range: [{enriched['z_score'].min():.2f}, {enriched['z_score'].max():.2f}]"
    )
    print(f"  Outliers: {enriched['is_outlier'].sum()} / {len(enriched)}")

    # Step 3: Filter clean laps for ML
    clean_laps = enriched[~enriched["is_outlier"]]
    print(f"\nStep 3 - Clean laps for ML: {len(clean_laps)} / {len(enriched)}")

    print("\n✓ Data is now ready for Scikit-learn and Ruptures!")

    return synchronized_matrix, clean_laps


if __name__ == "__main__":
    # Enable FastF1 cache
    fastf1.Cache.enable_cache("cache/")

    print("\n" + "=" * 80)
    print("SciPy Preprocessing Examples for F1 Telemetry")
    print("=" * 80)

    # Run examples
    try:
        # Example 1: Telemetry synchronization for driver comparison
        ver_sync, ham_sync = example_telemetry_synchronization()

        # Example 2: Signal processing and noise reduction
        processed = example_signal_processing()

        # Example 3: Statistical feature engineering
        enriched = example_statistical_features()

        # Example 4: Complete pipeline
        synchronized, clean = example_complete_pipeline()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
