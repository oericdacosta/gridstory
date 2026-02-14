"""Tests for statistical feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineering import (
    calculate_statistical_features,
    calculate_degradation_rate,
    calculate_descriptive_statistics,
    enrich_dataframe_with_stats,
)


def test_calculate_statistical_features_basic():
    """Test basic Z-score calculation and outlier detection."""
    # Create data with a very clear outlier
    df = pd.DataFrame({
        'Driver': ['VER'] * 15,
        'LapTime': [90.0, 90.5, 91.0, 91.5, 90.2, 90.8, 91.2, 90.5, 91.0, 90.3, 90.7, 91.1, 90.4, 90.9, 120.0],  # Last one is extreme outlier
    })

    result = calculate_statistical_features(df, value_column='LapTime')

    # Check new columns exist
    assert 'z_score' in result.columns
    assert 'is_outlier' in result.columns
    assert 'group_mean' in result.columns
    assert 'group_std' in result.columns

    # Last lap should be marked as outlier (z-score > 3)
    # With 14 laps around 90-91s and one at 120s, the z-score should be > 3
    assert result['is_outlier'].iloc[-1] == True
    assert result['is_outlier'].iloc[0] == False


def test_calculate_statistical_features_grouped():
    """Test Z-score calculation with grouping."""
    df = pd.DataFrame({
        'Driver': ['VER', 'VER', 'VER', 'HAM', 'HAM', 'HAM'],
        'Compound': ['SOFT', 'SOFT', 'SOFT', 'SOFT', 'SOFT', 'SOFT'],
        'LapTime': [90.0, 90.5, 91.0, 92.0, 92.5, 93.0],
    })

    result = calculate_statistical_features(
        df,
        value_column='LapTime',
        group_by=['Driver', 'Compound']
    )

    # Each driver should have their own group stats
    ver_mean = result[result['Driver'] == 'VER']['group_mean'].iloc[0]
    ham_mean = result[result['Driver'] == 'HAM']['group_mean'].iloc[0]

    assert ver_mean < ham_mean  # VER is faster


def test_calculate_degradation_rate():
    """Test tire degradation rate calculation."""
    # Create data with clear degradation (0.2s per lap)
    df = pd.DataFrame({
        'Driver': ['VER'] * 10,
        'Stint': [1] * 10,
        'LapNumber': range(1, 11),
        'LapTime': [90.0 + 0.2 * i for i in range(10)],
    })

    result = calculate_degradation_rate(
        df,
        lap_column='LapNumber',
        time_column='LapTime',
        group_by=['Driver', 'Stint']
    )

    # Check degradation columns exist
    assert 'degradation_slope' in result.columns
    assert 'degradation_r_squared' in result.columns
    assert 'degradation_intercept' in result.columns

    # Slope should be approximately 0.2
    slope = result['degradation_slope'].iloc[0]
    assert np.isclose(slope, 0.2, atol=0.01)

    # R² should be very high for linear data
    r_squared = result['degradation_r_squared'].iloc[0]
    assert r_squared > 0.99


def test_calculate_degradation_rate_no_degradation():
    """Test degradation calculation with consistent lap times."""
    df = pd.DataFrame({
        'Driver': ['VER'] * 10,
        'LapNumber': range(1, 11),
        'LapTime': [90.0] * 10,  # Constant lap times
    })

    result = calculate_degradation_rate(df)

    # Slope should be near zero
    slope = result['degradation_slope'].iloc[0]
    assert np.isclose(slope, 0.0, atol=0.01)


def test_calculate_degradation_rate_multiple_stints():
    """Test degradation with multiple stints."""
    df = pd.DataFrame({
        'Driver': ['VER'] * 6,
        'Stint': [1, 1, 1, 2, 2, 2],
        'LapNumber': [1, 2, 3, 1, 2, 3],
        'LapTime': [90.0, 90.5, 91.0, 89.0, 89.5, 90.0],  # Stint 2 is faster
    })

    result = calculate_degradation_rate(
        df,
        group_by=['Driver', 'Stint']
    )

    # Each stint should have its own degradation rate
    stint1_slope = result[result['Stint'] == 1]['degradation_slope'].iloc[0]
    stint2_slope = result[result['Stint'] == 2]['degradation_slope'].iloc[0]

    # Both should have positive degradation
    assert stint1_slope > 0
    assert stint2_slope > 0


def test_calculate_descriptive_statistics():
    """Test descriptive statistics calculation."""
    values = np.array([90.0, 90.5, 91.0, 91.5, 92.0])

    stats = calculate_descriptive_statistics(values)

    # Check all expected keys exist
    assert 'nobs' in stats
    assert 'mean' in stats
    assert 'variance' in stats
    assert 'skewness' in stats
    assert 'kurtosis' in stats
    assert 'min' in stats
    assert 'max' in stats

    # Check values
    assert stats['nobs'] == 5
    assert np.isclose(stats['mean'], 91.0)
    assert np.isclose(stats['min'], 90.0)
    assert np.isclose(stats['max'], 92.0)


def test_calculate_descriptive_statistics_empty():
    """Test descriptive statistics with empty data."""
    values = np.array([])

    stats = calculate_descriptive_statistics(values)

    assert stats['nobs'] == 0
    assert np.isnan(stats['mean'])


def test_enrich_dataframe_with_stats():
    """Test complete enrichment pipeline."""
    df = pd.DataFrame({
        'Driver': ['VER'] * 10,
        'Compound': ['SOFT'] * 10,
        'LapNumber': range(1, 11),
        'LapTime': [90.0 + 0.1 * i + np.random.randn() * 0.05 for i in range(10)],
    })

    enriched = enrich_dataframe_with_stats(
        df,
        value_column='LapTime',
        group_by=['Driver', 'Compound'],
        include_degradation=True
    )

    # Check all features are added
    assert 'z_score' in enriched.columns
    assert 'is_outlier' in enriched.columns
    assert 'degradation_slope' in enriched.columns
    assert 'degradation_r_squared' in enriched.columns

    # Should have positive degradation
    assert enriched['degradation_slope'].iloc[0] > 0


def test_calculate_statistical_features_missing_column():
    """Test error handling for missing column."""
    df = pd.DataFrame({
        'Driver': ['VER', 'HAM'],
    })

    with pytest.raises(ValueError, match="Coluna .* não encontrada"):
        calculate_statistical_features(df, value_column='LapTime')
