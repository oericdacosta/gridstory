"""Tests for signal processing."""

import numpy as np
import pytest

from src.preprocessing.signal_processing import (
    clean_signal,
    calculate_derivative,
    remove_outliers,
    apply_telemetry_pipeline,
)


def test_clean_signal_median():
    """Test median filter for noise reduction."""
    # Create signal with noise spikes
    signal = np.array([200, 200, 200, 500, 200, 200, 200])  # Spike at index 3

    cleaned = clean_signal(signal, method="median", kernel_size=3)

    # Median filter should remove the spike completely with kernel_size=3
    # The median of [200, 200, 500] is 200
    assert cleaned[3] == 200


def test_clean_signal_savgol():
    """Test Savitzky-Golay filter for smoothing."""
    # Create noisy sine wave
    x = np.linspace(0, 4 * np.pi, 100)
    signal = np.sin(x) + 0.1 * np.random.randn(100)

    cleaned = clean_signal(signal, method="savgol", kernel_size=11, polyorder=3)

    # Cleaned signal should have less variance
    assert np.var(cleaned) < np.var(signal)


def test_clean_signal_with_nan():
    """Test handling of NaN values."""
    signal = np.array([200, 200, np.nan, 200, 200])

    cleaned = clean_signal(signal, method="median", kernel_size=3)

    # Should interpolate NaN
    assert not np.isnan(cleaned).any()


def test_calculate_derivative_basic():
    """Test derivative calculation."""
    # Linear signal: derivative should be constant
    signal = np.array([0, 1, 2, 3, 4, 5])
    derivative = calculate_derivative(signal, delta_x=1.0, smooth=False)

    # Should be approximately 1.0 everywhere
    assert np.allclose(derivative, 1.0, atol=0.1)


def test_calculate_derivative_quadratic():
    """Test derivative of quadratic function."""
    # y = x^2, dy/dx = 2x
    x = np.linspace(0, 10, 100)
    signal = x ** 2
    derivative = calculate_derivative(signal, delta_x=x[1] - x[0], smooth=True)

    # Should approximate 2x
    expected = 2 * x
    # Allow some error due to numerical differentiation
    assert np.allclose(derivative[10:-10], expected[10:-10], rtol=0.2)


def test_remove_outliers_zscore():
    """Test outlier removal using Z-score."""
    # Normal data with one clear outlier (need more data for z-score to work well)
    signal = np.array([100, 101, 99, 100, 102, 98, 101, 100, 99, 102, 500, 100, 101])

    cleaned, outlier_mask = remove_outliers(signal, threshold=3.0, method="zscore")

    # Should detect the outlier at index 10
    assert outlier_mask[10] == True
    assert cleaned[10] != 500  # Should be replaced


def test_remove_outliers_median():
    """Test outlier removal using MAD (Median Absolute Deviation)."""
    # Data with outliers
    signal = np.array([100, 101, 99, 100, 102, 500, 98, 101, 1000])

    cleaned, outlier_mask = remove_outliers(signal, threshold=3.5, method="median")

    # Should detect both outliers
    assert outlier_mask[5] == True  # 500
    assert outlier_mask[8] == True  # 1000


def test_apply_telemetry_pipeline():
    """Test complete telemetry processing pipeline."""
    # Create sample telemetry
    speed = 200 + 10 * np.random.randn(100)
    speed[50] = 500  # Add outlier

    throttle = 80 + 5 * np.random.randn(100)

    telemetry = {
        'Speed': speed,
        'Throttle': throttle,
    }

    processed = apply_telemetry_pipeline(
        telemetry,
        noise_reduction=True,
        outlier_removal=True,
        calculate_derivatives=True,
    )

    # Check outputs
    assert 'Speed' in processed
    assert 'Throttle' in processed
    assert 'Speed_derivative' in processed  # Acceleration
    assert 'Throttle_derivative' in processed

    # Outlier should be reduced
    assert processed['Speed'][50] < 500


def test_clean_signal_invalid_method():
    """Test invalid method raises error."""
    signal = np.array([1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="Método desconhecido"):
        clean_signal(signal, method="invalid_method")
