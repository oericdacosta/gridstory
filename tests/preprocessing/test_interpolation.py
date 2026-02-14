"""Tests for telemetry interpolation."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.interpolation import (
    synchronize_telemetry,
    synchronize_multiple_laps,
)


def test_synchronize_telemetry_basic():
    """Test basic telemetry synchronization."""
    # Create sample telemetry data
    telemetry = pd.DataFrame({
        'Distance': np.linspace(0, 5000, 100),
        'Speed': 200 + 50 * np.sin(np.linspace(0, 4 * np.pi, 100)),
        'RPM': 10000 + 2000 * np.sin(np.linspace(0, 4 * np.pi, 100)),
        'Throttle': 80 + 20 * np.sin(np.linspace(0, 4 * np.pi, 100)),
    })

    synchronized = synchronize_telemetry(
        telemetry,
        track_length=5000,
        num_points=500,
    )

    # Check output structure
    assert len(synchronized) == 500
    assert 'Distance' in synchronized.columns
    assert 'Speed' in synchronized.columns
    assert 'RPM' in synchronized.columns

    # Check distance grid is uniform
    distances = synchronized['Distance'].values
    assert np.allclose(distances[0], 0)
    assert np.allclose(distances[-1], 5000)
    assert np.allclose(np.diff(distances), distances[1] - distances[0])


def test_synchronize_telemetry_with_gaps():
    """Test synchronization with missing data points."""
    # Create data with gaps
    distances = np.concatenate([
        np.linspace(0, 1000, 50),
        np.linspace(2000, 5000, 50),  # Gap between 1000-2000
    ])

    telemetry = pd.DataFrame({
        'Distance': distances,
        'Speed': 200 * np.ones(100),
        'Throttle': 100 * np.ones(100),
    })

    synchronized = synchronize_telemetry(
        telemetry,
        track_length=5000,
        num_points=100,
    )

    # Should interpolate the gap
    assert len(synchronized) == 100
    assert not synchronized['Speed'].isna().any()


def test_synchronize_telemetry_edge_cases():
    """Test edge cases in synchronization."""
    # Very sparse data
    telemetry = pd.DataFrame({
        'Distance': [0, 2500, 5000],
        'Speed': [200, 250, 200],
    })

    synchronized = synchronize_telemetry(
        telemetry,
        track_length=5000,
        num_points=50,
    )

    assert len(synchronized) == 50
    assert synchronized['Speed'].max() <= 250
    assert synchronized['Speed'].min() >= 200


def test_synchronize_telemetry_missing_column():
    """Test handling of missing required column."""
    telemetry = pd.DataFrame({
        'Speed': [200, 210, 220],
    })

    with pytest.raises(ValueError, match="deve conter coluna 'Distance'"):
        synchronize_telemetry(telemetry, track_length=5000)


def test_synchronize_telemetry_no_valid_columns():
    """Test handling when no requested columns exist."""
    telemetry = pd.DataFrame({
        'Distance': [0, 100, 200],
        'SomeOtherColumn': [1, 2, 3],
    })

    with pytest.raises(ValueError, match="Nenhuma das colunas de telemetria solicitadas"):
        synchronize_telemetry(
            telemetry,
            track_length=5000,
            telemetry_columns=['Speed', 'RPM']
        )
