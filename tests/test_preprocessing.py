"""Tests for preprocessing strategies and the pipeline context."""

from __future__ import annotations

import pytest

from recsys.preprocessing.pipeline import PreprocessingPipeline
from recsys.preprocessing.strategies import MinMaxScaler, StandardScaler


def test_min_max_scaler_scales_to_unit_range():
    result = MinMaxScaler().fit_transform([0.0, 5.0, 10.0])
    assert result == [0.0, 0.5, 1.0]


def test_min_max_scaler_handles_constant_column():
    assert MinMaxScaler().fit_transform([3.0, 3.0]) == [0.0, 0.0]


def test_standard_scaler_produces_zero_mean():
    result = StandardScaler().fit_transform([1.0, 2.0, 3.0])
    assert sum(result) == pytest.approx(0.0)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        MinMaxScaler().transform([1.0])


def test_fit_on_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty"):
        MinMaxScaler().fit([])


def test_pipeline_applies_one_strategy_per_column():
    pipeline = PreprocessingPipeline(
        {"price": MinMaxScaler(), "rating": StandardScaler()}
    )
    out = pipeline.fit_transform({"price": [0.0, 10.0], "rating": [1.0, 3.0]})
    assert out["price"] == [0.0, 1.0]
    assert out["rating"][0] == pytest.approx(-1.0)
