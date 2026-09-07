from datetime import datetime, timezone

import pytest

from xau_lean.research.regimes import (
    REGIMES,
    ResearchRegime,
    get_regime,
    validate_regimes,
)


def test_default_regimes_are_chronological_and_non_overlapping():
    validate_regimes(REGIMES)

    for previous, current in zip(
        REGIMES,
        REGIMES[1:],
    ):
        assert previous.end <= current.start


def test_regime_boundaries_are_half_open():
    assert (
        get_regime(
            datetime(
                2015,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
        is REGIMES[1]
    )

    assert (
        get_regime(
            datetime(
                2020,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
        is REGIMES[2]
    )


def test_first_and_last_regime_boundaries():
    assert (
        get_regime(
            datetime(
                2010,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )
        is REGIMES[0]
    )

    assert (
        get_regime(
            datetime(
                2026,
                8,
                20,
                23,
                58,
                tzinfo=timezone.utc,
            )
        )
        is REGIMES[-1]
    )

    assert (
        get_regime(
            datetime(
                2026,
                8,
                21,
                tzinfo=timezone.utc,
            )
        )
        is None
    )


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError):
        get_regime(
            datetime(
                2020,
                1,
                1,
            )
        )


def test_overlapping_regimes_are_rejected():
    regimes = (
        ResearchRegime(
            name="A",
            start=datetime(
                2010,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2015,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        ResearchRegime(
            name="B",
            start=datetime(
                2014,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2020,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        ),
    )

    with pytest.raises(ValueError):
        validate_regimes(regimes)
