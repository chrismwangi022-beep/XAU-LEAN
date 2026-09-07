"""
Bounded-memory streaming quantile estimation.

Implements the P² algorithm described by Jain & Chlamtac (1985).

The estimator maintains five markers and requires constant memory.
It is deterministic for a deterministic chronological input stream.

This module is intended for descriptive research statistics where
retaining every observation would defeat the streaming architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Iterable


@dataclass(slots=True)
class P2Quantile:
    """
    Streaming P² estimator for one quantile.

    Parameters
    ----------
    quantile:
        Target quantile in the interval [0, 1].

    Notes
    -----
    The first five observations are retained exactly. Once initialized,
    the estimator maintains five marker heights and positions.

    For fewer than five observations, the exact sample quantile is
    returned.

    Memory usage after initialization is constant.
    """

    quantile: float

    # Internal P² state.
    #
    # These must be declared explicitly because slots=True prevents
    # dynamically creating attributes in __post_init__.
    _initial: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _q: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _n: list[int] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _np: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _dn: list[float] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _count: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError(
                "quantile must be between 0.0 and 1.0"
            )

    @property
    def count(self) -> int:
        """Number of observations received."""
        return self._count

    @staticmethod
    def _exact_quantile(
        values: list[float],
        quantile: float,
    ) -> float:
        """Return the exact linear-interpolated sample quantile."""
        ordered = sorted(values)

        if len(ordered) == 1:
            return ordered[0]

        position = quantile * (len(ordered) - 1)

        lower = floor(position)
        upper = lower + 1

        if upper >= len(ordered):
            return ordered[-1]

        fraction = position - lower

        return (
            ordered[lower]
            + fraction
            * (ordered[upper] - ordered[lower])
        )

    def update(self, value: float) -> None:
        """Add one observation to the estimator."""
        value = float(value)
        self._count += 1

        # Initialization phase.
        #
        # The first five observations are retained exactly so that
        # the P² markers can be initialized from the empirical sample.
        if len(self._initial) < 5:
            self._initial.append(value)

            if len(self._initial) == 5:
                self._initial.sort()

                self._q = list(self._initial)

                # Actual marker positions.
                self._n = [1, 2, 3, 4, 5]

                q = self.quantile

                # Desired marker positions.
                self._np = [
                    1.0,
                    1.0 + 2.0 * q,
                    1.0 + 4.0 * q,
                    3.0 + 2.0 * q,
                    5.0,
                ]

                # Desired position increments.
                self._dn = [
                    0.0,
                    q / 2.0,
                    q,
                    (1.0 + q) / 2.0,
                    1.0,
                ]

            return

        # ------------------------------------------------------------------
        # P² update phase
        # ------------------------------------------------------------------

        # Find the interval k containing the new observation.
        #
        # q[0] and q[4] are the minimum and maximum markers.
        if value < self._q[0]:
            self._q[0] = value
            k = 0

        elif value < self._q[1]:
            k = 0

        elif value < self._q[2]:
            k = 1

        elif value < self._q[3]:
            k = 2

        elif value <= self._q[4]:
            k = 3

        else:
            self._q[4] = value
            k = 3

        # Every marker above the insertion interval advances by one.
        for index in range(k + 1, 5):
            self._n[index] += 1

        # Advance desired marker positions.
        for index in range(5):
            self._np[index] += self._dn[index]

        # Adjust interior markers when their actual position differs
        # sufficiently from their desired position.
        for index in range(1, 4):
            difference = self._np[index] - self._n[index]

            move_up = (
                difference >= 1.0
                and self._n[index + 1] - self._n[index] > 1
            )

            move_down = (
                difference <= -1.0
                and self._n[index - 1] - self._n[index] < -1
            )

            if not (move_up or move_down):
                continue

            direction = 1 if difference > 0.0 else -1

            proposed = self._parabolic(
                index,
                direction,
            )

            # P² requires the adjusted marker to remain between
            # its neighboring marker heights.
            if (
                self._q[index - 1]
                < proposed
                < self._q[index + 1]
            ):
                self._q[index] = proposed
            else:
                # Fall back to the linear estimate when the parabolic
                # estimate would violate marker ordering.
                self._q[index] = self._linear(
                    index,
                    direction,
                )

            self._n[index] += direction

    def _parabolic(
        self,
        index: int,
        direction: int,
    ) -> float:
        """Calculate a parabolic marker adjustment."""
        n0 = self._n[index - 1]
        n1 = self._n[index]
        n2 = self._n[index + 1]

        q0 = self._q[index - 1]
        q1 = self._q[index]
        q2 = self._q[index + 1]

        return q1 + (
            direction / (n2 - n0)
        ) * (
            (n1 - n0 + direction)
            * (q2 - q1)
            / (n2 - n1)
            +
            (n2 - n1 - direction)
            * (q1 - q0)
            / (n1 - n0)
        )

    def _linear(
        self,
        index: int,
        direction: int,
    ) -> float:
        """Calculate a linear marker adjustment."""
        neighbor = index + direction

        return self._q[index] + (
            direction
            * (
                self._q[neighbor]
                - self._q[index]
            )
            / (
                self._n[neighbor]
                - self._n[index]
            )
        )

    @property
    def value(self) -> float | None:
        """Return the current quantile estimate."""
        if self._count == 0:
            return None

        if self._count < 5:
            return self._exact_quantile(
                self._initial,
                self.quantile,
            )

        # For a P² estimator, the middle marker is the target
        # quantile estimate.
        return self._q[2]


DEFAULT_QUANTILES = (
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
)


class StreamingQuantiles:
    """
    Collection of bounded-memory P² estimators.

    One object maintains the standard Phase 6.1 quantile set.
    """

    def __init__(
        self,
        quantiles: Iterable[float] = DEFAULT_QUANTILES,
    ) -> None:
        selected = tuple(float(q) for q in quantiles)

        if not selected:
            raise ValueError(
                "at least one quantile is required"
            )

        if any(
            q < 0.0 or q > 1.0
            for q in selected
        ):
            raise ValueError(
                "quantiles must be between 0.0 and 1.0"
            )

        if len(set(selected)) != len(selected):
            raise ValueError(
                "quantiles must be unique"
            )

        self._estimators = {
            q: P2Quantile(q)
            for q in selected
        }

    def update(self, value: float) -> None:
        """Add one observation to every quantile estimator."""
        for estimator in self._estimators.values():
            estimator.update(value)

    def value(self, quantile: float) -> float | None:
        """Return the estimate for one configured quantile."""
        quantile = float(quantile)

        try:
            estimator = self._estimators[quantile]
        except KeyError as exc:
            raise KeyError(
                f"quantile {quantile} is not configured"
            ) from exc

        return estimator.value

    def to_dict(self) -> dict[str, float | None]:
        """Return quantiles using stable percentage keys."""
        return {
            f"p{int(q * 100):02d}": estimator.value
            for q, estimator in self._estimators.items()
        }


__all__ = [
    "P2Quantile",
    "StreamingQuantiles",
    "DEFAULT_QUANTILES",
]