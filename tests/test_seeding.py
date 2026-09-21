"""Tests for deterministic seeding."""

from __future__ import annotations

import random

import numpy as np
import pytest

from coloring_page.seeding import seed_everything


class TestSeedEverythingIsDeterministic:
    def test_numpy_random_state_is_reproducible(self) -> None:
        seed_everything(42)
        first = np.random.rand(5)

        seed_everything(42)
        second = np.random.rand(5)

        np.testing.assert_array_equal(first, second)

    def test_stdlib_random_state_is_reproducible(self) -> None:
        seed_everything(42)
        first = [random.random() for _ in range(5)]

        seed_everything(42)
        second = [random.random() for _ in range(5)]

        assert first == second


class TestSeedEverythingNoneIsNoop:
    def test_none_does_not_reset_random_state(self) -> None:
        seed_everything(42)
        before = np.random.rand()

        seed_everything(None)
        # np.random state should have advanced normally, not been reset,
        # so drawing again gives a different value than a fixed reseed would.
        after = np.random.rand()

        assert before != after


class TestSeedEverythingWithTorch:
    def test_torch_random_state_is_reproducible(self) -> None:
        torch = pytest.importorskip("torch")

        seed_everything(42)
        first = torch.rand(5)

        seed_everything(42)
        second = torch.rand(5)

        assert torch.equal(first, second)
