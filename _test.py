import jax.numpy as jnp
import numpy as np
import pytest
from jax_multislice import move_probe


# Test move_probe function.
@pytest.mark.parametrize("scan_position", [
    (0, 1),
    (5, 5),
    (10, 10),
    (3, 8)
])
def test_move_probe(scan_position):
    n = 12
    m = 14
    probe = jnp.zeros((n, m), dtype=jnp.float32)
    center = (n // 2, m // 2)
    center_row = center[1]
    center_col = center[0]
    probe = probe.at[center_col, center_row].set(1.0)

    moved_probe = move_probe(probe, scan_position)

    new_row = scan_position[0]
    new_col = scan_position[1]

    np.testing.assert_array_equal(moved_probe[new_row, new_col], 1.0, err_msg=f"Expected 1.0 at {(new_row, new_col)} and found {moved_probe[new_row, new_col]}")
