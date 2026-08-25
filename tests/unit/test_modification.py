"""Unit tests for dragen/substructure/modification.py functions that take a plain DataFrame."""
import pandas as pd

from dragen.substructure.modification import build_IDtree


def test_build_idtree_matches_unique_ids():
    rve_df = pd.DataFrame({
        'x': [0, 1, 0, 1, 0, 1, 0, 1],
        'y': [0, 0, 1, 1, 0, 0, 1, 1],
        'z': [0, 0, 0, 0, 1, 1, 1, 1],
        'GrainID': [1, 1, 1, 1, 2, 2, 2, 2],
        'packet_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'block_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'block_thickness': [0.5] * 8,
    })

    grain_nodes, packet_nodes, block_nodes = build_IDtree(rve_df)

    assert len(grain_nodes) == rve_df['GrainID'].nunique()
    assert len(packet_nodes) == rve_df['packet_id'].nunique()
    assert len(block_nodes) == rve_df['block_id'].nunique()
    for node in grain_nodes:
        assert node.points is not None and len(node.points) > 0
        assert len(node.children) > 0  # each grain owns at least one packet
