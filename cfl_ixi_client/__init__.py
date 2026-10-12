"""Clustered Federated Learning (Sattler et al., 2019) on FLamby Fed-IXI with
9 clients: each of the 3 hospitals is partitioned into 3 sub-clients.

The hospital of a client is therefore a KNOWN ground-truth cluster, which makes
the recovered partition scorable (ARI, separation gap) -- impossible with the
3 natural clients of Fed-IXI.
"""
__all__ = ["config", "data_utils", "models", "metrics", "stragglers", "fl_devices"]
