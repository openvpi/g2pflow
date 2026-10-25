"""Beam-search path decoding over the candidate lattice."""

import math

from .lexicon import Edge


def _forced_edges(edges: list[Edge], length: int) -> set[int]:
    locked = [(i, e) for i, e in enumerate(edges) if e.locked]
    locked.sort(key=lambda item: (item[1].start,
                                  -(item[1].end - item[1].start),
                                  -item[1].confidence))
    selected: set[int] = set()
    occupied: list[tuple[int, int]] = []
    for index, edge in locked:
        if any(edge.start < end and start < edge.end
               for start, end in occupied):
            continue
        selected.add(index)
        occupied.append((edge.start, edge.end))
    if any(start < 0 or end > length for start, end in occupied):
        raise ValueError("locked edge is outside the input")
    return selected


def _allowed_edges(edges: list[Edge], length: int) -> set[int]:
    forced = _forced_edges(edges, length)
    if not forced:
        return set(range(len(edges)))
    occupied = [(edges[i].start, edges[i].end) for i in forced]
    allowed = set(forced)
    for index, edge in enumerate(edges):
        if index in forced:
            continue
        if any(edge.start < end and start < edge.end
               for start, end in occupied):
            continue
        allowed.add(index)
    return allowed


def decode_beam(scores, edges: list[Edge], length: int, beam_width: int):
    allowed = _allowed_edges(edges, length)
    values = [float(s) for s in scores]
    by_start: list[list[int]] = [[] for _ in range(length)]
    for index in allowed:
        edge = edges[index]
        if 0 <= edge.start < edge.end <= length:
            by_start[edge.start].append(index)
    beam: list[list[tuple[float, tuple[int, ...]]]] = [
        [] for _ in range(length + 1)]
    beam[0].append((0.0, ()))
    for start in range(length):
        if not beam[start]:
            continue
        beam[start].sort(key=lambda item: -item[0])
        for total, path in beam[start][:beam_width]:
            for edge_idx in by_start[start]:
                beam[edges[edge_idx].end].append(
                    (total + values[edge_idx], path + (edge_idx,)))
    complete = sorted(beam[length], key=lambda item: -item[0])[:beam_width]
    if not complete:
        raise ValueError("candidate graph has no path covering the input")
    return [(score, list(path)) for score, path in complete]
