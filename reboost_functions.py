from __future__ import annotations

import awkward as ak
import numpy as np


def align_detectors(
    data_arr: ak.Array,
    field: str = "time",
    return_event_ids: bool = False,
) -> ak.Array:
    """Build jagged array [detector][event][hit], aligning events across detectors. Missing detector-event combinations become empty lists. Shape x * y * var.

    Parameters
    ----------
    data_arr
        Input array of shape n_detectors * var * {evtid, field, ...}. Needs to contain the field "evtid" and the specified field.
    field
        Field to extract and align (e.g. "time").
    return_event_ids
        If True, also return the array of unique event IDs corresponding to axis 1. Required for event building with other detector systems.

    Returns
    -------
    Jagged Awkward array [detector][event][hit] with the specified field (default time), with the event axis globally aligned across detectors.
    """
    # --- collect all event IDs ---
    all_evtids = ak.flatten(data_arr["evtid"])
    unique_evtids = np.unique(all_evtids)

    aligned = []

    for detector in data_arr:
        # Just sorts all hits by evtid
        det_sorted = detector[ak.argsort(detector["evtid"])]

        # group by event
        # This runs way quicker than reboost.shape.group.group_by_evtid but could be substituted.
        event_lengths = ak.run_lengths(det_sorted["evtid"])
        g = ak.unflatten(det_sorted, event_lengths)
        evtids = ak.firsts(g["evtid"])

        grouped_field = g[field]

        # Now align detector events correctly to the unique global events.
        evtids_list = ak.to_list(evtids)
        field_list = ak.to_list(grouped_field)
        local_map = dict(zip(evtids_list, field_list, strict=True))
        aligned_hits = [local_map.get(eid, []) for eid in unique_evtids]
        aligned_array = ak.Array(aligned_hits)
        aligned.append(aligned_array)
    try:
        aligned = ak.to_regular(aligned, axis=1)
    except ValueError as e:
        msg = "Can not convert event axis to regular. This means that not all detectors were correctly aligned to have the same number of events."
        raise ValueError(msg) from e
    if return_event_ids:
        return aligned, unique_evtids
    return aligned


def _group_candidate_triggers(group_event_hits, threshold, timegate_ns):
    """Helper function to find candidate trigger times for one event and subgroup (no deadtime here)."""
    if threshold <= 0:
        msg = "Group threshold must be > 0."
        raise ValueError(msg)

    all_times = []
    all_det_ids = []

    n_detectors = len(group_event_hits)
    for det_idx in range(n_detectors):
        det_hits = np.asarray(ak.to_numpy(group_event_hits[det_idx]), dtype=float)
        if det_hits.size == 0:
            continue
        all_times.append(det_hits)
        all_det_ids.append(np.full(det_hits.shape, det_idx, dtype=np.int32))

    if not all_times:
        return np.array([], dtype=float)

    times = np.concatenate(all_times)
    det_ids = np.concatenate(all_det_ids)

    order = np.argsort(times)
    times = times[order]
    det_ids = det_ids[order]

    candidate_times = []
    i = 0
    n_hits = len(times)

    while i < n_hits:
        t_start = times[i]
        seen_detectors = set()
        j = i

        while j < n_hits and (times[j] - t_start) <= timegate_ns:
            seen_detectors.add(int(det_ids[j]))
            if len(seen_detectors) > threshold:
                # Trigger is assigned to the time when threshold is reached.
                candidate_times.append(times[j])
                i = (
                    j + 1
                )  # We do not allow overlaps, so we jump to the next hit after the trigger time.
                break
            j += 1
        else:
            i += 1

    return np.array(candidate_times, dtype=float)


def build_hardware_triggers(
    data_array: ak.Array,
    multiplicity_threshold: int = 0,
    timegate: float = 60,
    trigger_deadtime: float = 0,
    trigger_groups: dict | None = None,
) -> ak.Array:
    """Build hardware trigger array based on multiplicity and light thresholds.

    .. note ::
        The selection is always strictly greater than the threshold not greater or equal.

    Parameters
    ----------
    data_array
        Jagged array [detector][event][hit] with hit times. Shape x * y * var.
    multiplicity_threshold : int
        Minimum number of different detectors triggering within timegate to trigger.
        Needs to be defined and > 0 if no trigger groups are defined.
    timegate
        Time window (in ns) to consider for multiplicity trigger.
    trigger_deadtime
        Time window (in ns) for which the system is unresponsive after a trigger.
    trigger_groups : dict | None
        Optional dict defining groups of detectors that are evaluated for their multiplicity threshold. Format:
        {group_name: {"detector_indices": (detector_indices),
                      "threshold": (threshold)
                      }
        },. If None, all detectors are treated as one group.

    Returns
    -------
    Awkward array with Trigger timestamps per event (ns), duplicated for each detector along axis 0. Shape x * y * var.
    The timestamps are the first hit times when the multiplicity threshold condition is satisfied,
    applying deadtime afterwards and not allowing overlaps in timegate (within the same trigger group.).
    """
    n_events = len(data_array[0])  # All detectors must have the same number of events.
    n_detectors = len(data_array)

    grouped_data = []
    thresholds = []
    if trigger_groups is not None:
        for _, group_info in trigger_groups.items():
            det_indices = group_info["detector_indices"]
            threshold = group_info["threshold"]
            grouped_data.append(data_array[det_indices])
            thresholds.append(threshold)
    else:
        if multiplicity_threshold == 0:
            msg = "Multiplicity threshold must be defined and > 0 if no trigger groups are defined."
            raise ValueError(msg)
        # Treat the entire detector array as one group.
        grouped_data.append(data_array)
        thresholds.append(multiplicity_threshold)

    del data_array  # free memory, we will use grouped_data from now on

    trigger_times_per_event: list[list[float]] = []

    for evt_idx in range(n_events):
        event_candidates = []

        for group_data, threshold in zip(grouped_data, thresholds, strict=True):
            # Takes all detectors in a group for the current event
            group_event_hits = group_data[:, evt_idx]
            group_candidates = _group_candidate_triggers(
                group_event_hits, threshold, timegate
            )
            if group_candidates.size:
                event_candidates.append(group_candidates)

        # If there is no candidate trigger in any group, this event was not triggered.
        if not event_candidates:
            trigger_times_per_event.append([])
            continue

        merged = np.sort(np.concatenate(event_candidates))

        if trigger_deadtime > 0:
            # First trigger is guaranteed.
            filtered = [float(merged[0])]
            dead_until = merged[0] + trigger_deadtime
            for t in merged[1:]:
                if t > dead_until:
                    filtered.append(float(t))
                    dead_until = t + trigger_deadtime
            trigger_times_per_event.append(filtered)
        else:
            trigger_times_per_event.append(merged.tolist())
    out = ak.Array(trigger_times_per_event)
    # Duplicate across the detector axis.
    return ak.concatenate([out[np.newaxis]] * n_detectors, axis=0)


def build_hits(
    data_array: ak.Array,
    hardware_triggers: ak.Array,
    trace_length: float,
    ns_per_sample: float,
    trigger_position: float = 0,
) -> ak.Array:
    """Build hits based on hardware triggers. Gives the pulse height in p.e. for each hardware trigger.

    Parameters
    ----------
    data_array
        Jagged array [detector][event][hit] with photon hit times. Shape x * y * var.
    hardware_triggers
        Jagged array [event][var] with trigger times. Shape y * var.
    trace_length
        Length of the trace around the trigger (in ns).
    ns_per_sample
        Time resolution of the trace (in ns) (bin width).
        If omitted, integration_time can be used as an alias.
    trigger_position, optional
        Position of the hardware trigger in the trace (in ns, relative to trace start).

    Returns
    -------
    Jagged Awkward array [detector][event][var] with the maximum pulse height in p.e. for each hardware trigger.
    If there are no hardware triggers for an event, the var axis will be empty for that event.
    """
    if ns_per_sample <= 0:
        msg = "ns_per_sample must be > 0."
        raise ValueError(msg)
    if trace_length <= 0:
        msg = "trace_length must be > 0."
        raise ValueError(msg)

    n_samples_float = trace_length / ns_per_sample
    n_samples = int(np.round(n_samples_float))
    if not np.isclose(n_samples_float, n_samples):
        msg = "trace_length must be an integer multiple of ns_per_sample."
        raise ValueError(msg)

    n_detectors = len(data_array)
    if n_detectors == 0:
        return ak.Array([])

    n_events = len(data_array[0])
    if len(hardware_triggers) != n_events:
        msg = "hardware_triggers must have the same number of events as data_array."
        raise ValueError(msg)

    trigger_lengths = ak.to_numpy(ak.num(hardware_triggers, axis=1))
    max_triggers = int(np.max(trigger_lengths)) if trigger_lengths.size else 0

    if max_triggers == 0:
        return ak.Array(np.empty((n_detectors, n_events, 0), dtype=np.int32))

    rel_times = (
        data_array[:, :, None, :]
        - hardware_triggers[None, :, :, None]
        + trigger_position
    )
    mask = (rel_times >= 0) & (rel_times < trace_length)
    sample_idx = ak.values_astype(np.floor(rel_times[mask] / ns_per_sample), np.int64)

    sample_flat = ak.to_numpy(ak.flatten(sample_idx, axis=None))
    flat_result = np.zeros(n_detectors * n_events * max_triggers, dtype=np.int32)

    if sample_flat.size:
        det_idx = ak.to_numpy(
            ak.flatten(
                ak.broadcast_arrays(ak.local_index(sample_idx, axis=0), sample_idx)[0],
                axis=None,
            )
        )
        evt_idx = ak.to_numpy(
            ak.flatten(
                ak.broadcast_arrays(ak.local_index(sample_idx, axis=1), sample_idx)[0],
                axis=None,
            )
        )
        trig_idx = ak.to_numpy(
            ak.flatten(
                ak.broadcast_arrays(ak.local_index(sample_idx, axis=2), sample_idx)[0],
                axis=None,
            )
        )

        group_flat = ((det_idx * n_events) + evt_idx) * max_triggers + trig_idx
        keys = group_flat * n_samples + sample_flat

        unique_keys, counts = np.unique(keys, return_counts=True)
        group_ids = unique_keys // n_samples
        group_starts = np.flatnonzero(np.r_[True, np.diff(group_ids) != 0])
        group_ids = group_ids[group_starts].astype(np.int64)
        max_counts = np.maximum.reduceat(counts, group_starts).astype(np.int32)
        flat_result[group_ids] = max_counts

    flat_result = flat_result.reshape(n_detectors * n_events, max_triggers)
    repeated_lengths = np.tile(trigger_lengths, n_detectors)
    valid_trig_mask = np.arange(max_triggers)[None, :] < repeated_lengths[:, None]
    trimmed = flat_result[valid_trig_mask]

    hits = ak.unflatten(ak.Array(trimmed), repeated_lengths, axis=0)
    hits = ak.unflatten(hits, np.full(n_detectors, n_events), axis=0)
    return ak.to_regular(hits, axis=1)


def build_traces(
    data_array: ak.Array,
    hardware_triggers: ak.Array,
    trace_length: float,
    ns_per_sample: float,
    trigger_position: float,
) -> ak.Array:
    """Build detector traces around hardware triggers. This will inflate the input data adding a lot of empty samples, so only use if you need the full trace information.

    Parameters
    ----------
    data_array
        Jagged array [detector][event][hit] with hit times. Shape x * y * var.
    hardware_triggers
        Jagged array [event][var] with trigger times. Shape y * var.
    trace_length
        Length of the trace to build around the trigger (in ns).
    ns_per_sample
        Time resolution of the trace (in ns) (bin width).
    trigger_position
        Position of the hardware trigger in the trace (in ns, relative to trace start).

    Returns
    -------
    Jagged Awkward array [detector][event][var][sample] with photon hit counts per sample. Shape x * y * var * z.
    If there are no hardware triggers for an event, the var axis will be empty for that event.
    """
    if ns_per_sample <= 0:
        msg = "ns_per_sample must be > 0."
        raise ValueError(msg)
    if trace_length <= 0:
        msg = "trace_length must be > 0."
        raise ValueError(msg)

    n_samples_float = trace_length / ns_per_sample
    n_samples = int(np.round(n_samples_float))
    if not np.isclose(n_samples_float, n_samples):
        msg = "trace_length must be an integer multiple of ns_per_sample."
        raise ValueError(msg)

    n_detectors = len(data_array)
    if n_detectors == 0:
        return ak.Array([])

    n_events = len(data_array[0])
    if len(hardware_triggers) != n_events:
        msg = "hardware_triggers must have the same number of events as data_array."
        raise ValueError(msg)

    trigger_lengths = ak.to_numpy(ak.num(hardware_triggers, axis=1))
    max_triggers = int(np.max(trigger_lengths)) if trigger_lengths.size else 0

    # If no triggers, return empty array with correct shape.
    if max_triggers == 0:
        empty = np.empty((n_detectors, n_events, 0, n_samples), dtype=np.int16)
        return ak.Array(empty)

    # Broadcast to detector x event x trigger x hit and build sample indices in one shot.
    rel_times = (
        data_array[:, :, None, :]
        - hardware_triggers[None, :, :, None]
        + trigger_position
    )
    mask = (rel_times >= 0) & (rel_times < trace_length)
    # Only keep the relevant times for which we will build the traces, and convert to sample indices.
    sample_idx = ak.values_astype(np.floor(rel_times[mask] / ns_per_sample), np.int16)

    # Build flattened detector/event/trigger/sample coordinates and count with one bincount.
    det_idx = ak.broadcast_arrays(ak.local_index(sample_idx, axis=0), sample_idx)[0]
    evt_idx = ak.broadcast_arrays(ak.local_index(sample_idx, axis=1), sample_idx)[0]
    trig_idx = ak.broadcast_arrays(ak.local_index(sample_idx, axis=2), sample_idx)[0]

    # Flatt 1-D arrays just telling the indice for each of our four dimensions (detector, event, trigger, sample) for each hit.
    # So the length is total number of hits across all events and detectors.
    s_flat = ak.to_numpy(ak.flatten(sample_idx, axis=None))
    d_flat = ak.to_numpy(ak.flatten(det_idx, axis=None))
    e_flat = ak.to_numpy(ak.flatten(evt_idx, axis=None))
    t_flat = ak.to_numpy(ak.flatten(trig_idx, axis=None))

    # This is a projection, so it only tells where each actual hit would be placed in this 1-d array.
    # (Two hits can have the same index, this means they would be placed in the same sample (of the same trace))
    linear_idx = (
        ((d_flat * n_events) + e_flat) * max_triggers + t_flat
    ) * n_samples + s_flat

    total_bins = n_detectors * n_events * max_triggers * n_samples
    # Doing the bincount now on the "where each hit would be" array gives us the correctly binned counts, just in a flattened form.
    counts = np.bincount(linear_idx, minlength=total_bins).astype(np.int16)
    # Now we need to restore our 4-D structure (detector x event x trigger x sample).
    counts = counts.reshape(n_detectors * n_events, max_triggers, n_samples)

    # Remove padded trigger slots and restore detector x event axes.
    repeated_lengths = np.tile(trigger_lengths, n_detectors)
    valid_trig_mask = np.arange(max_triggers)[None, :] < repeated_lengths[:, None]
    trimmed = counts[valid_trig_mask]

    traces = ak.unflatten(ak.Array(trimmed), repeated_lengths, axis=0)
    traces = ak.unflatten(traces, np.full(n_detectors, n_events), axis=0)
    return ak.to_regular(traces, axis=1)
