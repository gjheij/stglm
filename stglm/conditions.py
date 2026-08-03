# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from math import isclose
import numpy as np

from nipype.interfaces.base import (
    Bunch,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

import pandas as pd
from .events import ConditionFile
from stglm.logger import get_logger


logger = get_logger(__name__)

class ApplyConditionOffsetInputSpec(TraitedSpec):
    subject_info = traits.Any()
    scan_start = traits.Float()


class ApplyConditionOffsetOutputSpec(TraitedSpec):
    subject_info = traits.Any()


class ApplyConditionOffset(SimpleInterface):
    input_spec = ApplyConditionOffsetInputSpec
    output_spec = ApplyConditionOffsetOutputSpec

    def _run_interface(self, runtime):
        subject_info = self.inputs.subject_info
        scan_start = self.inputs.scan_start

        conditions = subject_info.conditions
        onsets = subject_info.onsets
        durations = subject_info.durations

        for condition, condition_onsets in zip(conditions, onsets, strict=False):
            for i, onset in enumerate(condition_onsets):
                onset -= scan_start
                if onset < 0:
                    logger.warning(f'Condition "{condition}" onset truncated from {onset:f} to {0.0:f} s.')
                    onset = 0
                condition_onsets[i] = onset

        self._results["subject_info"] = Bunch(
            conditions=conditions,
            onsets=onsets,
            durations=durations
        )

        return runtime


class ParseConditionFileInputSpec(TraitedSpec):
    in_any = traits.Any(
        mandatory=True,
    )

    condition_names = traits.List(traits.Str(), desc="filter conditions")

    contrasts = traits.List(
        traits.Tuple(
            traits.Str,
            traits.Enum("T"),
            traits.List(traits.Str),
            traits.List(traits.Float),
        ),
    )


class ParseConditionFileOutputSpec(TraitedSpec):
    subject_info = traits.Any()

    contrasts = traits.List(
        traits.Tuple(
            traits.Str,
            traits.Enum("T"),
            traits.List(traits.Str),
            traits.List(traits.Float),
        ),
    )

    condition_names = traits.List(traits.Str)
    contrast_names = traits.List(traits.Str)


class ParseConditionFile(SimpleInterface):
    input_spec = ParseConditionFileInputSpec
    output_spec = ParseConditionFileOutputSpec

    def _run_interface(self, runtime):
        # parse input files
        cf = ConditionFile(data=self.inputs.in_any)
        conditions = cf.conditions
        onsets = cf.onsets
        durations = cf.durations

        # use only selected conditions
        if isdefined(self.inputs.condition_names):
            conditions, onsets, durations = cf.select(self.inputs.condition_names)

        # remove empty or invalid conditions
        filtered_conditions = [
            (condition, onset, duration)
            for condition, onset, duration in zip(conditions, onsets, durations, strict=False)
            if len(onset) == len(duration) and len(onset) > 0
        ]
        assert len(filtered_conditions) > 0, "No events found"
        conditions, onsets, durations = map(list, zip(*filtered_conditions, strict=False))

        # filter and re-write contrasts based on available conditions
        if isdefined(self.inputs.contrasts):
            contrasts = self.inputs.contrasts
            new_contrasts = list()
            for name, contrast_type, contrast_conditions, contrast_values in contrasts:
                if any(
                    condition not in conditions  # is missing
                    and not isclose(value, 0)  # but is part of the contrast
                    for condition, value in zip(contrast_conditions, contrast_values, strict=False)
                ):
                    continue

                filtered_contrast = [
                    (condition, value)
                    for condition, value in zip(contrast_conditions, contrast_values, strict=False)
                    if condition in conditions
                ]
                contrast_conditions, contrast_values = map(list, zip(*filtered_contrast, strict=False))

                new_contrasts.append(
                    (
                        name,
                        contrast_type,
                        contrast_conditions,
                        contrast_values,
                    )
                )

            self._results["contrast_names"] = [name for name, _, _, _ in new_contrasts]
            self._results["contrasts"] = new_contrasts

        self._results["condition_names"] = list(conditions)
        self._results["subject_info"] = Bunch(conditions=conditions, onsets=onsets, durations=durations)

        return runtime

def events_to_glmsingle_dm(
    events_df: pd.DataFrame,
    n_vols: int,
    tr: float = 1.0,
    include_trials: list[str] | None = None,
    event_col: str="trial_type"
    ) -> pd.DataFrame:
    """
    Convert a BIDS events.tsv dataframe into a GLMsingle-style design matrix.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns ['onset', 'duration', event_col].
        Onsets and durations are in seconds.
    n_vols : int
        Total number of volumes in your run.
    tr : float, optional
        Repetition time, in seconds. Defaults to 1.0 (i.e. onsets already in TR-units).
    include_trials : list of str, optional
        Trial types to include from the design. If None, all events are considered.
    event_col: str, optional
        Column name that holds the event names. Default is 'trial_type'

    Returns
    -------
    dm : pd.DataFrame
        Shape = (n_vols, n_conditions).  Each column is a stick regressor
        for one trial type; a 1 at volume *i* means an event of that
        type began at that TR, 0 otherwise.
    """

    if include_trials is not None:
        events_df = events_df[events_df[event_col].isin(include_trials)].copy()

    logger.info(f"design shape: {events_df.shape}")

    # 1. Identify unique conditions (sorted for reproducibility)
    conds = sorted(events_df[event_col].unique())
    n_conds = len(conds)

    # 2. Initialize empty design matrix
    dm = np.zeros((n_vols, n_conds), dtype=int)

    # 3. Map condition names → column indices
    cond_to_idx = {cond: i for i, cond in enumerate(conds)}

    # 4. Fill in sticks
    for _, row in events_df.iterrows():
        onset_sec = row['onset']
        cond      = row[event_col]

        # convert onset (s) → volume index
        vol_idx = int(round(onset_sec / tr))

        if 0 <= vol_idx < n_vols:
            dm[vol_idx, cond_to_idx[cond]] = 1

    return dm
