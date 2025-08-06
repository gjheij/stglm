# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import pathlib
import zipfile
import subprocess
import nibabel as nb
from pathlib import Path

AnyPath = pathlib.Path | zipfile.Path

class File:
    def __init__(
        self,
        path: str,
        datatype: str,
        suffix: str | None = None,
        extension: str | None = None,
        **kwargs,
    ):
        self.path = path
        self.datatype = datatype
        self.suffix = suffix
        self.extension = extension

        self.tags: dict[str, str] = dict()
        self.intended_for: dict | None = None

        self.metadata: dict = dict()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if not hasattr(other, "path"):
            return False
        return self.path == other.path
        
def split_ext(path: str | AnyPath):
    """Splits filename and extension (.gz safe)
    >>> split_ext('some/file.nii.gz')
    ('file', '.nii.gz')
    >>> split_ext('some/other/file.nii')
    ('file', '.nii')
    >>> split_ext('otherext.tar.gz')
    ('otherext', '.tar.gz')
    >>> split_ext('text.txt')
    ('text', '.txt')

    Adapted from niworkflows
    """
    from pathlib import Path

    if isinstance(path, str):
        path = Path(path)

    name = str(path.name)

    safe_name = name
    for compound_extension in [".gz", ".xz"]:
        safe_name = safe_name.removesuffix(compound_extension)

    stem = Path(safe_name).stem
    return stem, name[len(stem) :]

def ravel(obj):
    if isinstance(obj, (str, dict)):
        return obj

    try:
        ret = []
        for val in obj:
            raveled_val = ravel(val)
            if not isinstance(raveled_val, (str, dict)):
                try:
                    ret.extend(raveled_val)
                    continue
                except TypeError:
                    pass
            ret.append(raveled_val)
        return ret

    except TypeError:
        return obj
    except StopIteration:
        return []  # empty input
    except Exception:
        return []

def get_node_by_name(wf_or_graph, name):
    """
    Return a node by name from either a Workflow or an executed graph.
    
    Parameters:
        wf_or_graph: Workflow or WorkflowRun (from .run())
        name: str, name of the node

    Returns:
        Node or None
    """
    # Case 1: it's an unexecuted Workflow object
    if hasattr(wf_or_graph, 'get_node'):
        try:
            return wf_or_graph.get_node(name)
        except Exception:
            return None

    # Case 2: it's a NodeView from an executed workflow
    return next((n for n in wf_or_graph.nodes if n.name == name), None)


def get_betas_from_graph(wf_or_graph, mode="file"):
    betas = {}
    for model in ["lsa", "lss"]:
        node = get_node_by_name(wf_or_graph, f"merge_betas_{model}")
        if node is None:
            print(f"Node 'merge_betas_{model}' not found.")
            betas[model] = None
        elif not hasattr(node, "result"):
            print(f"Node 'merge_betas_{model}' has not run.")
            betas[model] = None
        else:
            file_path = node.result.outputs.merged_file
            if mode == "file":
                set_key = file_path
            elif mode == "niimg":
                set_key = nb.load(file_path)
            elif mode == "array":
                set_key = nb.load(file_path).get_fdata()
            else:
                raise ValueError(f"mode must be one of 'file', 'niimg', or 'array', not '{mode}'")

            betas[model] = set_key

    return betas

def bids_filename(tags, suffix):
    fname = f"sub-{tags['subject']}"
    if "session" in tags:
        fname += f"_ses-{tags['session']}"
    fname += f"_task-{tags['task']}"
    if "run" in tags:
        fname += f"_run-{tags['run']}"
    fname += f"_desc-{suffix}_betas.nii.gz"
    return fname

def generate_and_write_combined_json(
    bold, condition_files, condition_names,
    condition_units, tags, vals, trial_list,
    feature_dict, workdir, derivatives_dir
):
    import json
    from pathlib import Path

    metadata = {
        "BasedOn": [bold, condition_files],
        "Conditions": condition_names,
        "ConditionUnits": condition_units,
        **tags,
        **vals,
        **trial_list
    }

    full_output = {
        "Settings": feature_dict,
        "Metadata": metadata
    }

    # Construct BIDS-style filename
    parts = []
    for key in ["subject", "session", "task", "run", "space"]:
        if key in tags:
            if key in ["subject", "session"]:
                kk = key[:3]
            else:
                kk = key
                
            parts.append(f"{kk}-{tags[key]}")

    fname = "_".join(parts)
    fname += "_desc-stglm.json"

    out_path = Path(derivatives_dir) / f"sub-{tags['subject']}"
    if "session" in tags:
        out_path = out_path / f"ses-{tags['session']}"
    out_path = out_path / "func" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=4)

    return str(out_path)



def merge_confounds(confound_files, out_file=None):
    import pandas as pd
    from pathlib import Path

    confound_dfs = [pd.read_csv(f, sep='\t') for f in confound_files]
    merged = pd.concat(confound_dfs, axis=0).reset_index(drop=True)

    if out_file:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_file, sep='\t', index=False)
        return str(out_file)
    
    return merged


def merge_events(event_files, bold_files, tr, out_file=None):
    import pandas as pd
    import nibabel as nb
    from pathlib import Path

    offset = 0.0
    merged_events = []
    
    for i, (event_file, bold_file) in enumerate(zip(event_files, bold_files)):
        df = pd.read_csv(event_file, sep='\t')
        nvols = nb.load(bold_file).shape[-1]
        df["onset"] = df["onset"] + offset
        merged_events.append(df)
        offset += nvols * tr
    
    merged_df = pd.concat(merged_events, axis=0).reset_index(drop=True)

    if out_file:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(out_file, sep='\t', index=False)
        return str(out_file)

    return merged_df


def merge_bold_fsl(bold_files, out_file=None):
    """
    Merge BOLD files using FSL's fslmerge to avoid memory overload.

    Parameters
    ----------
    bold_files : list of str
        List of BOLD file paths to concatenate.
    out_file : str or Path, optional
        Output file path. If not provided, it will be created in /tmp.

    Returns
    -------
    str
        Path to the merged BOLD file.
    """
    bold_files = [str(f) for f in bold_files]  # Ensure strings
    if out_file is None:
        out_file = Path("/tmp") / "merged_bold.nii.gz"
    else:
        out_file = Path(out_file)

    cmd = ["fslmerge", "-t", str(out_file)] + bold_files
    subprocess.run(cmd, check=True)

    return str(out_file)

def generate_sanitized_metadata(tags, vals, trial_list, **kwargs):
    import json

    def safe_val(val):
        # Convert to serializable types
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        if isinstance(val, (list, tuple)):
            return [safe_val(v) for v in val]
        if isinstance(val, dict):
            return {str(k): safe_val(v) for k, v in val.items()}
        # Fallback to string
        return str(val)

    out = {}

    # Flatten and sanitize tags, vals
    for d in [tags, vals]:
        for k, v in d.items():
            out[str(k)] = safe_val(v)

    # Add TrialList (flatten nested key if needed)
    if isinstance(trial_list, dict) and "TrialList" in trial_list:
        out["TrialList"] = safe_val(trial_list["TrialList"])
    else:
        out["TrialList"] = safe_val(trial_list)

    # Add other kwargs like Conditions, etc.
    for k, v in kwargs.items():
        if k in out:
            k = f"extra_{k}"  # avoid overwrite
        out[k] = safe_val(v)

    # Handle special merging for Sources + ConditionFile
    based_on = out.pop("BasedOn", None)
    condition_file = out.pop("ConditionFile", None)

    merged_sources = []
    if condition_file:
        merged_sources += condition_file if isinstance(condition_file, list) else [condition_file]
    if based_on:
        merged_sources += based_on if isinstance(based_on, list) else [based_on]

    if merged_sources:
        # De-duplicate while preserving order
        seen = set()
        merged = []
        for s in merged_sources:
            if s not in seen:
                seen.add(s)
                merged.append(safe_val(s))
        out["Sources"] = merged

    return out
