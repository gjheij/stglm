# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import logging
import os
import pathlib
import zipfile
from os import path as op
from pathlib import Path
from shutil import copyfile
from typing import Any, Iterator

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