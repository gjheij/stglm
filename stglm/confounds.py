# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from .tsv import FillNA, SelectColumns


def init_confounds_select_wf(
    confound_names: list[str] | None = None,
    name: str | None = None,
    suffix: str | None = None,
):
    """
    We select different confound regressors depending on the feature we want to calculate.
    The confounds from the inputnode come from setting_adapter_wf.
    """
    name = "confounds_select_wf"
    if suffix is not None:
        name = f"{name}_{suffix}"

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold",
                "confounds",
                "confound_names",
                "mask",
                "vals"
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold",
                "confounds_selected",
                "confounds",
                "mask",
                "vals"
            ]
        ),
        name="outputnode",
    )
    workflow.connect(inputnode, "bold", outputnode, "bold")
    workflow.connect(inputnode, "confounds", outputnode, "confounds")
    workflow.connect(inputnode, "mask", outputnode, "mask")
    workflow.connect(inputnode, "vals", outputnode, "vals")

    if confound_names is not None:
        inputnode.inputs.confound_names = confound_names

    selectcolumns = pe.Node(SelectColumns(), name="selectcolumns")
    workflow.connect(inputnode, "confounds", selectcolumns, "in_file")
    workflow.connect(inputnode, "confound_names", selectcolumns, "column_names")

    workflow.connect(selectcolumns, "out_with_header", outputnode, "confounds_selected")

    return workflow