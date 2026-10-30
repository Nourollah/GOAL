"""Literature ablations of SIMURGH, one mechanism per file.

Each model subclasses :class:`~goal.ml.nn.models.simurgh.dev.monolithic.SimurghDev`,
changes one section, starts zero-initialised (a fresh model equals
SimurghDev) and is registered as ``simurgh_dev_abl_<mechanism>``.  They
are kept only if they pass the selection rule of the ablation programme.
"""
