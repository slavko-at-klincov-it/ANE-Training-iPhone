#!/usr/bin/env python3
"""
Create a minimal CoreML model (.mlpackage) targeting the Apple Neural Engine.
Identity 1x1 convolution: 256 channels, spatial 64.
Input/Output: float16 multiarray [1, 256, 1, 64].
"""

import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# 1. Ensure coremltools is installed
# ---------------------------------------------------------------------------
try:
    import coremltools as ct
except ImportError:
    print("coremltools not found – installing …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coremltools"])
    import coremltools as ct

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.milproto import load as milproto_load

# ---------------------------------------------------------------------------
# 2. Build the model with the MIL builder
# ---------------------------------------------------------------------------
# Identity 1×1 convolution weights: out_channels x in_channels x kH x kW
# For an identity mapping we use a per-channel weight of 1.
weight = np.zeros((256, 256, 1, 1), dtype=np.float16)
for i in range(256):
    weight[i, i, 0, 0] = 1.0


@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, 256, 1, 64), dtype=ct.converters.mil.mil.types.fp16)],
    opset_version=ct.target.iOS16,
)
def identity_conv(x):
    return mb.conv(x=x, weight=weight, name="identity_conv")


# Convert the MIL program to a CoreML model
model = ct.convert(
    identity_conv,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,  # allows ANE
)

# Tag input/output as float16 multiarrays
spec = model.get_spec()

# Input
inp = spec.description.input[0]
inp.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16

# Output
out = spec.description.output[0]
out.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16

model = ct.models.MLModel(spec, weights_dir=model.weights_dir)

# ---------------------------------------------------------------------------
# 3. Save as .mlpackage
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MLPACKAGE_PATH = os.path.join(OUTPUT_DIR, "IdentityConv.mlpackage")
MLMODELC_DIR = OUTPUT_DIR  # compiled bundle goes here

model.save(MLPACKAGE_PATH)
print(f"Saved mlpackage -> {MLPACKAGE_PATH}")

# ---------------------------------------------------------------------------
# 4. Compile to .mlmodelc
# ---------------------------------------------------------------------------
print("Compiling to .mlmodelc …")
subprocess.check_call([
    "xcrun", "coremlcompiler", "compile",
    MLPACKAGE_PATH,
    MLMODELC_DIR,
])
compiled_path = os.path.join(MLMODELC_DIR, "IdentityConv.mlmodelc")
if os.path.isdir(compiled_path):
    print(f"Compiled model -> {compiled_path}")
else:
    print("Warning: compiled .mlmodelc not found at expected path.")
    print("Check the output directory for the compiled bundle.")
