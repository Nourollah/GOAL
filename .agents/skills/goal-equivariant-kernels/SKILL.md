---
name: goal-equivariant-kernels
description: Guides e3nn and NVIDIA cuEquivariance work in GOAL models. Covers irreps strings and parity, tensor-product instruction modes (uvu, uvw) with per-edge radial weights, data layouts (mul_ir versus ir_mul), basis-order checks for l = 1 features, symmetric contractions and product-basis multiplicities, optional cuEquivariance kernels behind an import guard (planned, not yet in GOAL), and float64 parity and equivariance tests. Use when writing or debugging equivariant layers, switching kernels, or chasing shape, layout or silent equivariance errors.
license: MIT
---

# e3nn and cuEquivariance in GOAL

GOAL requires e3nn 0.5 or later (`pyproject.toml`: `e3nn>=0.5`; `pixi.lock` pins 0.6.0). Packages that pin e3nn 0.4.x run only in separate environments; exchange plain tensors with them, never modules or pickles.

## Irreps and parity

- An irreps string such as `"128x0e + 128x1o + 128x2e"` lists multiplicity, degree l and parity (`e` even, `o` odd).
- Spherical harmonics of a direction have parity (−1)^l: 0e, 1o, 2e, 3o. A Cartesian vector is `1o`; a cross product of two vectors is `1e`.
- 1o ⊗ 1o = 0e + 1e + 2e.

## Tensor products with per-edge weights

```python
from e3nn import o3

tp = o3.TensorProduct(
    irreps_node, irreps_sh, irreps_msg,
    instructions=[(i1, i2, i_out, "uvu", True) for (i1, i2, i_out) in paths],
    shared_weights=False, internal_weights=False,
)
weights = radial_mlp(edge_basis)                  # shape (num_edges, tp.weight_numel)
messages = tp(node_feats[edge_src], edge_sh, weights)
```

- `"uvu"`: one weight per input channel and path; the output keeps the first input's multiplicity. This is the usual message form, with the radial network supplying the weights edge by edge.
- `"uvw"` and `FullyConnectedTensorProduct` mix channels and cost far more.
- Keep e3nn's default normalisations (`irrep_normalization="component"`, `path_normalization="element"`) unless you change them on purpose in every kernel; they set the scale at initialisation.

## Layouts: the silent error

- e3nn stores each irreps block as multiplicity × (2l + 1), flattened multiplicity-major (`mul_ir`).
- cuEquivariance can run in `mul_ir` (e3nn-compatible) or `ir_mul`. A tensor in one layout fed to a layer that expects the other has the right shape and the wrong physics.
- Convert explicitly at every boundary, and test with fixed random inputs. Never infer a layout from shapes.

## Basis order of l = 1 features

Do not assume that an l = 1 feature is ordered (x, y, z). Before comparing one with a Cartesian vector, compare `o3.Irreps("1o").D_from_matrix(R)` with `R` for a random rotation and apply the permutation that makes them equal.

## cuEquivariance kernels

- Status: no cuEquivariance package is listed in `pyproject.toml` or `pixi.lock`, and GOAL has no cuEquivariance code path. The intended setup is an optional dependency of the `cuda` pixi feature (`cuequivariance`, `cuequivariance-torch` and the CUDA ops package), imported behind a guard (planned):

```python
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
except ImportError:  # CPU-only or optional dependency missing
    cue = cuet = None
```

- SIMURGH (`simurgh_dev`) already accepts `kernels: auto | cuequivariance | e3nn` (default `auto`). Today `auto` resolves to e3nn and `cuequivariance` raises `NotImplementedError`.
- Target behaviour of the switch: `auto` uses cuEquivariance only when it imports and the device is CUDA; an explicit `cuequivariance` raises when it is unavailable. No silent fallback (planned).
- Constructor arguments change between releases (layouts, math dtype, fallback switches, weight sharing). Read `help(cuet.<Class>)` for the installed version before writing a call (planned).
- Weight layouts of symmetric contractions differ between implementations. Convert weights with a tested function, never by reshaping until the shapes match.

## Parity tests, required before a kernel produces results

1. Build both implementations with the same weights.
2. Float64 with fixed random inputs (use the CPU fallback where a GPU kernel lacks float64): outputs agree to about 1e-10 relative; on the GPU in float32 to about 1e-5.
3. Compare first and second derivatives, because force training backpropagates through the force: `torch.autograd.gradcheck` and `gradgradcheck` on a small case.
4. Equivariance: rotate the inputs with `irreps.D_from_matrix(R)` and check that the outputs rotate the same way.
5. Store fixtures as plain tensors (inputs, weights, outputs) with neutral file names in the tests' fixtures folder, so that the test needs no external package.

## Product-basis sanity check

Symmetric tensor powers of one irreps block have exact multiplicities (Young-tableau counts). For a single l = 1 block, Sym³ contains exactly one L = 1 and one L = 3, and nothing else. Check symmetric-contraction and ladder bases against such counts at each correlation order before training.

## Done when

- [ ] The parity test passes in float64 and in float32.
- [ ] Every new layer passes the equivariance test.
- [ ] `kernels: e3nn` and `kernels: cuequivariance` give the same energies and forces on a test batch (planned).
