# HLoD: Screen-Space Hierarchical Level-of-Detail Traversal for 3D Gaussian Splatting

Implementation of HLoD by **Jan U. Müller, Jakob Sommer and Reinhard Klein**, published in *Computer Graphics Forum* 45(7), 2026.

[Paper](https://diglib.eg.org/items/429b6aa4-7897-4217-b297-272992d745ae) · [DOI](https://doi.org/10.1111/cgf.70636)

HLoD trains an explicit Gaussian parent–child forest. At each view, a screen-space traversal selects a hierarchy cut: sufficiently small sibling groups are replaced by their trained parents. Training alternates adaptive traversal with current-level and randomly capped supervision.

## Installation

Linux, an NVIDIA GPU, a compatible driver, a C++ compiler and a CUDA toolkit containing `nvcc` are required for GPU training and rendering. The release checks use Python 3.10, PyTorch 2.5.1, CUDA 12.4 and an RTX 4090. CPU tests also exercise the PyTorch traversal fallback; the rasterizers and training require CUDA.

Clone **this HLoD repository**, then run the following commands from its root.
```bash
conda env create -f environment.yml
conda activate hlod
./scripts/install_extensions.sh
```

The compiler prerequisites on Ubuntu can be installed with `sudo apt install build-essential libglm-dev`. Use a host compiler supported by CUDA 12.4 (GCC 11 is used in the release checks). The environment pins the CUDA compiler and development headers to 12.4, matching PyTorch. Check `nvcc --version` against `python -c "import torch; print(torch.version.cuda)"` if using a pre-existing environment. To select an installed system toolkit instead:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
```

HLoD respects an explicit, complete `CUDA_HOME` before automatic toolkit detection. The selected prefix must contain both `bin/nvcc` and `include/crt/host_defines.h`. If Conda also configures an incompatible host compiler, select a supported system compiler with `export CC=/usr/bin/gcc CXX=/usr/bin/g++` and `unset NVCC_PREPEND_FLAGS` before building extensions.

Alternatively, install PyTorch 2.5.1 and torchvision 0.20.1 with the CUDA 12.4 wheels in a Python 3.10 virtual environment, then:

```bash
python -m pip install -r requirements.txt
./scripts/install_extensions.sh
```

The extension installer pins the INRIA rasterizer and `simple-knn` to specific commits. It applies [our rasterizer patch](patches/diff-gaussian-rasterization-low-pass.patch), which exposes the low-pass covariance dilation in **both forward and backward** and optionally reports Gaussian–tile intersections. An unpatched upstream rasterizer is not compatible with the HLoD `low_pass` argument. The hierarchy extension is compiled on first use. Set `MAX_JOBS=2` to limit build memory, if needed.

`gsplat==1.5.3` is included as an alternative backend. Use `--renderer gsplat` for training or rendering; this changes the rasterization/densification configuration and should be reported when comparing results. For gsplat, absolute-gradient densification is enabled by default, with an automatically adjusted default gradient threshold. `--no-absgrad` disables it.

## Data and training

Supported input formats are undistorted COLMAP scenes (`sparse/0/`, images) and NeRF synthetic scenes (`transforms_train.json`, `transforms_test.json`, images). Supply absolute or relative dataset paths with `-s`; outputs go to `-m`

Training keeps images on CPU by default; `--data_device cuda` explicitly moves them to GPU memory. Rendering independently supports `--load_image_device cpu` to reduce image residency on the GPU.

```bash
python train.py -s /path/to/scene -m output/scene
```

Only supply options that differ from the defaults. By default, training uses all views; add `--eval` to enable the train/test split. For COLMAP, evaluation holds out every eighth view unless the dataset provides explicit split files. Use `--llffhold` to change it. For NeRF synthetic evaluation, the provided train/test transforms are used.

The shell wrapper continues to enable evaluation splitting. For example, a paper dataset run with a scene-specific resolution is:

```bash
./train.sh /path/to/mipnerf360/garden output/garden -r 4
```

Paper datasets and example resolutions:

| Dataset | Scenes | Resolution argument |
| --- | --- | --- |
| [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | bicycle, flowers, garden, stump, treehill | `-r 4` |
| Mip-NeRF 360 | room, counter, kitchen, bonsai | `-r 2` |
| [Tanks and Temples](https://www.tanksandtemples.org/download/) | train, truck | `-r 1` |
| Deep Blending | drjohnson, playroom | `-r 1` |

The default HLoD configuration uses five levels with 10k/15k/20k/25k/30k iterations, initial scale lower bound `0.05`, scale ratio `2`, two children per expansion, training threshold `3`, `lambda_scale_upper=0.01`, `lambda_maha=0.005`, `current_prob=0.03`, `prefix_prob=0.05`, mip floor enabled and `low_pass=0`.`leafes_to_coarse` starts from all hierarchy leaves, including branches that ended at an earlier level. `fine_to_coarse` starts from the finest birth level only. `coarse_to_fine` is also supported.

Use `--lod_iterations` for the per-level schedule. A shorter list repeats its last entry. `--iterations N` explicitly overrides all levels to N iterations. For example:

```bash
python train.py -s /path/to/scene -m output/short \
  --lod_max 3 --lod_iterations 100 200 300
```

## Saved models and initialization

Each run records its effective loading, optimization and renderer settings in `config.json`, a compatibility `cfg_args`, and `train_command.txt`. Rendering inherits saved settings; only explicitly supplied CLI options override them. The priority is **explicit CLI options > saved configuration > code defaults**, except for `cull_frustum`, which uses an independent render default as described below.

A saved HLoD model directory contains:

```text
point_cloud/lod_5_iteration_30000/
├── point_cloud_lod_1.ply
├── point_cloud_lod_2.ply
├── ...
├── point_cloud_lod_5.ply
└── hierarchy_meta.pt
```

Each PLY holds one birth level. Metadata stores parent indices, birth levels, scale bounds, mip floor and training statistics. Keep the PLY chunks and metadata together. Flat model exports use `point_cloud.ply` together with `hierarchy_meta.pt`. A plain PLY can also initialize a new training run as described below. `--export_flat_legacy` additionally writes a combined PLY; it still requires the metadata for HLoD rendering.


## Rendering and evaluation

```bash
python render.py -m output/garden
```

This renders both train and test views when available. To evaluate only the test views of a model trained with `--eval`, use `python render.py -m output/garden --skip_train` or `./render.sh output/garden`. Train/test behavior is unchanged.

Hierarchical rendering enables additional frustum culling by default (`cull_frustum=1`), independently of the saved training value. Training keeps this culling disabled by default (`0`). With INRIA, `1` adds a one-pixel safety margin around the splat radius; with gsplat it enables the backend's projection visibility test. Use `--cull_frustum 0` to disable it or supply a different positive margin explicitly. This applies to adaptive hierarchy selection, including hierarchical tile counts and videos. The rasterizers also perform their own visibility checks during both training and rendering. `cull_near` continues to inherit its saved training setting.

`hierarchical_training` is the default render mode; the trained maximum LoD, renderer, traversal settings and threshold are read from the model configuration. No explicit `--render_mode`, `--lod` or `--sp_thresh` is needed for the standard run. To change an operating point, supply only its override, for example `python render.py -m output/garden --sp_thresh 9`. Rendering computes PSNR, SSIM, LPIPS, FPS and memory statistics; LPIPS/VGG weights are downloaded on first use and then cached. FPS includes hierarchy selection and rasterization by default; use `--metric_timing render_only` to measure rasterization alone. `--profile` inserts synchronization and writes timing CSVs; report its use when quoting timings.

A cumulative prefix is selected from the final trained model, with only the required chunks loaded for a chunked model:

```bash
python render.py -m output/garden --render_lod_max 3
```

A single birth level is different from an adaptive cumulative prefix:

```bash
python render.py -m output/garden \
  --render_mode single_level --render_lod 3
```

Setting `render_lod_min=render_lod_max` likewise selects that birth level. Other lower-bound ranges are reserved for diagnostic modes. `--lod` selects a source training model; it defaults to the trained maximum level. `--render_lod_max` caps the **final** hierarchy, whereas selecting an earlier source model uses its earlier parameter snapshot.

Other modes:

- `n_tiles`: counts Gaussian–tile intersections using the included rasterizer patch. No separate `diff_gaussian_rasterization_ntiles` package is required.
- `leaf_colors`, `family_colors`, `train_hits` and related `--hierarchical_mode` values: diagnostic visualizations; their image-quality numbers are not reconstruction benchmarks.

For a video without the quality-evaluation pass:

```bash
python render.py -m output/garden --save_video --video_only 
```

For post-hoc hierarchy ablations:

```bash
python build_hierarchy.py -m output/garden --mode kl
python render.py -m output/garden --posthoc_hierarchy hierarchy.pt
```

The builder uses the final birth-level chunks and their effective scale/mip-floor values. The renderer replaces only the parent mapping of that same model. Alternative builder modes are `maha` and `hierarchy_meta` (export trained HLoD links).

## License and acknowledgements

The inherited Gaussian-Splatting license is included in [LICENSE.md](LICENSE.md). Preserve the upstream copyright/license notices. HLoD builds on [FLoD](https://github.com/3DGS-FLoD/flod) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); we thank their authors. Additional source-file notices identify other reused code, including Apache-licensed camera-path utilities.

## Citation

```bibtex
@article{mueller2026hlod,
  author = {Jan U. M{\"u}ller and Jakob Sommer and Reinhard Klein},
  title = {HLoD: Screen-Space Hierarchical Level-of-Detail Traversal for 3D Gaussian Splatting},
  journal = {Computer Graphics Forum},
  volume = {45},
  number = {7},
  year = {2026},
  doi = {10.1111/cgf.70636}
}
```
