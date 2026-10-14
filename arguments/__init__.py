from argparse import ArgumentParser, Namespace, BooleanOptionalAction, SUPPRESS
from utils.config_utils import load_config
import sys
import os

DEFAULT_SP_THRESH = 3.0

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = int if key == "iterations" and value is None else type(value)
            parser._hlod_defaults = getattr(parser, "_hlod_defaults", {})
            parser._hlod_defaults[key] = value
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                elif t == list:
                    t = type(value[0])
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                elif t == list:
                    try:
                        t = type(value[0])
                    except (IndexError, TypeError):
                        pass
                    group.add_argument("--" + key, default=value, type=t, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._alpha_masks = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        self.eval = False
        self.llffhold = 8
        self.eval_proximity = False

        self.lod1_scaling_lower_bound = 0.05
        self.lod_scaling_ratio = 2.0
        self.increase_lod_num_childs = 2
        self.mip_floor = True
        self.mip_floor_mode = "fixed"
        self.mip_floor_clamp_raw = True

        self.use_voxel_sampling = False
        self.voxel_sampling_size = 0.2

        self.init_opacity = 0.1

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.low_pass = 0.0
        self.renderer = "inria"  # "inria" (diff-gaussian-rasterization) or "gsplat"
        self.absgrad = True       # gsplat only (ignored by INRIA): AbsGS absolute-gradient densification.
                                  # Default ON for gsplat; wants a ~4x higher densify_grad_threshold
                                  # (~0.0008) than INRIA's 2e-4 — train.py auto-bumps it from the default.
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Optional override of the per-LoD schedule; --lod_iterations is primary.
        self.iterations = None
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = [0.2]

        self.densify_grad_threshold = [0.0002]
        self.densification_interval = [2000, 1000, 500, 500, 200]
        self.densify_from_iter = [500]
        self.densify_until_iter = [5000, 6000, 8000, 10000, 15000]

        self.prune_opacity_threshold = [0.005]
        self.pruning_interval = [100]
        self.prune_from_iter = [500]
        self.prune_until_iter = [5000, 6000, 8000, 10000, 15000]

        self.prune_overlap_threshold = [0.1, 0.025, 0.00625, 0.0015625, 0.0]
        self.pruning_overlap_interval = [1000, 1000, 1000, 1000, 15001]

        self.prune_max_radii2D_lod = 999
        self.prune_max_radii2D = [0]

        self.opacity_reset_interval = [3000]

        self.random_background = False

        self.lambda_scale_upper = 0.01
        self.scale_upper_scope = "current_only"
        self.scale_upper_include_lod1 = False

        self.scale_upper_iter_weight_power = 0.0

        self.lambda_maha = 0.005
        self.maha_margin = 1.0

        self.maha_scope = "oldest"

        self.maha_reduce = "mean"

        self.regularizer_scope = "active"

        self.hierarchy_aware_prune = 'protect'

        self.prune_overlap_same_lod_only = True

        self.prune_current_lod_only = True

        self.allow_previous_lod_prune = False

        self.min_hits_for_opacity_prune = 0

        self.end_of_lod_prune = True

        self.end_of_lod_prune_mode = "family_cleanup"

        self.end_of_lod_prune_dead_families = False

        self.end_of_lod_prune_scope = "frozen"

        self.densify_current_lod_only = True

        self.save_hierarchy_chunks = True
        self.export_flat_legacy = False

        self.freeze_previous_lod_regularizers = True
        self.freeze_previous_lod_mutations = True
        self.sp_thresh = DEFAULT_SP_THRESH
        self.hierarchical_mode = "leafes_to_coarse"

        # Frustum culling during the per-view active-set selection (0 = off).
        # cull_near drops Gaussians behind the near plane inside the traversal;
        # cull_frustum enables renderer-aware off-screen culling. Its numeric
        # value is an extra pixel margin only for the INRIA backend.
        self.cull_near = 0.0
        self.cull_frustum = 0.0

        self.prefix_prob = 0.05
        self.current_prob = 0.03
        self.prefix_min_lod = 1
        self.train_profile = False
        self.train_profile_interval = 100
        self.train_profile_warmup = 0

        self.per_parent_budget = 0


        self.per_parent_budget_debug = False



        self.lod_iterations = [10000, 15000, 20000, 25000, 30000]
        self.lod_min = 1
        self.lod_max = 5

        super().__init__(parser, "Optimization Parameters")

def explicit_args(parser, argv):
    # Parse again without defaults: only explicitly supplied options override
    # saved values. This also distinguishes --no-flag from an omitted flag.
    action_defaults = [(action, action.default) for action in parser._actions]
    parser_defaults = parser._defaults
    try:
        parser._defaults = {}
        for action, _ in action_defaults:
            action.default = SUPPRESS
        explicit = vars(parser.parse_args(argv))
    finally:
        parser._defaults = parser_defaults
        for action, default in action_defaults:
            action.default = default
    return explicit


def get_combined_args(parser : ArgumentParser, exclude_saved=()):
    argv = sys.argv[1:]
    defaults = parser.parse_args(argv)
    if not defaults.model_path:
        parser.error("--model_path/-m is required")
    try:
        config = vars(load_config(defaults.model_path))
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))
    explicit = explicit_args(parser, argv)
    merged = dict(getattr(parser, "_hlod_defaults", {}))
    merged.update({key: value for key, value in vars(defaults).items() if value is not None})
    # Some evaluation settings deliberately differ from training defaults.
    # Keep the parser default for these keys, while still accepting CLI overrides.
    merged.update({key: value for key, value in config.items() if key not in exclude_saved})
    merged.update(explicit)
    if "lod" not in explicit:
        merged["lod"] = config.get("lod_max", merged.get("lod", 5))
    return Namespace(**merged)
