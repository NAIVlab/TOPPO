from wrappers.vec_env import MultiParallelEnvExecutor
from wrappers.vec_env import MultiParallelEvalEnvExecutor

# EpisodeMonitor / SinglePrecision are old-gym (0.18.3) wrappers used only by
# the vendored-metaworld CPU path; the metaworld3 env stack
# (gymnasium-only, no old gym installed) never touches them.
try:
    from wrappers.episode_monitor import EpisodeMonitor
    from wrappers.single_precision import SinglePrecision
except ModuleNotFoundError:  # old gym not installed (metaworld3 envs)
    EpisodeMonitor = None
    SinglePrecision = None
