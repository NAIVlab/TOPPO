# Optional: source before running python by hand, e.g. the one-seed command in
# README.md (the sbatch templates in slurm/ do not need it).
# Headless MuJoCo (new bindings): EGL, never a display.
export MUJOCO_GL=egl
unset DISPLAY
# JAX: don't grab the whole GPU (train_ars.py also sets this; belt+braces for
# any direct python invocations).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# One BLAS thread per process, as in both sbatch templates: the 50 lockstep env
# workers would otherwise oversubscribe the CPUs.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
