#!/bin/bash

_conda_root_candidates() {
    if [ -n "${CONDA_ROOT:-}" ]; then
        printf '%s\n' "$CONDA_ROOT"
    fi
    printf '%s\n' \
        "${CONDA_ROOT}" \
        "${CONDA_ROOT_ALT}/mambaforge" \
        "${CONDA_ROOT}" \
        "${CONDA_ROOT_ALT}/miniconda3" \
        "${CONDA_ROOT}" \
        "${CONDA_ROOT_ALT}/anaconda3" \
        "$HOME/mambaforge" \
        "$HOME/miniconda3" \
        "$HOME/anaconda3"
}

_activate_conda_shell() {
    local root
    while IFS= read -r root; do
        if [ -f "$root/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$root/etc/profile.d/conda.sh"
            return 0
        fi
    done < <(_conda_root_candidates)

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi

    echo "ERROR: Could not find conda via CONDA_ROOT, common install paths, or PATH." >&2
    return 1
}

init_slurm_conda_shell() {
    _activate_conda_shell
}

activate_slurm_conda_env() {
    local env_name="${1:-mtrl_env}"
    init_slurm_conda_shell || return 1
    conda activate "$env_name"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    activate_slurm_conda_env "$@"
fi
