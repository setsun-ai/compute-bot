# compute-bot: the job environment in your own shell (bash/zsh): compute-env, then gmx, python, vina, obabel...
export COMPUTE_HOME=/var/lib/compute-bot
compute-env() {
  export MAMBA_ROOT_PREFIX=/var/lib/compute-bot/mamba
  eval "$(/opt/compute-bot/bin/micromamba shell hook -s "$(basename "${SHELL:-bash}")")"
  micromamba activate /var/lib/compute-bot/env
  export PATH="/opt/compute-bot/app/compute_bot/bin:/var/lib/compute-bot/tools/bin:$PATH"
}
