# Thin wrapper used by the Docker CMD to serve env.py templates AND the
# custom sae_heist template defined in tasks/sae_heist/task.py.
#
# Why this file exists:
#   hud serve env:env loads only env.py, which registers diagnose_research_study
#   but knows nothing about sae_heist. The @env.template(id="sae_heist")
#   decorator must execute at serve time. Importing the task module here runs
#   the decorator on the same env object, making sae_heist available to the
#   Docker control channel.
#
# Dockerfile CMD:
#   CMD ["hud", "serve", "env_with_tasks:env", "--host", "0.0.0.0"]
from env import *            # re-exports env (Environment) + all public helpers
from env import env          # ensure 'env' attribute is in this module's vars
import tasks.sae_heist.task  # noqa: F401 — registers @env.template(id="sae_heist")
