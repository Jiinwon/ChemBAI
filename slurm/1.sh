set -e
trap 'echo "[ERR] line:$LINENO cmd:$BASH_COMMAND (exit:$?)" >&2' ERR
# --- script dir ---
PYTHONPATH="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model" \
python - <<'PY'
import sys, config
print("config.__file__ =", getattr(config, "__file__", None))
print("HAS_VERSION     =", hasattr(config, "VERSION"))
print("HAS_BASE_DIR    =", hasattr(config, "BASE_DIR"))
if hasattr(config,"VERSION"):  print("VERSION =", config.VERSION)
if hasattr(config,"BASE_DIR"): print("BASE_DIR=", config.BASE_DIR)
PY
echo "rc=$?"
