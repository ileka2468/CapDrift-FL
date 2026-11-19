import importlib.util
import sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "scripts"))
spec = importlib.util.spec_from_file_location("scope_probe_module", p)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
base = mod.frozen_baseline()[0]
cfg = mod.candidates(base)
print("anchor=" + cfg[mod.BASELINE].drift_parameter_prefix + "; whole-classifier=" +
      (cfg["whole-classifier"].drift_parameter_prefix if "whole-classifier" in cfg else "absent"))
