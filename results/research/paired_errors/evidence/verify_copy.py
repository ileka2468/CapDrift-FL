import inspect
from pathlib import Path
import sys
from types import ModuleType
import unittest

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
context = ModuleType("run_competitor_benchmark")
context.__file__ = str(root / "scripts" / "run_competitor_benchmark.py")
copied = Path(sys.argv[1])
sys.modules[context.__name__] = context
exec(compile(copied.read_bytes(), context.__file__, "exec"), context.__dict__)
args = inspect.signature(context.run_competitor_trial).parameters
print("final_model_observer=", "final_model_observer" in args,
      "feature_lr_multiplier=", "feature_lr_multiplier" in args, sep="", flush=True)
runner = unittest.TextTestRunner()
suite = unittest.defaultTestLoader.discover("tests", pattern="test_capdrift*.py")
raise SystemExit(not runner.run(suite).wasSuccessful())
