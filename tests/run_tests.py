import importlib.util
import sys
import traceback

spec = importlib.util.spec_from_file_location('tests.test_gp_symbolic', 'tests/test_gp_symbolic.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
failed = 0
for name in dir(mod):
    if name.startswith('test_'):
        func = getattr(mod, name)
        if callable(func):
            try:
                func()
                print(f"{name}: OK")
            except Exception:
                failed += 1
                print(f"{name}: FAILED")
                traceback.print_exc()

if failed:
    print(f"{failed} tests failed")
    sys.exit(1)
else:
    print("All tests passed")
    sys.exit(0)
