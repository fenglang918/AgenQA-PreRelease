import asyncio
import sys
import tempfile
import unittest

from infra.code_verifier.executor import ExecutionConfig
from infra.code_verifier.python_executor import PythonExecutor


class TestPythonExecutorTempDirFallback(unittest.TestCase):
    def test_falls_back_when_temp_dir_is_a_file(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            cfg = ExecutionConfig(timeout=2.0, temp_dir=f.name, python_bin=sys.executable)
            executor = PythonExecutor(cfg)
            res = asyncio.run(executor.execute_single("print('ok')"))

        self.assertTrue(res.success)
        self.assertIn("ok", res.output)
        self.assertNotEqual(executor.config.temp_dir, f.name)

    def test_falls_back_when_python_bin_missing(self) -> None:
        cfg = ExecutionConfig(timeout=2.0, temp_dir=tempfile.gettempdir(), python_bin="__missing_python_bin__")
        executor = PythonExecutor(cfg)
        res = asyncio.run(executor.execute_single("print('ok')"))

        self.assertTrue(res.success)
        self.assertIn("ok", res.output)
        self.assertNotEqual(executor.config.python_bin, "__missing_python_bin__")
