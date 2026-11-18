"""Install a fresh wheel plus locked core dependencies in an isolated environment."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, help="exact freshly built wheel (default: sole dist wheel)"
    )
    parser.add_argument("--dist", type=Path, default=ROOT / "dist", help="build output directory")
    args = parser.parse_args()
    wheel_files = (
        [args.wheel.resolve()]
        if args.wheel
        else sorted(args.dist.resolve().glob("omni_eval-*.whl"))
    )
    if len(wheel_files) != 1:
        raise RuntimeError("build exactly one wheel before smoke checking")
    if not wheel_files[0].is_file() or wheel_files[0].suffix != ".whl":
        raise ValueError("--wheel must identify an existing wheel")

    with tempfile.TemporaryDirectory(prefix="omnieval-wheel-") as location:
        temp = Path(location)
        target = temp / "venv"
        requirements = temp / "requirements.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(requirements),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(target)], cwd=temp, check=True
        )
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "--requirements",
                str(requirements),
                str(wheel_files[0]),
            ],
            cwd=temp,
            check=True,
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        code = (
            "import omni_eval; "
            "from pathlib import Path; "
            f"assert Path(omni_eval.__file__).is_relative_to({str(target)!r}); "
            "from omni_eval import evaluate; "
            "from omni_eval.plugins.catalog import Catalog; "
            "assert len(Catalog().list()) == 22; "
            "from omni_eval.types.case import Case; "
            "run=evaluate([Case(id='a',expected='A',actual='A')], "
            "task='classification',metrics=['accuracy']); "
            "assert run.metric('accuracy').value==1; "
            "print('wheel import and actual scoring OK')"
        )
        subprocess.run([str(python), "-I", "-c", code], cwd=temp, env=env, check=True)
        subprocess.run(
            [str(python), "-I", "-m", "omni_eval", "demo", "retrieval", "--dry-run"],
            cwd=temp,
            env=env,
            check=True,
        )
        # Only copy example data and the verification script; imports must use the wheel.
        shutil.copytree(ROOT / "examples", temp / "examples")
        (temp / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/smoke.py", temp / "scripts/smoke.py")
        subprocess.run(
            [str(python), "-I", str(temp / "scripts/smoke.py")], cwd=temp, env=env, check=True
        )
        command = target / ("Scripts/omni-eval.exe" if os.name == "nt" else "bin/omni-eval")
        subprocess.run(
            [str(command), "init", "project", "--task", "classification"],
            cwd=temp,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(command), "run", "-c", "project/eval.yaml", "--out", "runs"],
            cwd=temp,
            env=env,
            check=True,
        )
    print("Wheel installed outside source tree and CLI smoke passed")


if __name__ == "__main__":
    main()
