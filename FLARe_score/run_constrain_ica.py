import subprocess
from pathlib import Path

from scipy.io import loadmat


def run_constrain_ica(input_path, save_dir, priors, label):
    """Run GIFT in MATLAB and return the time-by-component output."""
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    matlab_dir = Path(__file__).resolve().parent
    arguments = (matlab_dir, Path(input_path).resolve(), save_dir,
                 label, 'MOO-ICAR', Path(priors).resolve())
    quoted = ["'" + str(value).replace("'", "''") + "'" for value in arguments]
    command = f"addpath({quoted[0]}); constrain_ICA({', '.join(quoted[1:])});"
    subprocess.run(['matlab', '-batch', command], check=True)
    return loadmat(save_dir / f'constrained_{label}_timecourse_5.mat')['timecourse']
