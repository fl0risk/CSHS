#!/bin/bash
#SBATCH --job-name=run_experiments_with_n_iter_progress
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=Output_2/run_%j_%a.out
#SBATCH --error=Error_2/run_%j_%a.err
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fkoster@ethz.ch
#SBATCH --array=0-1

# Define variables
python_script_list=("model_gp_boosting.py" "model_lin_regr.py")
requirements_file="requirements.txt"
venv_name="Semesterarbeit"

#source "${venv_name}/bin/activate"
task_id=$SLURM_ARRAY_TASK_ID
python_script=${python_script_list[task_id]}
# Run the Python script with calculated suite, seed, and custom task ID
python "${python_script}" 

# Deactivate virtual environment (not strictly necessary)
#deactivate
exit 0