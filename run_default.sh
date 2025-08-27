#!/bin/bash
#SBATCH --job-name=run_experiments_with_n_iter_progress
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=Output_2/run_%j_%a.out
#SBATCH --error=Error_2/run_%j_%a.err
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=650
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fkoster@ethz.ch
#SBATCH --array=0-0

# Define variables
python_script="run_default.py"
result_folder="Results_Default_Run"
requirements_file="requirements.txt"
#venv_name="Semesterarbeit"

#source "${venv_name}/bin/activate"

task_id=1 #$SLURM_ARRAY_TASK_ID

# Run the Python script with calculated suite, seed, and custom task ID
echo "Running array task ${SLURM_ARRAY_TASK_ID} with progress:"
python "${python_script}" --result_folder "${result_folder}" --task "${task_id}"

# Deactivate virtual environment (not strictly necessary)
deactivate
exit 0