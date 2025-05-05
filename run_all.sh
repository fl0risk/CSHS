#!/bin/bash
#SBATCH --job-name=add_hpo_selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=Output/run_%j_%a.out
#SBATCH --error=Error/run_%j_%a.err
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=1028
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fkoster@ethz.ch
#SBATCH --array=0-79 

# Define variables
python_script="run_experiment_H_SMAC.py"
result_folder_SMAC= "Results_SMAC"
result_folder_H = "Results_H"
requirements_file="requirements.txt"
venv_name="Semesterarbeit"

source "${venv_name}/bin/activate"

# Define the specific suites
suites=(334 335 336 337)

# Define the seed values as arrays
seeds=(27225 32244 34326 92161 99246 108473 117739 147316 235053 257787 89389 443417 572858 620176 671487 710570 773246 777646 778572 936518)

# Calculate suite and seed based on the array index
num_suites=${#suites[@]}
num_seeds=${#seeds[@]}
task_id=$SLURM_ARRAY_TASK_ID

suite_index=$((task_id / num_seeds))
seed_index=$((task_id % num_seeds))

suite_value=${suites[$suite_index]}
seed_value=${seeds[$seed_index]}

# Basic error checking for array bounds (shouldn't happen with correct calculation)
if [ "$suite_index" -ge "$num_suites" ] || [ "$seed_index" -ge "$num_seeds" ]; then
  echo "Error: Array index out of bounds for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# Run the Python script with calculated suite, seed, and custom task ID
echo "Running array task ${SLURM_ARRAY_TASK_ID} with suite: ${suite_value}, seed: ${seed_value}, and custom_task_id: ${custom_task_id}"
python "${python_script}" --suite "${suite_value}" --seed "${seed_value}" --result_folder_H "${result_folder_H}" --result_folder_SMAC "${result_folder_SMAC}"

# Deactivate virtual environment (not strictly necessary)
deactivate
exit 0