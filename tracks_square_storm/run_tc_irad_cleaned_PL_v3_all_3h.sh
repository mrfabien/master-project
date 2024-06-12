#!/bin/bash -l

#SBATCH --mail-type END
#SBATCH --mail-user fabien.augsburger@unil.ch
#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/case_study
#SBATCH --job-name 1h_rel_hum_all
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/PL_variables/log/con/con-%A_%a.out
#SBATCH --error /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/PL_variables/log/error/err-%A_%a.err
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --time 00:30:00
#SBATCH --array=1-609

# Set your environment
module purge
dcsrsoft use 20240303
module load gcc
source ~/.bashrc
conda activate kera_lgbm

# Specify the path to the config file
config=/work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/PL_variables/config_ML_to_SL_rel.txt
# echo "SLURM_ARRAY_TASK_ID is :${SLURM_ARRAY_TASK_ID}" >> /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/case_study/output_test_all.txt

# Extract the nom_var for the current $SLURM_ARRAY_TASK_ID
nom_var=$(awk -v ArrayTaskID=${SLURM_ARRAY_TASK_ID} '$1==ArrayTaskID {print $2}' $config)

# Extract the annee for the current $SLURM_ARRAY_TASK_ID
annee=$(awk -v ArrayTaskID=${SLURM_ARRAY_TASK_ID} '$1==ArrayTaskID {print $3}' $config)

# Extract the level for the current $SLURM_ARRAY_TASK_ID
level=$(awk -v ArrayTaskID=${SLURM_ARRAY_TASK_ID} '$1==ArrayTaskID {print $4}' $config)

# see if the nom_var and annee are correctly extracted 

# echo "var is :"$nom_var >> /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/case_study/output_test.txt
# echo "annee is :"$annee >> /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/case_study/output_test.txt

# Execute the python script
python3 /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/PL_variables/tc_irad_multi_cleaned_PL_allyears_stats_v4_3h.py "$nom_var" "$annee" "$level"

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same variable, and the year of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, the variable name is ${nom_var} and the year is ${annee}. Level is ${level}" >> /work/FAC/FGSE/IDYST/tbeucler/default/fabien/repos/curnagl/PL_variables/output_all_3h.txt