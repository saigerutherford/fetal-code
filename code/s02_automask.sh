# Part 2: Run automask code, quality check masks (Pass/Fail)
# createMask.py expects there to be a directory called "images/" where the 96 x 96 sized files live,
# and that they are named zpr_SubjectID_runID.nii
source /home/slab/environments/tensorflow/bin/activate #activate the virtual environment 
python createMask.py