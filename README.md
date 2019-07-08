# Automated Brain Masking of Fetal Functional MRI Data
**Preprint:** https://www.biorxiv.org/content/early/2019/01/21/525386

**Abstract:** Fetal resting-state functional magnetic resonance imaging (rs-fMRI) has emerged as a critical new approach for characterizing brain development before birth. Despite rapid and widespread growth of this approach, at present we lack neuroimaging processing pipelines suited to address the unique challenges inherent in this data type. Here, we solve the most challenging processing step, rapid and accurate isolation of the fetal brain from surrounding tissue across thousands of non-stationary 3D brain volumes. Leveraging our library of 1,241 manually traced fetal fMRI images from 207 fetuses, we trained a Convolutional Neural Network (CNN) that achieved excellent performance across two held-out test sets from separate scanners and populations. Furthermore, we unite the auto-masking model with additional fMRI preprocessing steps from existing software and provide insight into our adaptation of each step. This work represents an initial advancement towards a fully comprehensive, open source workflow for fetal functional MRI data preprocessing. 

![](figures/FetalExample_Axial.gif) 
![](figures/FetalExample_Sagittal.gif)
![](figures/FetalExample_Coronal.gif)

Primary data used for pipeline development were acquired at Wayne State
University School of Medicine during the course of projects supported by National
Institutes of Health (NIH) awards MH110793 and ES026022.

For access to raw fetal functional time-series data used in the development of this code please contact Moriah Thomason Moriah.Thomason@nyulangone.org

### Repository organization
**checkpoints -->** contains the saved models. **2018-06-07_14:07** is the model trained using _**train, validation, and test split**_ (129, 20, 48 subjects; 855, 102, 211 volumes) **2018-06-08_10:47** is the model trained on _**all**_ labeled WSU data and tested on Yale data.

**code -->** this directiory contains all necessary scripts for running the pretrained model (`createMasks.py`), or training your own model (`buildModel.py` and `trainModel.py`).  code/FullFetalPreprocessPipeline.sh --> Example pipeline using auto-mask and FSL. s02_automask.sh is an example of how to activate virtual environment and run get masks using the pre-trained model. **Work in progress, not fully tested or setup without hard-coded paths**.


**figures -->** Jupyter notebook used to make the figures in the manuscript. 

**summaries -->** Contains the summaries for both models described above (in the checkpoints directory description) that can be viewed using tensorboard.
`tensorboard --logdir=summaries/model_name`

### Installation & Requirements
Required libraries: 
For running on Mac CPU --> CPU_Mac_Requirements.txt (note: some of these libraries are probably unnecessary if you do not use Jupyter)
For running on Linux using GPU --> GPU_Linux_Requirements.txt (note: tensorflow_gpu==1.11 requires CUDA 9.0 see https://www.tensorflow.org/install/gpu for more details)

### Necessary data prep steps (prior to auto-masking)
Input data must be of dimensions 96 x 96 x N. The data used in training was resampled to voxel size 3.5 mm^3, then zero padded to 96 x 96 x 37. See lines s01_prep.sh for an example of preprocessing commands.

## Running the auto-mask code options
### Use pre-trained model
Images should be in 3D volume format (split the 4D time series into individual volumes). File naming should be consistent. Currently, the code expects images to be in a folder called "images/" and named as "zpr_SubjectID_runID_vol0000.nii".
createMask.py is the script to run to create new masks using the model trained with all hand drawn brain masks. In order to use the correct pre-trained model, make sure that line 152 of trainModel.py is set as follows: `main(train=False, timeString='2018-06-08_10:47')`
You will also need to edit lines 57-62 of createMasks.py in order to match the path to the directory of where your data lives. You can also specify (in createMasks.py lines 64-67) a single file to be masked if you are not masking multiple volumes/subjects.

### Train model on your data
Instructions for training model on new data coming soon.

## Other preprocessing steps
1. Merge auto-masks into 4D file to view as overlay on data in order to quality check the masks.
2. Cluster and binarize the probability masks
3. Quality check binarized mask
4. Resample mask back into subject's native space
5. Apply binarized native space brain mask to raw data to extract the fetal brain and discard other tissues
6. Quality check
7. Realign time series and identify usable low motion volumes
8. Quality Check
9. Normalize to age-matched fetal template
10. Quality check
