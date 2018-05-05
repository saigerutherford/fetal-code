# Global Values
# This isn't great style, but we are being a bit hacky here.
width = 96
height = 96
n_channels = 1
n_classes = 2
cappedIterations = 200001
batchStepsBetweenSummaries = 500

stepsBeforeStoppingCriteria = 40000

imageBatchDims = (-1, width, height, n_channels)
labelBatchDims = (-1, width, height)