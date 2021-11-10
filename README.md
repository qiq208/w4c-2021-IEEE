# Weather4cast 2021 IEEE BigData
3rd place solution for the Weather4cast 2021 IEEE BigData Challenge

### Dependencies
The code can be executed from a fresh environment using the provided list of requirements: `conda env create -f environment.yml`.

### Inference
A script has been create to made predictions using a trained model on new data as per requirements detailed in competition: (#https://github.com/iarai/weather4cast#code-and-abstract-submission)

The model weights of the final submission for both core and transfer learning can be downloaded from https://drive.google.com/drive/folders/1PXHNNRIcIzb1TywuKf-YU2AEYo_9w2MY?usp=sharing

To run predictions on a test dataset use ('inference.py'). This should fine on a CPU machine

examples of usage:

    inference for Region R5 using A42.pth and A44.pth

    R=R5
    INPUT_PATH=data
    WEIGHT_FOLDER=weights
    OUT_PATH=.
    python weather4cast/inference.py -d $INPUT_PATH -r $R -f $WEIGHT_FOLDER -w -w A42.pth A44.pth -o $OUT_PATH -g 'cuda'



### Train/evaluate a UNet
To replicate the training for the 3rd place solutions use of the training notebooks (`Training_ModelA.ipynb` and `Training_ModelB.ipynb`).

