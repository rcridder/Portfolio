Using a Neural Network to classify EEG Brain Signals

Our script is calling data from a matlab file and a csv file that can be found at the follwing links
You will need a Kaggle account to download them, If you do not have one please email us at
jbalisa1@swarthmore.edu or rridder1@swarthmore.edu

matlab data - https://www.kaggle.com/inancigdem/eeg-data-for-mental-attention-state-detection

csv data - https://www.kaggle.com/birdy654/eeg-brainwave-dataset-mental-state

In order to run the code you must change the location of the 2 files to the directory in which they are
located on your machine on lines 15 and 16.

You will also need the following libraries:
Keras, from Tensorflow
numpy
scipy
pandas
matplotlib
sys
os

run the code using python 3+. The command structure is:

python3 classifyMentalState.py <file input (csv or mat)> <classification domain (time, freq, or our_freq)>
file input = csv will run the csv data, mat will run the matlab data
classification domain = time will classify the data in the time domain, freq will use the fft data columns from the csv or compute the fft of the matlab time domain data, our_freq will calculate our own fft of the csv data (this does not work as expected, but was being used experimentally)

for example, the following four commands compose the majority of our assessment data:
python3 classifyMentalState.py csv time
python3 classifyMentalState.py csv freq
python3 classifyMentalState.py mat time
python3 classifyMentalState.py mat freq

JJ Balisanyuka-Smith and Rose Ridder
