#!/bin/bash

# training dataset
printf  "Starting download of dataset..."
../../utils/gdown.pl "https://drive.google.com/open?id=165fUt_SiS50syL8QtdON50D_uyudcsrW" "./training.zip"

printf "Unzipping"
unzip -qq "training.zip"
mv "Training Dataset" "Training"
rm "training.zip"

printf "Done!"

# test dataset
printf  "Starting download of test dataset..."
../../utils/gdown.pl "https://drive.google.com/open?id=18nI3pLuB_JdnnYgt-u5284j7FGkNfOAa" "./testing.zip"

printf "Unzipping"
unzip -qq "testing.zip"
mv "Test_Dataset1__Our_Own_Dataset" "Test"
rm "testing.zip"

printf "Done!"
