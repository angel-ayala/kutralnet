#!/bin/bash

# training dataset
printf  "Starting download of dataset...\n"
../../utils/gdown.pl "https://drive.google.com/open?id=165fUt_SiS50syL8QtdON50D_uyudcsrW" "./training.zip"

printf "Unzipping\n"
unzip -qq "training.zip"
mv "Training Dataset" "Training"
rm "training.zip"

printf "Some images needs to be transformed!\n"
printf "Training/Fire/fire1.jpg "
mogrify -format jpg "Training/Fire/fire1.jpg"
printf "OK\n"
printf "Training/NoFire/nofire630.jpg "
mogrify -format jpg "Training/NoFire/nofire630.jpg"
printf "OK\n"
printf "Training/NoFire/nofire631.jpg "
mogrify -format jpg "Training/NoFire/nofire631.jpg"
printf "OK\n"
printf "Training/NoFire/nofire650.jpg "
mogrify -format jpg "Training/NoFire/nofire650.jpg"
printf "OK\n"
printf "Training/NoFire/nf1238.jpg "
mogrify -format jpg "Training/NoFire/nf1238.jpg"
printf "OK\n"

printf "Done!\n"

# test dataset
printf  "Starting download of test dataset...\n"
../../utils/gdown.pl "https://drive.google.com/open?id=18nI3pLuB_JdnnYgt-u5284j7FGkNfOAa" "./testing.zip"

printf "Unzipping\n"
unzip -qq "testing.zip"
mv "Test_Dataset1__Our_Own_Dataset" "Test"
rm "testing.zip"

printf "Done!\n"
