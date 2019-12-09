#!/bin/bash

# training dataset
printf  "Starting download of dataset...\n"
../../utils/gdown.pl "https://drive.google.com/file/d/1Cq9KGYzmQ2IlFnkWyji-03DSJWZY36jS/view" "./FiSmo-Images.zip"

printf "Unzipping\n"
unzip -qq "FiSmo-Images.zip"
mv FiSmo-Images/* ./
rm "FiSmo-Images.zip"

printf "Done!"
