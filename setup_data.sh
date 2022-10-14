

# This code will download (and then delete) the e-SNLI and HANS repos into the parent folder of snli_logic

# Open in snli_logic folder
mkdir dataset_esnli
cd ..
git clone https://github.com/OanaMariaCamburu/e-SNLI.git
mv e-SNLI/dataset/esnli_dev.csv snli_logic/dataset_esnli/
mv e-SNLI/dataset/esnli_test.csv snli_logic/dataset_esnli/
mv e-SNLI/dataset/esnli_train_1.csv snli_logic/dataset_esnli/
mv e-SNLI/dataset/esnli_train_2.csv snli_logic/dataset_esnli/
rm -rf e-SNLI

## SICK (corrected and uncorrected)
git clone https://github.com/huhailinguist/SICK_correction.git
mv SICK_correction/SICK_corrected.tsv snli_logic/data/SICK/
rm -rf SICK_correction
cd snli_logic/
# Formatting SICK
cd data/SICK/corrected_SICK/
python convert_sick.py
cd ..
cd uncorrected_SICK/
python convert_sick.py
cd ../../..

## HANS

cd ..
git clone https://github.com/tommccoy1/hans.git
mv hans/heuristics_evaluation_set.txt snli_logic/data/
rm -rf hans
cd snli_logic

## SNLI-hard

cd data/

# Download SNLI
wget https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl

mv snli_1.0_test_hard.jsonl snli_hard.jsonl

