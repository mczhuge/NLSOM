cd data
mkdir coco
cd coco

for split in val; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip"
done