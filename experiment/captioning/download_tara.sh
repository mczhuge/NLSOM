cd data
mkdir tara
cd tara

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aHu9nbdmnQ8SDr-0P7VySLsKxl54SUQp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aHu9nbdmnQ8SDr-0P7VySLsKxl54SUQp" -O image.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vOyyk8MNF9m47ZoFPxhdDCdbW5VqQpLh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vOyyk8MNF9m47ZoFPxhdDCdbW5VqQpLh" -O tara.txt && rm -rf /tmp/cookies.txt

unzip image.zip