poetry run ipython WES2024/Generate/diamond_BEM_uncertainty.py  &> log1.txt
poetry run ipython WES2024/Generate/diamond_BEM_uncertainty_postproc.py  &> log2.txt

aws s3 cp --recursive data s3://jamalamaa

aws s3 cp --recursive log1.txt s3://jamalamaa
aws s3 cp --recursive log2.txt s3://jamalamaa

sudo shutdown now -h
