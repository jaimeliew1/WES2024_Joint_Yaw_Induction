poetry run ipython WES2024/Generate/diamond_BEM_uncertainty.py  &> log1.txt
# poetry run ipython WES2024/Plot/Fig11_diamond_pitch_tsr_surface.py  &> log2.txt
aws s3 cp --recursive data s3://jamalamaa
aws s3 cp --recursive log1.txt s3://jamalamaa
# aws s3 cp --recursive log2.txt s3://jamalamaa
sudo shutdown now -h
