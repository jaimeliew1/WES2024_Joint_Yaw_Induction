poetry run ipython WES2024/Analysis_BEM_windfarm_optima.py  &> log1.txt
poetry run ipython WES2024/Fig11_diamond_pitch_tsr_surface.py  &> log2.txt
aws s3 cp --recursive data s3://jamalama
aws s3 cp --recursive log1.txt s3://jamalama
aws s3 cp --recursive log2.txt s3://jamalama
sudo shutdown now -h
