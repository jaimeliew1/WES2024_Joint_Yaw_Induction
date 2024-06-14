poetry run ipython test_LUT_Ctprime.py &> log_test_LUT_Ctprime.txt
poetry run ipython test_LUT_Ct.py &> log_test_LUT_Ct.txt

aws s3 cp log_test_LUT_Ctprime.txt s3://jamalamaa
aws s3 cp log_test_LUT_Ct.txt s3://jamalamaa

poetry run ipython WES2024/Generate/_single_turbine_opt.py &> log__single_turbine_opt.txt
poetry run ipython WES2024/Generate/pitch_tsr_surface.py &> log_pitch_tsr_surface.txt
poetry run ipython WES2024/Generate/two_turbine_AD.py &> log_two_turbine_AD.txt
poetry run ipython WES2024/Generate/two_turbine_AD_sensitivity.py &> log_two_turbine_AD_sensitivity.txt
poetry run ipython WES2024/Generate/two_turbine_BEM.py &> log_two_turbine_BEM.txt
poetry run ipython WES2024/Generate/two_turbine_BEM_sensitivity.py &> log_two_turbine_BEM_sensitivity.txt
poetry run ipython WES2024/Generate/minCt_trajectory.py &> log_minCt_trajectory.txt
poetry run ipython WES2024/Generate/diamond_AD.py &> log_diamond_AD.txt
poetry run ipython WES2024/Generate/diamond_BEM.py &> log_diamond_BEM.txt
poetry run ipython WES2024/Generate/diamond_BEM_uncertainty.py &> log_diamond_BEM_uncertainty.txt
poetry run ipython WES2024/Generate/diamond_BEM_uncertainty_postproc.py &> log_diamond_BEM_uncertainty_postproc.txt
poetry run ipython WES2024/Generate/diamond_AD_sensitivity.py &> log_diamond_AD_sensitivity.txt

aws s3 cp --recursive data s3://jamalamaa


aws s3 cp log__single_turbine_opt.txt s3://jamalamaa
aws s3 cp log_pitch_tsr_surface.txt s3://jamalamaa
aws s3 cp log_two_turbine_AD.txt s3://jamalamaa
aws s3 cp log_two_turbine_AD_sensitivity.txt s3://jamalamaa
aws s3 cp log_two_turbine_BEM.txt s3://jamalamaa
aws s3 cp log_two_turbine_BEM_sensitivity.txt s3://jamalamaa
aws s3 cp log_minCt_trajectory.txt s3://jamalamaa
aws s3 cp log_diamond_AD.txt s3://jamalamaa
aws s3 cp log_diamond_BEM.txt s3://jamalamaa
aws s3 cp log_diamond_BEM_uncertainty.txt s3://jamalamaa
aws s3 cp log_diamond_BEM_uncertainty_postproc.txt s3://jamalamaa
aws s3 cp log_diamond_AD_sensitivity.txt s3://jamalamaa

sudo shutdown now -h




