# render
python render.py --model_path output/dynerf_final/coffee_martini_down2_4dgs --skip_train --skip_video --iteration 13000 --configs  arguments/dynerf/coffee_martini.py
python render.py --model_path output/dynerf_final/flame_salmon_down2_4dgs --skip_train  --skip_video --configs  13000 arguments/dynerf/flame_salmon.py
python render.py --model_path output/dynerf_final/cook_spinach_down2_4dgs --skip_train  --skip_video --configs  13000 arguments/dynerf/cook_spinach.py
python render.py --model_path output/dynerf_final/cut_roasted_beef_down2_4dgs --skip_train  --skip_video --configs  13000 arguments/dynerf/cut_roasted_beef.py
python render.py --model_path output/dynerf_final/flame_steak_down2_4dgs --skip_train  --skip_video  --iteration  13000 --configs  arguments/dynerf/flame_steak.py
python render.py --model_path output/dynerf_final/sear_steak_down2_4dgs --skip_train  --skip_video  --iteration  13000 --configs  arguments/dynerf/sear_steak.py


# metrics
python3 metrics.py --model_path output/dynerf_lite/coffee_martini_down2_4dgs 
python3 metrics.py --model_path output/dynerf_lite/flame_salmon_down2_4dgs 
python3 metrics.py --model_path output/dynerf_lite/cook_spinach_down2_4dgs 
python3 metrics.py --model_path output/dynerf_lite/cut_roasted_beef_down2_4dgs 
python3 metrics.py --model_path output/dynerf_lite/flame_steak_down2_4dgs
python3 metrics.py --model_path output/dynerf_lite/sear_steak_down2_4dgs 
