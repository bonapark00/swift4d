# render
python render.py --model_path output/dynerf_final/coffee_martini --skip_train --skip_video --iteration 13000 --configs  arguments/dynerf/coffee_martini.py
python render.py --model_path output/dynerf_final/flame_salmon --skip_train  --skip_video --configs  13000 arguments/dynerf/flame_salmon.py
python render.py --model_path output/dynerf_final/cook_spinach --skip_train  --skip_video --configs  13000 arguments/dynerf/cook_spinach.py
python render.py --model_path output/dynerf_final/cut_roasted_beef --skip_train  --skip_video --configs  13000 arguments/dynerf/cut_roasted_beef.py
python render.py --model_path output/dynerf_final/flame_steak --skip_train  --skip_video  --iteration  13000 --configs  arguments/dynerf/flame_steak.py
python render.py --model_path output/dynerf_final/sear_steak --skip_train  --skip_video  --iteration  13000 --configs  arguments/dynerf/sear_steak.py


# metrics
python3 metrics.py --model_path output/dynerf_lite/coffee_martini
python3 metrics.py --model_path output/dynerf_lite/flame_salmon 
python3 metrics.py --model_path output/dynerf_lite/cook_spinach
python3 metrics.py --model_path output/dynerf_lite/cut_roasted_beef
python3 metrics.py --model_path output/dynerf_lite/flame_steak
python3 metrics.py --model_path output/dynerf_lite/sear_steak
