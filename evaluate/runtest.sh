nohup python3 eval_age_methods_mosaic.py --dataset lap --partition test --gpu 0 --path ../fine_tuned_chalearnlap_from_vggface2/ --out_path results_vggface2_to_lap > runtest.v.log &
nohup python3 eval_age_methods_mosaic.py --dataset lap --partition test --gpu 1 --path ../fine_tuned_chalearnlap_from_imdbwiki/ --out_path results_imdbwiki_to_lap > runtest.i.log &

