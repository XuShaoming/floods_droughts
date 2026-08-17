# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "Historical" &
# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "RCP4.5" &
# python process_eddev_data.py --all --basin "WildRiceMarsh_Watersheds" --scenario "RCP8.5" &
# wait
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "hist_scaled" &
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "RCP4.5" &
# python combine_eddev_flow.py --basin "WildRiceMarsh" --scenario "RCP8.5" &