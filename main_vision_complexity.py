import sys
import json
from src.common.vision_complexity import fitness_vison_complexity

if __name__ == "__main__":
    
    

    file_path = r"data\01-07-2024-all_tests_ga.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    for run_key, test_cases in data.items():
        print("***************Run Key: ", run_key)
        for test_id, test_info in test_cases.items():
            if test_info.get("test_outcome") == "FAIL":
                
                fitness_vison_complexity(test_info) 
    