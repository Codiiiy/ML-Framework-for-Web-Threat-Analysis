import json
import os
import joblib
import xgboost as xgb
import numpy as np
from pathlib import Path

def convert_xgboost_to_json(model_path: str, output_path: str):
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()
    model_json = booster.get_dump(dump_format='json')
    trees = []
    for tree_str in model_json:
        tree = json.loads(tree_str)
        trees.append(parse_tree_node(tree))
    model_dict = {
        "n_estimators": len(trees),
        "base_score": 0.5,
        "trees": trees,
        "objective": "binary:logistic"
    }
    print(f"Converted {len(trees)} trees")
    with open(output_path, 'w') as f:
        json.dump(model_dict, f, indent=2)
    print(f"Model saved to {output_path}")
    return model_dict

def parse_tree_node(node):
    if 'leaf' in node:
        return {
            'leaf': float(node['leaf'])
        }
    result = {
        'nodeid': node.get('nodeid', 0),
        'split_feature': node.get('split', 'f0').replace('f', ''),
        'split_condition': float(node.get('split_condition', 0)),
    }
    if 'children' in node:
        children = node['children']
        if len(children) >= 2:
            result['left'] = parse_tree_node(children[0])
            result['right'] = parse_tree_node(children[1])
    return result

def convert_scaler_to_json(scaler_path: str, feature_names: list, output_path: str):
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    scaler_dict = {
        "mean": {name: float(mean) for name, mean in zip(feature_names, scaler.mean_)},
        "scale": {name: float(scale) for name, scale in zip(feature_names, scaler.scale_)}
    }
    with open(output_path, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"Scaler saved to {output_path}")
    return scaler_dict

def main():
    base_dir = "./policies"
    model_path = os.path.join(base_dir, "xgb_phishing_detector.json")
    scaler_path = os.path.join(base_dir, "feature_scaler.pkl")
    output_dir = "./extension_model"
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "model.json")
    output_scaler_path = os.path.join(output_dir, "scaler.json")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    try:
        model_dict = convert_xgboost_to_json(model_path, output_model_path)

        print("CONVERSION COMPLETE!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
