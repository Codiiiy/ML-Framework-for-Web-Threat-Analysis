class XGBoostPredictor {
  constructor() {
    this.model = null;
    this.featureNames = null;
    this.scaler = null;
  }

  async loadModel(modelData, scalerData = null) {
    this.model = modelData;
    this.scaler = scalerData;
    
    if (this.model.trees && this.model.trees.length > 0) {
      const firstTree = this.model.trees[0];
      this.featureNames = this.extractFeatureNames(firstTree);
    }
  }

  extractFeatureNames(tree) {
    const names = new Set();
    const traverse = (node) => {
      if (node.split_feature !== undefined) {
        names.add(node.split_feature);
      }
      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right);
    };
    traverse(tree);
    return Array.from(names);
  }

  scaleFeatures(features) {
    if (!this.scaler) return features;
    
    const scaled = {};
    for (const [key, value] of Object.entries(features)) {
      if (this.scaler.mean && this.scaler.scale) {
        const mean = this.scaler.mean[key] || 0;
        const scale = this.scaler.scale[key] || 1;
        scaled[key] = (value - mean) / scale;
      } else {
        scaled[key] = value;
      }
    }
    return scaled;
  }

  predictTree(tree, features) {
    let node = tree;
    
    while (node.leaf === undefined) {
      const featureValue = features[node.split_feature] || 0;
      
      if (featureValue < node.split_condition) {
        node = node.left;
      } else {
        node = node.right;
      }
      
      if (!node) break;
    }
    
    return node ? (node.leaf || 0) : 0;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(features) {
    if (!this.model) {
      throw new Error('Model not loaded');
    }

    const scaledFeatures = this.scaleFeatures(features);
    
    let rawScore = this.model.base_score || 0;
    
    for (const tree of this.model.trees) {
      rawScore += this.predictTree(tree, scaledFeatures);
    }
    
    const probability = this.sigmoid(rawScore);
    
    return {
      probability: probability,
      prediction: probability > 0.5 ? 1 : 0,
      rawScore: rawScore,
      isPhishing: probability > 0.5
    };
  }
}

class ModelLoader {
  static async loadFromStorage() {
    try {
      const result = await browser.storage.local.get(['xgboost_model', 'feature_scaler']);
      return {
        model: result.xgboost_model,
        scaler: result.feature_scaler
      };
    } catch (error) {
      console.error('Error loading from storage:', error);
      return { model: null, scaler: null };
    }
  }

  static async saveToStorage(model, scaler = null) {
    try {
      const data = { xgboost_model: model };
      if (scaler) data.feature_scaler = scaler;
      
      await browser.storage.local.set(data);
      return true;
    } catch (error) {
      console.error('Error saving to storage:', error);
      throw error;
    }
  }
}