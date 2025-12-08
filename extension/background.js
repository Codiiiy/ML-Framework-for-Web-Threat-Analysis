let predictor = null;
let isModelLoaded = false;

async function loadModelFromFile() {
  try {
    console.log('Attempting to load model from extension/model/model.json...');
    
    const modelUrl = browser.runtime.getURL('model/model.json');
    console.log('Model URL:', modelUrl);
    
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status}`);
    }
    
    const modelData = await response.json();
    console.log('Model file loaded:', modelData.n_estimators, 'trees');
    
    await browser.storage.local.set({ xgboost_model: modelData });
    console.log('Model saved to storage');
    
    return { model: modelData, scaler: null };
  } catch (error) {
    console.error(' Error loading model from file:', error);
    throw error;
  }
}

async function initializePredictor() {
  try {
    console.log('Initializing predictor...');
    
    let result = await ModelLoader.loadFromStorage();
    
    if (!result.model) {
      console.log(' No model in storage, loading from file...');
      result = await loadModelFromFile();
    } else {
      console.log(' Model loaded from storage');
    }
    
    if (!result.model) {
      console.log(' No model found');
      return false;
    }
    
    predictor = new XGBoostPredictor();
    await predictor.loadModel(result.model, result.scaler);
    isModelLoaded = true;
    return true;
  } catch (error) {
    return false;
  }
}


async function analyzePage(url, htmlContent) {
  if (!isModelLoaded) {
    console.log('Model not loaded, initializing...');
    await initializePredictor();
  }
  
  if (!predictor) {
    return {
      error: 'Model not loaded',
      isPhishing: false,
      confidence: 0
    };
  }
  
  try {
    const features = FeatureExtractor.extractFeatures(url, htmlContent);
    
    if (!features) {
      return {
        error: 'Feature extraction failed',
        isPhishing: false,
        confidence: 0
      };
    }
    
    const result = predictor.predict(features);
    console.log('🔍 Analysis result:', {
      url: url.substring(0, 50) + '...',
      isPhishing: result.isPhishing,
      confidence: (result.probability * 100).toFixed(1) + '%'
    });
    
    return {
      isPhishing: result.isPhishing,
      confidence: result.probability,
      rawScore: result.rawScore,
      features: features,
      url: url
    };
  } catch (error) {
    console.error('Error analyzing page:', error);
    return {
      error: error.message,
      isPhishing: false,
      confidence: 0
    };
  }
}


browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzePage') {
    analyzePage(request.url, request.htmlContent)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true; 
  }
  
  if (request.action === 'uploadModel') {
    ModelLoader.saveToStorage(request.model, request.scaler)
      .then(() => initializePredictor())
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
  
  if (request.action === 'checkModelStatus') {
    sendResponse({ isLoaded: isModelLoaded });
    return false;
  }
  
  if (request.action === 'reloadModel') {
    loadModelFromFile()
      .then(() => initializePredictor())
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
});

function updateBadge(tabId, result) {
  if (result.isPhishing) {
    browser.browserAction.setBadgeBackgroundColor({ color: '#DC2626', tabId });
    browser.browserAction.setBadgeText({ text: '⚠', tabId });
  } else {
    browser.browserAction.setBadgeBackgroundColor({ color: '#16A34A', tabId });
    browser.browserAction.setBadgeText({ text: '✓', tabId });
  }
}

browser.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    browser.tabs.sendMessage(tabId, { action: 'startAnalysis' }).catch(() => {
    });
  }
});

console.log(' Phishing Shield starting...');
initializePredictor();