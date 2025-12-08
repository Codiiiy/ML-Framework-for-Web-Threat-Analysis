document.addEventListener('DOMContentLoaded', async () => {
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const statusCard = document.getElementById('statusCard');
  const statusIcon = document.getElementById('statusIcon');
  const statusText = document.getElementById('statusText');
  const confidenceText = document.getElementById('confidenceText');
  const featuresToggle = document.getElementById('featuresToggle');
  const featuresContent = document.getElementById('featuresContent');
  const toggleIcon = document.getElementById('toggleIcon');


  browser.runtime.sendMessage({ action: 'checkModelStatus' }).then((response) => {
    if (!response.isLoaded) {
      showError('Model not loaded. Please upload your trained model.');
    }
  }).catch((error) => {
    showError('Error checking model status: ' + error.message);
  });

  loadAnalysis();

  

  if (featuresToggle) {
    featuresToggle.addEventListener('click', () => {
      console.log('Features toggle clicked');
      featuresContent.classList.toggle('show');
      toggleIcon.textContent = featuresContent.classList.contains('show') ? '▲' : '▼';
    });
  }

  async function loadAnalysis() {
    loading.style.display = 'block';
    results.style.display = 'none';

    try {
      const tabs = await browser.tabs.query({ active: true, currentWindow: true });
      const tab = tabs[0];
      
      if (!tab) {
        showError('Unable to get current tab');
        return;
      }

      browser.storage.local.get([`analysis_${tab.url}`, 'lastAnalysis']).then((data) => {
        const analysis = data[`analysis_${tab.url}`] || data.lastAnalysis;
        
        if (analysis) {
          displayAnalysis(analysis);
        } else {
          browser.tabs.sendMessage(tab.id, { action: 'startAnalysis' }).then(() => {
            setTimeout(() => {
              browser.storage.local.get('lastAnalysis').then((data) => {
                if (data.lastAnalysis) {
                  displayAnalysis(data.lastAnalysis);
                } else {
                  showError('Unable to analyze this page');
                }
              });
            }, 1000);
          }).catch(() => {
            showError('Unable to analyze this page');
          });
        }
      });
    } catch (error) {
      showError('Error: ' + error.message);
    }
  }

  function displayAnalysis(analysis) {
    loading.style.display = 'none';
    results.style.display = 'block';

    if (analysis.error) {
      showError(analysis.error);
      return;
    }

    const isPhishing = analysis.isPhishing;
    const rawConfidence = analysis.confidence;
    const displayConfidence = isPhishing ? rawConfidence : (1 - rawConfidence);
    const confidence = (displayConfidence * 100).toFixed(1);

    if (isPhishing) {
      statusCard.className = 'status-card danger';
      statusIcon.textContent = '⚠️';
      statusText.textContent = 'Potential Phishing';
      statusText.style.color = '#991B1B';
    } else {
      statusCard.className = 'status-card safe';
      statusIcon.textContent = '✓';
      statusText.textContent = 'Looks Safe';
      statusText.style.color = '#065F46';
    }
    
    confidenceText.textContent = `Confidence: ${confidence}%`;

    if (analysis.features) {
      const f = analysis.features;
      console.log('Features received:', Object.keys(f).length, 'features');
      
      document.getElementById('urlLen').textContent = f.url_len || '-';
      document.getElementById('hasHttps').textContent = f.protocol_https ? 'Yes' : 'No';
      document.getElementById('numForms').textContent = f.num_forms || '0';
      document.getElementById('hasPassword').textContent = f.has_password_field ? 'Yes' : 'No';

      const featuresList = Object.entries(f)
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(([key, value]) => {
          const displayValue = typeof value === 'number' ? 
            (Number.isInteger(value) ? value : value.toFixed(4)) : value;
          return `<div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #E5E7EB;">
            <span style="color: #6B7280;">${key}</span>
            <span style="font-weight: 600;">${displayValue}</span>
          </div>`;
        })
        .join('');
      
      featuresContent.innerHTML = featuresList;
      console.log('Features HTML populated:', featuresList.length, 'characters');
    } else {
      console.warn('No features in analysis result');
    }
  }

function showError(message) {
  const errorMessage = document.getElementById('errorMessage');
  if (errorMessage) {
    errorMessage.style.display = 'block';
    errorMessage.textContent = message;
  } else {
    console.error(message);
  }
  loading.style.display = 'none';
}

  function showSuccess(message) {
    uploadStatus.style.display = 'block';
    uploadStatus.className = 'success';
    uploadStatus.textContent = message;
    
    setTimeout(() => {
      uploadStatus.style.display = 'none';
    }, 3000);
  }
});