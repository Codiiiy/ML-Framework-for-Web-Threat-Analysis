(function() {
  'use strict';
  
  let analysisResult = null;
  let warningDisplayed = false;

  async function analyzePage() {
    try {
      const url = window.location.href;
      const htmlContent = document.documentElement.outerHTML;
      
      if (url.startsWith('about:') || url.startsWith('moz-extension://')) {
        return;
      }
      
      browser.runtime.sendMessage({
        action: 'analyzePage',
        url: url,
        htmlContent: htmlContent
      }).then((result) => {
        analysisResult = result;
        
        browser.storage.local.set({
          [`analysis_${url}`]: result,
          lastAnalysis: result
        });
        
        if (result.isPhishing && result.confidence > 0.7 && !warningDisplayed) {
          showPhishingWarning(result);
        }
      }).catch((error) => {
        console.error('Error communicating with background:', error);
      });
    } catch (error) {
      console.error('Error analyzing page:', error);
    }
  }

  function showPhishingWarning(result) {
    warningDisplayed = true;
    
    const overlay = document.createElement('div');
    overlay.id = 'phishing-shield-warning';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.95);
      z-index: 2147483647;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    `;
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    overlay.innerHTML = `
      <div style="background: white; padding: 40px; border-radius: 12px; max-width: 500px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
        <div style="width: 80px; height: 80px; background: #DC2626; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center;">
          <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <h1 style="color: #DC2626; font-size: 28px; margin: 0 0 10px 0; font-weight: 700;">
          Phishing Warning
        </h1>
        <p style="color: #4B5563; font-size: 16px; margin: 0 0 20px 0; line-height: 1.5;">
          This website has been identified as potentially malicious with <strong>${confidencePercent}%</strong> confidence.
        </p>
        <div style="background: #FEF2F2; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 4px solid #DC2626;">
          <p style="color: #991B1B; margin: 0; font-size: 14px; text-align: left;">
            <strong>Detected risks:</strong><br/>
            • This site may be attempting to steal your personal information<br/>
            • Do not enter passwords, credit card numbers, or personal data<br/>
            • The URL may be impersonating a legitimate website
          </p>
        </div>
        <div style="display: flex; gap: 10px; justify-content: center;">
          <button id="phishing-shield-leave" style="background: #DC2626; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s;">
            Leave This Site
          </button>
          <button id="phishing-shield-continue" style="background: transparent; color: #6B7280; border: 2px solid #E5E7EB; padding: 12px 24px; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.2s;">
            Continue Anyway
          </button>
        </div>
        <p style="color: #9CA3AF; font-size: 12px; margin: 20px 0 0 0;">
          Protected by Phishing Shield AI
        </p>
      </div>
    `;
    
    document.body.appendChild(overlay);
    
    document.getElementById('phishing-shield-leave').addEventListener('mouseenter', function() {
      this.style.background = '#B91C1C';
    });
    document.getElementById('phishing-shield-leave').addEventListener('mouseleave', function() {
      this.style.background = '#DC2626';
    });
    
    document.getElementById('phishing-shield-continue').addEventListener('mouseenter', function() {
      this.style.background = '#F9FAFB';
    });
    document.getElementById('phishing-shield-continue').addEventListener('mouseleave', function() {
      this.style.background = 'transparent';
    });
    
    document.getElementById('phishing-shield-leave').addEventListener('click', () => {
      window.location.href = 'about:blank';
    });
    
    document.getElementById('phishing-shield-continue').addEventListener('click', () => {
      overlay.remove();
    });
  }

  browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'startAnalysis') {
      analyzePage();
      sendResponse({ status: 'started' });
    }
    
    if (request.action === 'getAnalysis') {
      sendResponse(analysisResult);
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', analyzePage);
  } else {
    analyzePage();
  }
})();