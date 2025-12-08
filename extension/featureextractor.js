class FeatureExtractor {
  static urlEntropy(s) {
    if (!s || s.length === 0) return 0.0;
    const cnt = {};
    for (const ch of s) {
      cnt[ch] = (cnt[ch] || 0) + 1;
    }
    const probs = Object.values(cnt).map(v => v / s.length);
    return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
  }

  static countSpecialChars(s) {
    return {
      num_dots: (s.match(/\./g) || []).length,
      num_hyphens: (s.match(/-/g) || []).length,
      num_underscores: (s.match(/_/g) || []).length,
      num_slashes: (s.match(/\//g) || []).length,
      num_digits: (s.match(/\d/g) || []).length,
      num_at_signs: (s.match(/@/g) || []).length,
      num_question_marks: (s.match(/\?/g) || []).length,
      num_equals: (s.match(/=/g) || []).length,
      num_ampersands: (s.match(/&/g) || []).length
    };
  }

  static isIpAddress(host) {
    if (!host) return false;
    host = host.split(':')[0];
    const parts = host.split('.');
    if (parts.length !== 4) return false;
    try {
      return parts.every(part => {
        const num = parseInt(part, 10);
        return num >= 0 && num <= 255;
      });
    } catch {
      return false;
    }
  }

  static extractLexical(url) {
    try {
      const urlObj = new URL(url);
      const host = urlObj.hostname.toLowerCase();
      const path = urlObj.pathname;
      const query = urlObj.search.substring(1);
      
      const specialChars = this.countSpecialChars(url);
      
      const suspiciousKeywords = ['login', 'signin', 'verify', 'account', 'secure'];
      const payKeywords = ['pay', 'bank', 'update', 'confirm'];
      const suspiciousTlds = ['.tk', '.ml', '.ga', '.cf', '.gq'];
      const badBrands = ['paypai', 'g00gle', 'faceb00k', 'micros0ft'];
      const urgentKeywords = ['urgent', 'alert', 'warning', 'unlock'];
      const rareTlds = ['.zip', '.kim', '.country', '.science', '.work'];
      
      const urlLower = url.toLowerCase();
      const digitCount = (url.match(/\d/g) || []).length;
      const letterCount = (url.match(/[a-zA-Z]/g) || []).length;
      
      return {
        url_len: url.length,
        host_len: host.length,
        path_len: path.length,
        query_len: query.length,
        ...specialChars,
        has_ip_host: this.isIpAddress(host) ? 1 : 0,
        entropy_host: this.urlEntropy(host),
        entropy_path: this.urlEntropy(path),
        entropy_url: this.urlEntropy(url),
        has_login_kw: suspiciousKeywords.some(kw => urlLower.includes(kw)) ? 1 : 0,
        has_pay_kw: payKeywords.some(kw => urlLower.includes(kw)) ? 1 : 0,
        has_suspicious_tld: suspiciousTlds.some(tld => url.endsWith(tld)) ? 1 : 0,
        protocol_https: urlObj.protocol === 'https:' ? 1 : 0,
        subdomain_count: Math.max(0, host.split('.').length - 2),
        digit_letter_ratio: letterCount > 0 ? digitCount / letterCount : 0,
        has_punycode: url.includes('xn--') ? 1 : 0,
        has_misspelled_brand: badBrands.some(bad => url.includes(bad)) ? 1 : 0,
        keyword_pressure: urgentKeywords.some(kw => urlLower.includes(kw)) ? 1 : 0,
        rare_tld: rareTlds.some(t => url.endsWith(t)) ? 1 : 0,
        path_depth: (path.match(/\//g) || []).length,
        num_params: query ? query.split('&').length : 0
      };
    } catch (e) {
      console.error('Error extracting lexical features:', e);
      return null;
    }
  }

  static extractHtmlFeatures(htmlContent) {
    try {
      const htmlLower = htmlContent.toLowerCase();
      
      const suspiciousKeywords = ['verify', 'suspend', 'confirm', 'urgent', 'click here', 'update', 'secure'];
      const brandTerms = ['bank', 'apple id', 'office365', 'paypal', 'google account'];
      const securityTerms = ['secure login', 'security check', 'validate account'];
      
      return {
        html_len: htmlContent.length,
        num_scripts: (htmlContent.match(/<script/gi) || []).length,
        num_iframes: (htmlContent.match(/<iframe/gi) || []).length,
        num_forms: (htmlContent.match(/<form/gi) || []).length,
        num_inputs: (htmlContent.match(/<input/gi) || []).length,
        num_links: (htmlContent.match(/<a /gi) || []).length,
        num_images: (htmlContent.match(/<img/gi) || []).length,
        has_password_field: (htmlLower.includes('type="password"') || htmlLower.includes("type='password'")) ? 1 : 0,
        has_hidden_input: (htmlLower.includes('type="hidden"') || htmlLower.includes("type='hidden'")) ? 1 : 0,
        num_external_links: (htmlContent.match(/https?:\/\//gi) || []).length,
        num_suspicious_keywords: suspiciousKeywords.filter(kw => htmlLower.includes(kw)).length,
        html_entropy: this.urlEntropy(htmlContent.substring(0, 10000)),
        num_obfuscated_js: (htmlLower.match(/eval\(/g) || []).length + 
                           (htmlLower.match(/atob\(/g) || []).length + 
                           (htmlLower.match(/unescape\(/g) || []).length,
        has_onclick_hooks: htmlLower.includes('onclick=') ? 1 : 0,
        has_onmouseover_hooks: htmlLower.includes('onmouseover=') ? 1 : 0,
        external_script_loads: (htmlLower.match(/src=["']https?/g) || []).length,
        form_action_external: (htmlLower.includes('action="http') || htmlLower.includes("action='http")) ? 1 : 0,
        spoofed_brand_terms: brandTerms.filter(t => htmlLower.includes(t)).length,
        fake_security_indicators: securityTerms.filter(t => htmlLower.includes(t)).length
      };
    } catch (e) {
      console.error('Error extracting HTML features:', e);
      return null;
    }
  }

  static extractFeatures(url, htmlContent) {
    const lexical = this.extractLexical(url);
    const html = this.extractHtmlFeatures(htmlContent);
    
    if (!lexical || !html) return null;
    
    return { ...lexical, ...html };
  }
}