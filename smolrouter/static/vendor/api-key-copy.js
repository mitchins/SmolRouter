(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.smolrouterApiKeyCopy = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const COPY_LABEL = 'Copy API key';
  const COPIED_LABEL = 'API key copied';

  function legacyCopy(text, doc) {
    const previousFocus = doc.activeElement;
    const temporary = doc.createElement('textarea');
    temporary.setAttribute('readonly', '');
    temporary.setAttribute('aria-label', 'Temporary API key copy field');
    temporary.setAttribute('tabindex', '-1');
    temporary.style.position = 'fixed';
    temporary.style.left = '-9999px';
    temporary.style.opacity = '0';
    temporary.value = text;
    doc.body.appendChild(temporary);
    try {
      temporary.focus();
      temporary.select();
      temporary.setSelectionRange(0, temporary.value.length);
      if (!doc.execCommand('copy')) throw new Error('Copy command was rejected');
    } finally {
      temporary.value = '';
      temporary.remove();
      if (previousFocus && previousFocus.isConnected !== false && typeof previousFocus.focus === 'function') {
        previousFocus.focus();
      }
    }
  }

  async function copyText(text, options) {
    const win = options && options.window ? options.window : window;
    const doc = options && options.document ? options.document : document;
    const clipboard = win.navigator && win.navigator.clipboard;
    if (win.isSecureContext && clipboard && typeof clipboard.writeText === 'function') {
      try {
        await clipboard.writeText(text);
        return;
      } catch (_) {
        // Firefox and hardened browsers can reject Clipboard API calls even when exposed.
      }
    }
    legacyCopy(text, doc);
  }

  function createController(options) {
    const win = options.window;
    const input = options.input;
    const panel = options.panel;
    const copyButton = options.copyButton;
    const copyIcon = options.copyIcon;
    const dismissButton = options.dismissButton;
    const status = options.status;
    const performCopy = options.copyText || ((text) => copyText(text, {window: win, document: options.document}));
    let secretCopied = false;
    let copyResetTimer = null;

    function clearResetTimer() {
      if (copyResetTimer !== null) {
        win.clearTimeout(copyResetTimer);
        copyResetTimer = null;
      }
    }

    function resetCopyFeedback() {
      clearResetTimer();
      copyIcon.textContent = 'content_copy';
      copyButton.setAttribute('aria-label', COPY_LABEL);
      copyButton.setAttribute('title', COPY_LABEL);
      status.textContent = '';
      status.classList.remove('success', 'warning');
    }

    function markCopied() {
      secretCopied = true;
      clearResetTimer();
      copyIcon.textContent = 'check';
      copyButton.setAttribute('aria-label', COPIED_LABEL);
      copyButton.setAttribute('title', COPIED_LABEL);
      status.textContent = 'API key copied.';
      status.classList.remove('warning');
      status.classList.add('success');
      copyResetTimer = win.setTimeout(resetCopyFeedback, 2000);
    }

    function clearSecret() {
      clearResetTimer();
      input.value = '';
      secretCopied = false;
      resetCopyFeedback();
    }

    function showSecret(secret) {
      clearSecret();
      input.value = secret;
      panel.hidden = false;
      input.focus();
      input.select();
    }

    async function handleCopy() {
      try {
        await performCopy(input.value);
        markCopied();
      } catch (_) {
        resetCopyFeedback();
        status.textContent = 'Automatic copy is unavailable. The key is selected; press Ctrl+C or Command+C to copy it.';
        status.classList.add('warning');
        input.focus();
        input.select();
      }
    }

    function handleDismiss() {
      if (input.value && !secretCopied && !win.confirm('This key will not be shown again. Dismiss without copying?')) return;
      clearSecret();
      panel.hidden = true;
      win.location.reload();
    }

    copyButton.addEventListener('click', handleCopy);
    input.addEventListener('copy', () => {
      if (input.value && input.selectionStart === 0 && input.selectionEnd === input.value.length) markCopied();
    });
    dismissButton.addEventListener('click', handleDismiss);
    win.addEventListener('pagehide', clearSecret);

    return {showSecret: showSecret, clearSecret: clearSecret};
  }

  return {copyText: copyText, createController: createController};
});
