'use strict';

const assert = require('node:assert/strict');
const {copyText, createController} = require('../../smolrouter/static/vendor/api-key-copy.js');

function element(initial) {
  const listeners = {};
  const classes = new Set();
  return Object.assign({
    value: '',
    textContent: '',
    hidden: true,
    selectionStart: 0,
    selectionEnd: 0,
    attributes: {},
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      remove(...names) { names.forEach((name) => classes.delete(name)); },
      contains(name) { return classes.has(name); },
    },
    focusCalls: 0,
    selectCalls: 0,
    addEventListener(name, callback) { listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    focus() { this.focusCalls += 1; },
    select() { this.selectCalls += 1; this.selectionStart = 0; this.selectionEnd = this.value.length; },
    emit(name) { return listeners[name](); },
  }, initial || {});
}

function copyEnvironment(options) {
  const previousFocus = element();
  if (options && options.disconnectedFocus) previousFocus.isConnected = false;
  const temporaries = [];
  const body = {
    appendChild(node) { node.appended = true; temporaries.push(node); },
  };
  const doc = {
    activeElement: previousFocus,
    body,
    execCalls: 0,
    createElement(tag) {
      assert.equal(tag, 'textarea');
      return {
        value: '',
        style: {},
        attributes: {},
        removed: false,
        setAttribute(name, value) { this.attributes[name] = value; },
        focus() {},
        select() {},
        setSelectionRange(start, end) { this.range = [start, end]; },
        remove() { this.removed = true; },
      };
    },
    execCommand(command) {
      assert.equal(command, 'copy');
      this.execCalls += 1;
      if (options && options.execError) throw options.execError;
      return !(options && options.execFalse);
    },
  };
  const clipboard = options && options.clipboard;
  const win = {isSecureContext: !(options && options.insecure), navigator: clipboard ? {clipboard} : {}};
  return {doc, win, previousFocus, temporaries};
}

function controllerEnvironment(copyImplementation) {
  const windowListeners = {};
  const timers = new Map();
  let timerId = 0;
  const input = element();
  const panel = element({hidden: true});
  const copyButton = element();
  const copyIcon = element({textContent: 'content_copy'});
  const dismissButton = element();
  const status = element();
  const win = {
    confirms: [],
    confirmResult: true,
    reloadCalls: 0,
    clearCalls: [],
    addEventListener(name, callback) { windowListeners[name] = callback; },
    confirm(message) { this.confirms.push(message); return this.confirmResult; },
    location: {reload: () => { win.reloadCalls += 1; }},
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, {callback, delay}); return id; },
    clearTimeout(id) { this.clearCalls.push(id); timers.delete(id); },
  };
  const controller = createController({
    window: win,
    document: {},
    input,
    panel,
    copyButton,
    copyIcon,
    dismissButton,
    status,
    copyText: copyImplementation || (async () => {}),
  });
  return {win, input, panel, copyButton, copyIcon, dismissButton, status, controller, timers, windowListeners};
}

async function run() {
  {
    let modernCalls = 0;
    const env = copyEnvironment({clipboard: {writeText: async (text) => { modernCalls += 1; assert.equal(text, 'secret'); }}});
    await copyText('secret', {window: env.win, document: env.doc});
    assert.equal(modernCalls, 1, 'secure Clipboard API should be preferred');
    assert.equal(env.doc.execCalls, 0, 'secure success must not invoke the fallback');
  }

  for (const setup of [
    {insecure: true, clipboard: {writeText: async () => { throw new Error('must not be called'); }}},
    {},
    {clipboard: {writeText: async () => { throw new Error('denied'); }}},
  ]) {
    const env = copyEnvironment(setup);
    await copyText('fallback-secret', {window: env.win, document: env.doc});
    assert.equal(env.doc.execCalls, 1, 'fallback should execute for insecure, unavailable, or rejected Clipboard API');
    const temporary = env.temporaries[0];
    assert.equal(temporary.value, '', 'temporary secret must be wiped');
    assert.equal(temporary.removed, true, 'temporary textarea must be removed');
    assert.equal(temporary.attributes['aria-hidden'], undefined, 'a focused fallback field must remain exposed');
    assert.equal(temporary.attributes['aria-label'], 'Temporary API key copy field');
    assert.equal(temporary.attributes.tabindex, '-1');
    assert.equal(env.previousFocus.focusCalls, 1, 'focus must be restored');
  }

  for (const setup of [{execFalse: true}, {execError: new Error('blocked')}]) {
    const env = copyEnvironment(setup);
    await assert.rejects(copyText('failed-secret', {window: env.win, document: env.doc}));
    assert.equal(env.temporaries[0].value, '');
    assert.equal(env.temporaries[0].removed, true);
    assert.equal(env.previousFocus.focusCalls, 1);
  }

  {
    const env = copyEnvironment({insecure: true, disconnectedFocus: true});
    await copyText('secret', {window: env.win, document: env.doc});
    assert.equal(env.previousFocus.focusCalls, 0, 'a detached prior element must not receive focus');
  }

  {
    const env = controllerEnvironment();
    env.controller.showSecret('first-secret');
    assert.equal(env.panel.hidden, false);
    assert.equal(env.input.value, 'first-secret');
    await env.copyButton.emit('click');
    assert.equal(env.copyIcon.textContent, 'check');
    assert.equal(env.copyButton.attributes['aria-label'], 'API key copied');
    assert.equal(env.status.textContent, 'API key copied.');
    assert.equal(env.status.classList.contains('success'), true);
    const feedbackTimer = [...env.timers.values()][0];
    assert.equal(feedbackTimer.delay, 2000);

    env.controller.showSecret('second-secret');
    assert.equal(env.input.value, 'second-secret', 'new secret replaces and wipes the old value');
    assert.equal(env.copyIcon.textContent, 'content_copy', 'new secret resets copied feedback');
    assert.equal(env.status.textContent, '');
    assert.equal(env.status.classList.contains('success'), false);
    assert.equal(env.timers.size, 0, 'new secret cancels the prior feedback timer');
  }

  {
    const env = controllerEnvironment();
    env.controller.showSecret('uncopied-secret');
    env.win.confirmResult = false;
    env.dismissButton.emit('click');
    assert.equal(env.win.reloadCalls, 0, 'cancelled uncopied dismissal must retain the secret');
    assert.equal(env.input.value, 'uncopied-secret');
    assert.match(env.win.confirms[0], /will not be shown again/i);

    env.input.selectionStart = 0;
    env.input.selectionEnd = 4;
    env.input.emit('copy');
    env.win.confirmResult = false;
    env.dismissButton.emit('click');
    assert.equal(env.win.reloadCalls, 0, 'a partial selection must not count as copying the generated key');
    assert.equal(env.win.confirms.length, 2);

    env.input.select();
    env.input.emit('copy');
    env.dismissButton.emit('click');
    assert.equal(env.win.confirms.length, 2, 'a full manual copy must bypass the irreversible-loss warning');
    assert.equal(env.win.reloadCalls, 1);
    assert.equal(env.input.value, '');
    assert.equal(env.panel.hidden, true);
  }

  {
    const env = controllerEnvironment(async () => { throw new Error('copy unavailable'); });
    env.controller.showSecret('manual-secret');
    await env.copyButton.emit('click');
    assert.equal(env.status.textContent, 'Automatic copy is unavailable. The key is selected; press Ctrl+C or Command+C to copy it.');
    assert.equal(env.status.classList.contains('warning'), true);
    assert.ok(env.input.focusCalls >= 2 && env.input.selectCalls >= 2, 'failure should select the key for manual copying');
  }

  {
    const env = controllerEnvironment();
    env.controller.showSecret('pagehide-secret');
    await env.copyButton.emit('click');
    assert.equal(env.timers.size, 1);
    env.windowListeners.pagehide();
    assert.equal(env.input.value, '', 'pagehide must wipe the generated secret');
    assert.equal(env.timers.size, 0, 'pagehide must cancel copied feedback cleanup');
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
