/**
 * ViewerState — State persistence, binary encoding helpers, and debug logging.
 *
 * Exports to window.ViewerState following the existing IIFE + classic script pattern.
 * No ES modules — PyWebView breaks with type="module".
 */
(function (global) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Binary encoding / decoding helpers
  // ---------------------------------------------------------------------------

  function base64FromUint8(bytes) {
    var binary = '';
    var chunk = 0x8000;
    for (var i = 0; i < bytes.length; i += chunk) {
      var sub = bytes.subarray(i, i + chunk);
      binary += String.fromCharCode.apply(null, sub);
    }
    return btoa(binary);
  }

  function decodeBase64ToUint8(encoded) {
    if (!encoded) {
      return new Uint8Array(0);
    }
    var binary = atob(encoded);
    var len = binary.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function base64FromUint32(array) {
    if (!array || array.length === 0) {
      return '';
    }
    var bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    return base64FromUint8(bytes);
  }

  function uint32FromBase64(encoded, expectedLength) {
    var bytes = decodeBase64ToUint8(encoded);
    if (bytes.length % 4 !== 0) {
      throw new Error('uint32FromBase64 length mismatch');
    }
    var view = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    if (typeof expectedLength === 'number' && expectedLength > 0 && view.length !== expectedLength) {
      var copy = new Uint32Array(expectedLength);
      copy.set(view.subarray(0, Math.min(view.length, expectedLength)));
      return copy;
    }
    return view;
  }

  function base64FromUint8Array(array) {
    if (!array || array.length === 0) {
      return '';
    }
    return base64FromUint8(array instanceof Uint8Array ? array : new Uint8Array(array.buffer));
  }

  // ---------------------------------------------------------------------------
  // History serialization helpers
  // ---------------------------------------------------------------------------

  function encodeHistoryField(array) {
    if (!array || array.length === 0) {
      return '';
    }
    var view = array instanceof Uint32Array ? array : new Uint32Array(array);
    return base64FromUint32(view);
  }

  function decodeHistoryField(value, expectedLength) {
    if (value == null || value === '') {
      if (typeof expectedLength === 'number' && expectedLength > 0) {
        return new Uint32Array(expectedLength);
      }
      return new Uint32Array(0);
    }
    if (value instanceof Uint32Array) {
      if (typeof expectedLength === 'number' && expectedLength > 0 && value.length !== expectedLength) {
        var copy = new Uint32Array(expectedLength);
        copy.set(value.subarray(0, Math.min(value.length, expectedLength)));
        return copy;
      }
      return value;
    }
    if (Array.isArray(value)) {
      return new Uint32Array(value);
    }
    if (typeof value === 'string') {
      return uint32FromBase64(value, expectedLength);
    }
    return new Uint32Array(0);
  }

  // ---------------------------------------------------------------------------
  // Debug logging
  // ---------------------------------------------------------------------------

  var _debugStateSaveFlag = false;
  var _logFn = null;

  function stateDebugEnabled() {
    if (_debugStateSaveFlag) {
      return true;
    }
    if (typeof window !== 'undefined' && window.__VIEWER_SAVE_DEBUG__) {
      return Boolean(window.__VIEWER_SAVE_DEBUG__);
    }
    return false;
  }

  function formatStateDebugPart(part) {
    if (part === null || part === undefined) {
      return String(part);
    }
    if (typeof part === 'object') {
      try {
        return JSON.stringify(part);
      } catch (_) {
        return Object.prototype.toString.call(part);
      }
    }
    return String(part);
  }

  function stateDebugLog() {
    if (!stateDebugEnabled()) {
      return;
    }
    var parts = Array.prototype.slice.call(arguments);
    var message = parts.map(function (part) { return formatStateDebugPart(part); }).join(' ');
    try {
      if (typeof _logFn === 'function') {
        _logFn('[state-save] ' + message);
      } else if (typeof console !== 'undefined' && typeof console.debug === 'function') {
        console.debug('[state-save]', message);
      }
    } catch (err) {
      if (typeof console !== 'undefined' && typeof console.debug === 'function') {
        console.debug('[state-save]', message);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Local state loading
  // ---------------------------------------------------------------------------

  function loadLocalViewerState(localStateKey) {
    if (typeof window === 'undefined' || !window.localStorage) {
      return null;
    }
    try {
      var raw = window.localStorage.getItem(localStateKey);
      if (!raw) {
        return null;
      }
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') {
        return null;
      }
      delete parsed.imageVisible;
      delete parsed.maskVisible;
      return parsed;
    } catch (_) {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // State save scheduling
  // ---------------------------------------------------------------------------

  var _stateSaveTimer = null;
  var _stateDirty = false;
  var _stateDirtySeq = 0;
  var _lastSavedSeq = 0;
  var _isRestoringState = false;

  // Callbacks supplied via init()
  var _collectViewerState = null;
  var _getLocalStateKey = null;
  var _getSessionId = null;
  var _getCurrentImagePath = null;

  function nextStateDirtySeq() {
    _stateDirtySeq += 1;
    return _stateDirtySeq;
  }

  function getStateDirtySeq() {
    return _stateDirtySeq;
  }

  function isStateDirty() {
    return _stateDirty;
  }

  function isRestoringState() {
    return _isRestoringState;
  }

  function setRestoringState(value) {
    _isRestoringState = Boolean(value);
  }

  function resetSaveState() {
    _stateDirty = false;
    _stateDirtySeq = 0;
    _lastSavedSeq = 0;
  }

  function scheduleStateSave(delay) {
    if (delay === undefined) { delay = 600; }
    if (_isRestoringState) {
      stateDebugLog('skip schedule (restoring)', {});
      return;
    }
    _stateDirty = true;
    var scheduledSeq = nextStateDirtySeq();
    var delayMs = Math.max(150, delay | 0);
    stateDebugLog('schedule', { seq: scheduledSeq, delay: delayMs, immediate: false });
    if (_stateSaveTimer) {
      clearTimeout(_stateSaveTimer);
    }
    _stateSaveTimer = setTimeout(function () {
      _stateSaveTimer = null;
      stateDebugLog('timer fire', { seq: scheduledSeq });
      saveViewerState({ seq: scheduledSeq }).catch(function (err) {
        console.warn('saveViewerState failed', err);
        stateDebugLog('save error (scheduled)', { seq: scheduledSeq, message: err && err.message ? err.message : String(err) });
      });
    }, delayMs);
  }

  function saveViewerState(opts) {
    if (!opts) { opts = {}; }
    var immediate = opts.immediate || false;
    var seq = opts.seq != null ? opts.seq : null;

    if (!_stateDirty && !immediate) {
      stateDebugLog('skip save (clean state)', { seq: _stateDirtySeq, immediate: immediate });
      return Promise.resolve(true);
    }
    var requestSeq = typeof seq === 'number' ? seq : _stateDirtySeq;
    stateDebugLog('save start', {
      seq: requestSeq,
      immediate: immediate,
      dirtySeq: _stateDirtySeq,
      lastSavedSeq: _lastSavedSeq,
    });
    var viewerState = typeof _collectViewerState === 'function' ? _collectViewerState() : {};
    var localStateKey = typeof _getLocalStateKey === 'function' ? _getLocalStateKey() : '';
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    var currentImagePath = typeof _getCurrentImagePath === 'function' ? _getCurrentImagePath() : null;

    // Always save to localStorage (works without sessionId)
    if (typeof window !== 'undefined' && window.localStorage && localStateKey) {
      try {
        window.localStorage.setItem(localStateKey, JSON.stringify(viewerState));
      } catch (_) {
        /* ignore */
      }
    }
    // Server-side save requires sessionId
    if (!sessionId) {
      stateDebugLog('skip server save (no session)', { immediate: immediate, seq: seq });
      _stateDirty = false;
      return Promise.resolve(true);
    }
    var payload = {
      sessionId: sessionId,
      imagePath: currentImagePath,
      viewerState: viewerState,
    };
    var body = JSON.stringify(payload);

    if (immediate && typeof navigator !== 'undefined' && navigator.sendBeacon) {
      var blob = new Blob([body], { type: 'application/json' });
      navigator.sendBeacon('/api/save_state', blob);
      _lastSavedSeq = Math.max(_lastSavedSeq, requestSeq);
      if (requestSeq === _stateDirtySeq) {
        _stateDirty = false;
        stateDebugLog('state clean (beacon)', { seq: requestSeq });
      } else {
        stateDebugLog('stale beacon save', { seq: requestSeq, latest: _stateDirtySeq });
      }
      return Promise.resolve(true);
    }

    return fetch('/api/save_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body,
      keepalive: immediate,
    }).then(function (response) {
      if (!response.ok) {
        throw new Error('HTTP ' + response.status);
      }
      _lastSavedSeq = Math.max(_lastSavedSeq, requestSeq);
      if (requestSeq === _stateDirtySeq) {
        _stateDirty = false;
        stateDebugLog('state clean (fetch)', { seq: requestSeq, status: response.status });
      } else {
        stateDebugLog('stale fetch save', { seq: requestSeq, latest: _stateDirtySeq, status: response.status });
      }
      return true;
    }).catch(function (err) {
      if (!immediate) {
        _stateDirty = true;
        stateDebugLog('save failed, state remains dirty', {
          seq: requestSeq,
          message: err && err.message ? err.message : String(err),
        });
      }
      throw err;
    });
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  function init(opts) {
    if (!opts) { opts = {}; }
    _debugStateSaveFlag = Boolean(opts.debugStateSave);
    _logFn = typeof opts.log === 'function' ? opts.log : null;
    _collectViewerState = typeof opts.collectViewerState === 'function' ? opts.collectViewerState : null;
    _getLocalStateKey = typeof opts.getLocalStateKey === 'function' ? opts.getLocalStateKey : null;
    _getSessionId = typeof opts.getSessionId === 'function' ? opts.getSessionId : null;
    _getCurrentImagePath = typeof opts.getCurrentImagePath === 'function' ? opts.getCurrentImagePath : null;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  var api = {
    // Init
    init: init,

    // Binary encoding/decoding
    base64FromUint8: base64FromUint8,
    decodeBase64ToUint8: decodeBase64ToUint8,
    base64FromUint32: base64FromUint32,
    uint32FromBase64: uint32FromBase64,
    base64FromUint8Array: base64FromUint8Array,

    // History serialization
    encodeHistoryField: encodeHistoryField,
    decodeHistoryField: decodeHistoryField,

    // Debug logging
    stateDebugEnabled: stateDebugEnabled,
    stateDebugLog: stateDebugLog,

    // Local state
    loadLocalViewerState: loadLocalViewerState,

    // Save scheduling
    scheduleStateSave: scheduleStateSave,
    saveViewerState: saveViewerState,
    nextStateDirtySeq: nextStateDirtySeq,
    getStateDirtySeq: getStateDirtySeq,
    isStateDirty: isStateDirty,
    isRestoringState: isRestoringState,
    setRestoringState: setRestoringState,
    resetSaveState: resetSaveState,
  };

  global.ViewerState = api;
})(window);
