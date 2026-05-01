/**
 * ViewerFileNav — File/image navigation, folder browsing, drag-and-drop.
 *
 * Exports to window.ViewerFileNav following the existing IIFE + classic script pattern.
 * No ES modules — PyWebView breaks with type="module".
 */
(function (global) {
  'use strict';

  // Callbacks supplied via init()
  var _getSessionId = null;
  var _saveBeforeNavigate = null;
  var _onReinitialize = null;

  // In-flight guard — prevents concurrent navigation requests
  var _navigationInFlight = false;

  // Stored change handler for image navigator (to prevent stacking)
  var _navigatorChangeHandler = null;
  var _navigatorElement = null;

  // ---------------------------------------------------------------------------
  // Transition helpers
  // ---------------------------------------------------------------------------

  function transitionReload() {
    window.location.reload();
  }

  function handleNavigationResult(result) {
    if (result && result.ok && result.config && typeof _onReinitialize === 'function') {
      // In-place swap — no reload needed
      try {
        _onReinitialize(result.config);
      } catch (err) {
        console.warn('reinitialize failed, falling back to reload', err);
        transitionReload();
      }
    } else if (result && result.ok) {
      // Server didn't return config — fall back to reload
      transitionReload();
    } else if (result && result.error) {
      console.warn('navigation error', result.error);
    }
  }

  function dimPage() {
    // no-op — user prefers no visual dimming during navigation
  }

  function undimPage() {
    // no-op
  }

  // ---------------------------------------------------------------------------
  // Core navigation
  // ---------------------------------------------------------------------------

  function requestImageChange(opts) {
    if (!opts) { opts = {}; }
    if (_navigationInFlight) { return Promise.resolve(); }
    var path = opts.path || null;
    var direction = opts.direction || null;
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    if (!sessionId) {
      console.warn('Session not initialized; cannot change image');
      return Promise.resolve();
    }
    var payload = { sessionId: sessionId };
    if (typeof path === 'string' && path) {
      payload.path = path;
    }
    if (typeof direction === 'string' && direction) {
      payload.direction = direction;
    }
    _navigationInFlight = true;
    dimPage();
    var save = typeof _saveBeforeNavigate === 'function'
      ? _saveBeforeNavigate()
      : Promise.resolve();
    return save.catch(function () {}).then(function () {
      return fetch('/api/open_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    }).then(function (response) {
      if (!response.ok) {
        return response.text().catch(function () { return 'unknown'; }).then(function (message) {
          console.warn('open_image failed', response.status, message);
          undimPage();
        });
      }
      return response.json().catch(function () { return {}; }).then(function (result) {
        handleNavigationResult(result);
      });
    }).catch(function (err) {
      console.warn('open_image request failed', err);
      undimPage();
    }).then(function () {
      _navigationInFlight = false;
    });
  }

  // Browser-side single-file picker. Uploads bytes to /api/upload_image so it
  // works for remote viewers (no server-side display required).
  var _filePickerInput = null;
  function _ensureFilePickerInput() {
    if (_filePickerInput && document.body.contains(_filePickerInput)) return _filePickerInput;
    var inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = 'image/*,.tif,.tiff,.npy,.npz';
    inp.style.display = 'none';
    document.body.appendChild(inp);
    _filePickerInput = inp;
    return inp;
  }

  function selectImageFile() {
    if (_navigationInFlight) { return Promise.resolve(); }
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    if (!sessionId) {
      console.warn('Session not initialized; cannot pick image');
      return Promise.resolve();
    }
    var inp = _ensureFilePickerInput();
    inp.value = '';
    return new Promise(function (resolve) {
      var done = false;
      function finish(result) {
        if (done) return;
        done = true;
        inp.removeEventListener('change', onChange);
        resolve(result || null);
      }
      function onChange() {
        var f = inp.files && inp.files[0];
        if (!f) return finish(null);
        _navigationInFlight = true;
        var save = typeof _saveBeforeNavigate === 'function'
          ? _saveBeforeNavigate()
          : Promise.resolve();
        save.catch(function () {}).then(function () {
          var fd = new FormData();
          fd.append('sessionId', sessionId);
          fd.append('file', f, f.name);
          return fetch('/api/upload_image', { method: 'POST', body: fd });
        }).then(function (response) {
          if (!response.ok) {
            return response.text().catch(function () { return ''; }).then(function (msg) {
              console.warn('upload_image failed', response.status, msg);
            });
          }
          return response.json().then(function (result) {
            handleNavigationResult(result);
          });
        }).catch(function (err) {
          console.warn('upload_image request failed', err);
        }).then(function () {
          _navigationInFlight = false;
          finish();
        });
      }
      inp.addEventListener('change', onChange, { once: true });
      inp.click();
    });
  }

  // Browser-side folder picker (webkitdirectory). Uploads every image-like
  // file under the chosen directory, then opens the first one. Uses a tight
  // extension allowlist to keep transfer bounded.
  var _folderPickerInput = null;
  var _IMAGE_EXTS = /\.(tif|tiff|png|jpg|jpeg|bmp|gif|webp|npy|npz)$/i;
  function _ensureFolderPickerInput() {
    if (_folderPickerInput && document.body.contains(_folderPickerInput)) return _folderPickerInput;
    var inp = document.createElement('input');
    inp.type = 'file';
    inp.setAttribute('webkitdirectory', '');
    inp.setAttribute('directory', '');
    inp.multiple = true;
    inp.style.display = 'none';
    document.body.appendChild(inp);
    _folderPickerInput = inp;
    return inp;
  }

  function selectImageFolder() {
    if (_navigationInFlight) { return Promise.resolve(); }
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    if (!sessionId) {
      console.warn('Session not initialized; cannot pick folder');
      return Promise.resolve();
    }
    var inp = _ensureFolderPickerInput();
    inp.value = '';
    return new Promise(function (resolve) {
      var done = false;
      function finish() {
        if (done) return;
        done = true;
        inp.removeEventListener('change', onChange);
        resolve(null);
      }
      function onChange() {
        var allFiles = inp.files ? Array.from(inp.files) : [];
        var images = allFiles.filter(function (f) { return _IMAGE_EXTS.test(f.name); });
        if (!images.length) return finish();
        _navigationInFlight = true;
        var save = typeof _saveBeforeNavigate === 'function'
          ? _saveBeforeNavigate()
          : Promise.resolve();
        // Upload each image sequentially. The last response carries the
        // session into the right state; the navigator dropdown then refreshes
        // automatically via handleNavigationResult.
        save.catch(function () {}).then(function () {
          return images.reduce(function (chain, f) {
            return chain.then(function () {
              var fd = new FormData();
              fd.append('sessionId', sessionId);
              fd.append('file', f, f.name);
              return fetch('/api/upload_image', { method: 'POST', body: fd })
                .then(function (r) { return r.ok ? r.json() : null; });
            });
          }, Promise.resolve(null));
        }).then(function (lastResult) {
          if (lastResult) handleNavigationResult(lastResult);
        }).catch(function (err) {
          console.warn('folder upload failed', err);
        }).then(function () {
          _navigationInFlight = false;
          finish();
        });
      }
      inp.addEventListener('change', onChange, { once: true });
      inp.click();
    });
  }

  function openImageFolder(path) {
    if (!path) {
      console.warn('No path provided for openImageFolder');
      return Promise.resolve();
    }
    if (_navigationInFlight) { return Promise.resolve(); }
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    _navigationInFlight = true;
    dimPage();
    var save = typeof _saveBeforeNavigate === 'function'
      ? _saveBeforeNavigate()
      : Promise.resolve();
    return save.catch(function () {}).then(function () {
      return fetch('/api/open_image_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: sessionId, path: path }),
      });
    }).then(function (response) {
      if (!response.ok) {
        return response.text().catch(function () { return 'unknown'; }).then(function (message) {
          console.warn('open_image_folder failed', response.status, message);
          undimPage();
        });
      }
      return response.json().catch(function () { return {}; }).then(function (result) {
        handleNavigationResult(result);
      });
    }).catch(function (err) {
      console.warn('open_image_folder request failed', err);
      undimPage();
    }).then(function () {
      _navigationInFlight = false;
    });
  }

  function openImageByPath(path) {
    if (!path) {
      console.warn('No path provided for openImageByPath');
      return Promise.resolve();
    }
    return requestImageChange({ path: path });
  }

  function navigateDirectory(delta) {
    if (delta === 0) {
      return Promise.resolve();
    }
    var direction = delta > 0 ? 'next' : 'prev';
    return requestImageChange({ direction: direction });
  }

  // ---------------------------------------------------------------------------
  // Drag and drop
  // ---------------------------------------------------------------------------

  var MODEL_EXTENSIONS = ['.pth', '.pt', '.onnx', '.ckpt', '.bin', '.model'];
  var IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'];

  function getFileExtension(name) {
    if (!name) return '';
    var dot = name.lastIndexOf('.');
    return dot >= 0 ? name.slice(dot).toLowerCase() : '';
  }

  function isModelFile(name) {
    if (!name) return false;
    if (MODEL_EXTENSIONS.indexOf(getFileExtension(name)) >= 0) return true;
    // Detect extensionless model files by common naming patterns
    var lower = name.toLowerCase();
    if (/cellpose[_\s]/.test(lower)) return true;
    if (/epoch[_\s]?\d+$/.test(lower)) return true;
    if (/nclasses[_\s]\d+/.test(lower)) return true;
    // No extension and not a recognized image → treat as model
    if (getFileExtension(name) === '' && !isImageFile(name)) return true;
    return false;
  }

  function isImageFile(name) {
    return IMAGE_EXTENSIONS.indexOf(getFileExtension(name)) >= 0;
  }

  function uploadImage(file) {
    var sessionId = typeof _getSessionId === 'function' ? _getSessionId() : null;
    if (!sessionId) {
      return Promise.reject(new Error('no session'));
    }
    var formData = new FormData();
    formData.append('file', file);
    formData.append('sessionId', sessionId);
    return fetch('/api/upload_image', {
      method: 'POST',
      body: formData,
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (t) { throw new Error(t); });
      }
      return response.json();
    });
  }

  function uploadModel(file) {
    var formData = new FormData();
    formData.append('file', file);
    return fetch('/api/upload_model', {
      method: 'POST',
      body: formData,
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (t) { throw new Error(t); });
      }
      return response.json();
    });
  }

  function setupDragAndDrop(viewer, dropOverlay) {
    if (!viewer || !dropOverlay) {
      return;
    }
    var dragDepth = 0;

    var showOverlay = function (text) {
      if (!dropOverlay) return;
      if (text) dropOverlay.textContent = text;
      dropOverlay.classList.add('drop-overlay--visible');
    };

    var hideOverlay = function () {
      dragDepth = 0;
      if (!dropOverlay) return;
      dropOverlay.classList.remove('drop-overlay--visible');
      dropOverlay.textContent = 'Drop file to load';
    };

    viewer.addEventListener('dragenter', function (evt) {
      evt.preventDefault();
      dragDepth += 1;
      showOverlay('Drop file to load');
    });

    viewer.addEventListener('dragover', function (evt) {
      evt.preventDefault();
      evt.dataTransfer.dropEffect = 'copy';
    });

    viewer.addEventListener('dragleave', function (evt) {
      evt.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) {
        hideOverlay();
      }
    });

    viewer.addEventListener('drop', function (evt) {
      evt.preventDefault();
      hideOverlay();
      var files = evt.dataTransfer && evt.dataTransfer.files;
      if (!files || files.length === 0) {
        return;
      }
      var file = files[0];
      var filePath = (file && typeof file.path === 'string') ? file.path
        : (file && file.webkitRelativePath) ? file.webkitRelativePath
        : null;
      var fileName = file ? file.name : null;

      // Detect model files and route to model adding flow
      if (fileName && isModelFile(fileName)) {
        if (filePath) {
          // PyWebView / Electron — we have the real path
          if (typeof window.__viewer_addCustomModel === 'function') {
            window.__viewer_addCustomModel(filePath, fileName);
          }
        } else {
          // Browser mode — upload the file first
          uploadModel(file).then(function (result) {
            if (result && result.ok && result.path && typeof window.__viewer_addCustomModel === 'function') {
              window.__viewer_addCustomModel(result.path, result.name || fileName);
            }
          }).catch(function (err) {
            console.warn('Model upload failed', err);
          });
        }
        return;
      }

      // Otherwise treat as image
      if (filePath) {
        openImageByPath(filePath);
      } else if (file) {
        // Browser mode — upload the file (server validates format)
        uploadImage(file).then(function (result) {
          handleNavigationResult(result);
        }).catch(function (err) {
          console.warn('Image upload failed', err);
        });
      }
    });

    window.addEventListener('dragover', function (evt) {
      evt.preventDefault();
    });
    window.addEventListener('drop', function (evt) {
      evt.preventDefault();
      if (!viewer.contains(evt.target)) {
        hideOverlay();
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Image navigator dropdown setup
  // ---------------------------------------------------------------------------

  function setupImageNavigator(opts) {
    if (!opts) { opts = {}; }
    var imageNavigator = opts.imageNavigator;
    var directoryEntries = opts.directoryEntries || [];
    var currentImageName = opts.currentImageName || null;
    var currentImagePath = opts.currentImagePath || null;
    var truncateFilenameFn = opts.truncateFilename || function (n) { return n; };
    var refreshDropdownFn = opts.refreshDropdown || function () {};
    var getDropdownFn = opts.getDropdown || function () { return null; };

    if (!imageNavigator) {
      return;
    }
    function syncDropdownMenu() {
      var dd = getDropdownFn('imageNavigator');
      if (!dd) return;
      dd.options = Array.from(imageNavigator.options).map(function (o) {
        return {
          value: o.value,
          label: o.textContent || o.value,
          disabled: o.disabled,
          title: o.dataset.fullPath || o.dataset.fullLabel || o.title || o.textContent || o.value,
        };
      });
      if (typeof dd.buildMenu === 'function') dd.buildMenu();
    }

    if (!Array.isArray(directoryEntries) || directoryEntries.length === 0) {
      imageNavigator.innerHTML = '';
      var opt = document.createElement('option');
      opt.value = '';
      opt.textContent = currentImageName || 'Select';
      imageNavigator.appendChild(opt);
      var openFileOption = document.createElement('option');
      openFileOption.value = '__open_file__';
      openFileOption.textContent = 'Open image file...';
      openFileOption.title = 'Open image file';
      imageNavigator.appendChild(openFileOption);
      var openFolderOption = document.createElement('option');
      openFolderOption.value = '__open_folder__';
      openFolderOption.textContent = 'Open image folder...';
      openFolderOption.title = 'Open image folder';
      imageNavigator.appendChild(openFolderOption);
      // Sync the visible menu so the new options actually appear (this branch
      // previously skipped the sync, so the empty-folder dropdown looked
      // blank to the user).
      syncDropdownMenu();
      // Bind the change handler in this branch too — otherwise picking
      // "Open image file/folder" from a fresh viewer (no folder loaded yet)
      // does nothing.
      if (_navigatorChangeHandler && _navigatorElement) {
        _navigatorElement.removeEventListener('change', _navigatorChangeHandler);
      }
      _navigatorElement = imageNavigator;
      _navigatorChangeHandler = function () {
        var v = imageNavigator.value;
        if (v === '__open_file__') {
          selectImageFile().then(function () {
            imageNavigator.value = currentImagePath || '';
            refreshDropdownFn('imageNavigator');
          });
        } else if (v === '__open_folder__') {
          selectImageFolder().then(function () {
            imageNavigator.value = currentImagePath || '';
            refreshDropdownFn('imageNavigator');
          });
        }
      };
      imageNavigator.addEventListener('change', _navigatorChangeHandler);
      return;
    }
    imageNavigator.innerHTML = '';
    directoryEntries.forEach(function (entry) {
      var opt2 = document.createElement('option');
      opt2.value = entry.path || entry.name;
      opt2.textContent = truncateFilenameFn(entry.name);
      opt2.dataset.fullLabel = entry.name;
      opt2.dataset.fullPath = entry.path || entry.name;
      opt2.title = entry.path || entry.name;
      if (entry.isCurrent) {
        opt2.selected = true;
      }
      imageNavigator.appendChild(opt2);
    });
    var openFileOption2 = document.createElement('option');
    openFileOption2.value = '__open_file__';
    openFileOption2.textContent = 'Open image file...';
    openFileOption2.title = 'Open image file';
    imageNavigator.appendChild(openFileOption2);
    var openFolderOption2 = document.createElement('option');
    openFolderOption2.value = '__open_folder__';
    openFolderOption2.textContent = 'Open image folder...';
    openFolderOption2.title = 'Open image folder';
    imageNavigator.appendChild(openFolderOption2);
    var dropdownEntry = getDropdownFn('imageNavigator');
    if (dropdownEntry) {
      dropdownEntry.options = Array.from(imageNavigator.options).map(function (o) {
        return {
          value: o.value,
          label: o.textContent || o.value,
          disabled: o.disabled,
          title: o.dataset.fullPath || o.dataset.fullLabel || o.title || o.textContent || o.value,
        };
      });
      if (typeof dropdownEntry.buildMenu === 'function') {
        dropdownEntry.buildMenu();
      }
    }
    // Remove old change handler to prevent stacking on re-init
    if (_navigatorChangeHandler && _navigatorElement) {
      _navigatorElement.removeEventListener('change', _navigatorChangeHandler);
    }
    _navigatorElement = imageNavigator;
    _navigatorChangeHandler = function () {
      var nextPath = imageNavigator.value;
      if (!nextPath || nextPath === currentImagePath) {
        return;
      }
      if (nextPath === '__open_file__') {
        selectImageFile().then(function () {
          imageNavigator.value = currentImagePath || '';
          refreshDropdownFn('imageNavigator');
        });
        return;
      }
      if (nextPath === '__open_folder__') {
        selectImageFolder().then(function () {
          imageNavigator.value = currentImagePath || '';
          refreshDropdownFn('imageNavigator');
        });
        return;
      }
      openImageByPath(nextPath).catch(function (err) {
        console.warn('openImageByPath failed', err);
      });
    };
    imageNavigator.addEventListener('change', _navigatorChangeHandler);
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  function init(opts) {
    if (!opts) { opts = {}; }
    _getSessionId = typeof opts.getSessionId === 'function' ? opts.getSessionId : null;
    _saveBeforeNavigate = typeof opts.saveBeforeNavigate === 'function' ? opts.saveBeforeNavigate : null;
    _onReinitialize = typeof opts.onReinitialize === 'function' ? opts.onReinitialize : null;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  var api = {
    init: init,
    requestImageChange: requestImageChange,
    selectImageFolder: selectImageFolder,
    selectImageFile: selectImageFile,
    openImageFolder: openImageFolder,
    openImageByPath: openImageByPath,
    navigateDirectory: navigateDirectory,
    setupDragAndDrop: setupDragAndDrop,
    setupImageNavigator: setupImageNavigator,
  };

  global.ViewerFileNav = api;
})(window);
