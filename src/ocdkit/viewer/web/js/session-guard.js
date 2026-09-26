/* session-guard.js — tell the user when the viewer server lost this page's session.
 *
 * Sessions live in the server's memory, so after the server restarts (relaunch,
 * crash, code reload) an open page keeps a session id the server no longer
 * knows. Every API call then fails with 404 {"error": "unknown session"} and the
 * page silently stops responding: slices don't change, axis switches do
 * nothing. This watches API responses for that error and shows a banner with a
 * reload button (reloading creates a new session and restores the view state,
 * which lives in the browser). Loaded before app.js so it sees every request.
 *
 * window.__viewerCheckSession(url) re-requests a URL (e.g. a slice <img> that
 * failed to load, which fetch never saw) so a lost session is detected there too.
 */
(function () {
  'use strict';
  if (typeof window === 'undefined' || !window.fetch) return;
  let shown = false;

  function showBanner() {
    if (shown) return;
    shown = true;
    const el = document.createElement('div');
    el.id = 'sessionLostBanner';
    el.setAttribute('role', 'alert');
    el.style.cssText = [
      'position:fixed', 'top:12px', 'left:50%', 'transform:translateX(-50%)', 'z-index:10000',
      'display:flex', 'align-items:center', 'gap:12px', 'padding:8px 10px 8px 14px',
      'border-radius:10px', 'font:13px -apple-system,system-ui,sans-serif',
      'background:var(--panel-bg, #262626)', 'color:var(--panel-text-color, #e5e5e5)',
      'border:1px solid var(--control-border, #404040)', 'box-shadow:0 4px 18px rgba(0,0,0,.35)',
    ].join(';');
    const msg = document.createElement('span');
    msg.textContent = 'The viewer server restarted, so this page lost its session. Reload to continue.';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Reload';
    btn.style.cssText = 'border:0;border-radius:999px;padding:4px 12px;cursor:pointer;font:inherit;' +
      'background:var(--accent-color, #e5e5e5);color:var(--accent-ink, #171717);';
    btn.addEventListener('click', function () { location.reload(); });
    el.appendChild(msg); el.appendChild(btn);
    (document.body || document.documentElement).appendChild(el);
  }

  function inspect(resp) {
    if (resp && resp.status === 404) {
      resp.clone().json().then(function (j) {
        if (j && j.error === 'unknown session') showBanner();
      }).catch(function () {});
    }
    return resp;
  }

  const origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const p = origFetch(input, init);
    return url.indexOf('/api/') >= 0 ? p.then(inspect) : p;
  };

  window.__viewerCheckSession = function (url) {
    if (shown || !url) return;
    window.fetch(url).catch(function () {});
  };
  window.__viewerSessionLost = function () { return shown; };
})();
