/* tile_grid_element.js — <hipr-tile-grid> custom element.
 *
 * A Shadow-DOM CSS-grid of image/canvas tiles. Replaces the SVG-nested-tile
 * layout (each tile a <svg class="ocd-linked-cell" overflow="hidden"> with an
 * <image>) that bleeds in browsers where JupyterLab's
 * `.jp-RenderedHTMLCommon svg { height:auto; max-width:100% }` rewrites the
 * nested <svg> viewport height. Here every tile is a plain HTML box with
 * `overflow:hidden` + `object-fit:cover`, so clipping is the CSS box model —
 * the same robust path the existing RGB/exc overlay canvas already uses.
 *
 * Config (transport-agnostic):
 *   element.tiles = [{src|render, label, kind}]   (set before connect), or
 *   data-tiles='[{"src":"…","label":"R1","kind":"img"}, …]' attribute, plus
 *   data-cols / data-rows.
 *
 *   kind: 'img'    → <img src> (PNG/JXL tile from the in-kernel tile server)
 *         'canvas' → <canvas>; if spec.render(canvas, cell) is given it's called
 *                    once the cell has a non-zero box (for LabelGL/WebGPU tiles)
 *
 * Tiles fill their cell (object-fit:cover); the grid keeps a fixed aspect per
 * cell so rows/cols stay aligned regardless of host width.
 */
(function () {
  'use strict';
  if (typeof customElements === 'undefined') return;
  if (customElements.get('hipr-tile-grid')) return;

  class HiprTileGrid extends HTMLElement {
    connectedCallback() {
      if (this._mounted) return;
      this._mounted = true;
      var cols = parseInt(this.getAttribute('data-cols') || '7', 10);
      var rows = parseInt(this.getAttribute('data-rows') || '2', 10);
      var aspect = this.getAttribute('data-aspect') || '1';
      var gap = this.getAttribute('data-gap') || '3px';
      var tiles = this.tiles ||
        JSON.parse(this.getAttribute('data-tiles') || '[]');

      var root = this.attachShadow({ mode: 'open' });
      root.innerHTML =
        '<style>' +
        ':host{display:block}' +
        '.grid{display:grid;gap:' + gap + ';width:100%;' +
        'grid-template-columns:repeat(' + cols + ',1fr)}' +
        '.cell{position:relative;overflow:hidden;background:#000;' +
        'aspect-ratio:' + aspect + '}' +
        '.cell>img,.cell>canvas{position:absolute;inset:0;width:100%;height:100%;' +
        'object-fit:cover;display:block}' +
        '.lab{position:absolute;left:0;right:0;top:2px;z-index:2;color:#fff;' +
        'text-align:center;' +
        'font:11px/1.2 system-ui,sans-serif;text-shadow:0 0 3px #000,0 0 3px #000;' +
        'pointer-events:none}' +
        '</style>' +
        '<div class="grid"></div>';
      var grid = root.querySelector('.grid');
      this._cells = [];

      tiles.forEach((t) => {
        var cell = document.createElement('div');
        cell.className = 'cell';
        var node;
        if (t.kind === 'canvas') {
          node = document.createElement('canvas');
          if (typeof t.render === 'function') this._deferRender(t, node, cell);
        } else {
          node = document.createElement('img');
          node.loading = 'eager';
          node.decoding = 'async';
          if (t.src) node.src = t.src;
        }
        cell.appendChild(node);
        if (t.label) {
          var lab = document.createElement('div');
          lab.className = 'lab';
          lab.textContent = t.label;
          cell.appendChild(lab);
        }
        grid.appendChild(cell);
        this._cells.push({ cell: cell, node: node, spec: t });
      });
    }

    _deferRender(spec, canvas, cell) {
      var run = function () {
        var r = cell.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) { requestAnimationFrame(run); return; }
        spec.render(canvas, cell);
      };
      requestAnimationFrame(run);
    }
  }

  customElements.define('hipr-tile-grid', HiprTileGrid);
  if (typeof window !== 'undefined') window.HiprTileGrid = HiprTileGrid;
})();
