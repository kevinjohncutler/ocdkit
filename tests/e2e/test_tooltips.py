"""A title set after load must become our tooltip, not the browser's native one.

Otherwise hovering shows the styled tooltip and then the native tooltip on top
of it (a regression that has come back whenever code sets .title at runtime).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_runtime_title_is_adopted(page):
    page.wait_for_function("() => !!document.querySelector('.omni-tooltip')")
    result = page.evaluate(
        """async () => {
          const el = document.createElement('button');
          el.textContent = 'x';
          document.body.appendChild(el);
          el.title = 'first';                       // set after the tooltip system started
          await new Promise((r) => setTimeout(r, 0));
          const a = [el.getAttribute('title'), el.dataset.tooltip];
          el.title = 'second';                      // and changed again later
          await new Promise((r) => setTimeout(r, 0));
          const added = document.createElement('span');
          added.setAttribute('title', 'third');     // a node inserted with a title
          document.body.appendChild(added);
          await new Promise((r) => setTimeout(r, 0));
          return { a, b: [el.getAttribute('title'), el.dataset.tooltip],
                   c: [added.getAttribute('title'), added.dataset.tooltip] };
        }"""
    )
    assert result["a"] == [None, "first"]
    assert result["b"] == [None, "second"]
    assert result["c"] == [None, "third"]


def test_no_native_titles_left_in_the_panels(page):
    page.wait_for_function("() => !!document.querySelector('.omni-tooltip')")
    page.wait_for_timeout(300)
    left = page.evaluate("() => [...document.querySelectorAll('[title]')].map((e) => e.id || e.className)")
    assert left == []
