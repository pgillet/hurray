// Language tabs for the Hurray book.
//
// Turns a `<div class="lang-tabs">` that wraps two or more fenced code blocks
// (e.g. ```rust and ```python) into a Rust/Python switcher. The choice is shared
// across every tab group on the page and remembered in localStorage, so picking
// "Python" once shows Python everywhere.
//
// Authoring (in Markdown):
//
//   <div class="lang-tabs">
//
//   ```rust
//   // ...
//   ```
//
//   ```python
//   # ...
//   ```
//
//   </div>
//
// Tabs are grouped by language: one tab per distinct language, and that tab
// toggles *every* code block of that language in the group. This is deliberate —
// different mdBook versions emit a different number of <pre> elements for a
// runnable Rust block, so counting <pre> is unreliable; counting languages is not.
//
// Degrades gracefully: with JavaScript disabled, every code block is shown.

(function () {
  "use strict";

  var STORAGE_KEY = "hurray-code-lang";
  var LABELS = { rust: "Rust", python: "Python", c: "C", bash: "Shell", toml: "TOML" };
  var groups = []; // { order: [lang], byLang: { lang: [pre] }, buttons: [{ lang, button }] }

  function label(lang) {
    return LABELS[lang] || (lang ? lang.charAt(0).toUpperCase() + lang.slice(1) : "Code");
  }

  function langOf(pre) {
    var code = pre.querySelector("code");
    if (!code) { return null; }
    var m = /(?:^|\s)language-([a-z0-9]+)/i.exec(code.className || "");
    return m ? m[1].toLowerCase() : null;
  }

  function preferred() {
    try { return window.localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }
  function remember(lang) {
    try { window.localStorage.setItem(STORAGE_KEY, lang); } catch (e) { /* ignore */ }
  }

  function selectLang(lang) {
    groups.forEach(function (g) {
      // A group shows `lang` if it has it; otherwise it falls back to its first.
      var target = g.byLang[lang] ? lang : g.order[0];
      g.order.forEach(function (l) {
        var active = l === target;
        g.byLang[l].forEach(function (pre) { pre.hidden = !active; });
      });
      g.buttons.forEach(function (b) {
        var active = b.lang === target;
        b.button.setAttribute("aria-selected", active ? "true" : "false");
        b.button.tabIndex = active ? 0 : -1;
      });
    });
  }

  function build(container) {
    if (container.classList.contains("lang-tabs-ready")) { return; }

    // Group every language-tagged <pre> in this container by its language.
    var order = [];
    var byLang = {};
    container.querySelectorAll("pre").forEach(function (pre) {
      if (pre.closest(".lang-tabs") !== container) { return; }
      var lang = langOf(pre);
      if (!lang) { return; }
      if (!byLang[lang]) { byLang[lang] = []; order.push(lang); }
      byLang[lang].push(pre);
    });
    if (order.length < 2) { return; } // nothing to switch between

    var bar = document.createElement("div");
    bar.className = "lang-tabs-bar";
    bar.setAttribute("role", "tablist");
    var buttons = [];

    order.forEach(function (lang) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lang-tab";
      btn.textContent = label(lang);
      btn.setAttribute("role", "tab");
      btn.dataset.lang = lang;
      btn.addEventListener("click", function () {
        remember(lang);
        selectLang(lang);
      });
      bar.appendChild(btn);
      buttons.push({ lang: lang, button: btn });
    });

    // Arrow-key navigation across the tablist.
    bar.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") { return; }
      var idx = buttons.findIndex(function (b) { return b.button === document.activeElement; });
      if (idx < 0) { return; }
      var next = e.key === "ArrowRight" ? (idx + 1) % buttons.length
                                        : (idx - 1 + buttons.length) % buttons.length;
      buttons[next].button.focus();
      remember(buttons[next].lang);
      selectLang(buttons[next].lang);
      e.preventDefault();
    });

    container.insertBefore(bar, container.firstChild);
    container.classList.add("lang-tabs-ready");
    groups.push({ order: order, byLang: byLang, buttons: buttons });
  }

  function init() {
    var containers = document.querySelectorAll(".lang-tabs");
    if (!containers.length) { return; }
    containers.forEach(build);
    if (!groups.length) { return; }

    var pref = preferred();
    var first = groups[0].order[0];
    selectLang(pref && LABELS[pref] !== undefined ? pref : first);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
