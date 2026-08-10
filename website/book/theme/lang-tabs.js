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
// Degrades gracefully: with JavaScript disabled, every code block is shown,
// each prefixed by its language label.

(function () {
  "use strict";

  var STORAGE_KEY = "hurray-code-lang";
  var LABELS = { rust: "Rust", python: "Python", c: "C", bash: "Shell", toml: "TOML" };
  var groups = []; // { el, tabs: [{ lang, pre, button }] }

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
      // A group shows `lang` if it has it; otherwise it keeps its first tab.
      var has = g.tabs.some(function (t) { return t.lang === lang; });
      var target = has ? lang : g.tabs[0].lang;
      g.tabs.forEach(function (t) {
        var active = t.lang === target;
        t.pre.hidden = !active;
        t.button.setAttribute("aria-selected", active ? "true" : "false");
        t.button.tabIndex = active ? 0 : -1;
      });
    });
  }

  function build(container) {
    var pres = [];
    // Only direct <pre> descendants of this container are tabbed.
    container.querySelectorAll("pre").forEach(function (pre) {
      if (pre.closest(".lang-tabs") === container) { pres.push(pre); }
    });
    if (pres.length < 2) { return; } // nothing to switch

    var tabs = [];
    var bar = document.createElement("div");
    bar.className = "lang-tabs-bar";
    bar.setAttribute("role", "tablist");

    pres.forEach(function (pre) {
      var lang = langOf(pre) || "code";
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
      tabs.push({ lang: lang, pre: pre, button: btn });
    });

    // Arrow-key navigation within the tablist.
    bar.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") { return; }
      var idx = tabs.findIndex(function (t) { return t.button === document.activeElement; });
      if (idx < 0) { return; }
      var next = e.key === "ArrowRight" ? (idx + 1) % tabs.length
                                        : (idx - 1 + tabs.length) % tabs.length;
      tabs[next].button.focus();
      remember(tabs[next].lang);
      selectLang(tabs[next].lang);
      e.preventDefault();
    });

    container.insertBefore(bar, container.firstChild);
    container.classList.add("lang-tabs-ready");
    groups.push({ el: container, tabs: tabs });
  }

  function init() {
    var containers = document.querySelectorAll(".lang-tabs");
    if (!containers.length) { return; }
    containers.forEach(build);
    if (!groups.length) { return; }

    // Default to the remembered language, else the first group's first tab.
    var pref = preferred();
    var known = groups[0].tabs.map(function (t) { return t.lang; });
    selectLang(pref && LABELS[pref] !== undefined ? pref : known[0]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
