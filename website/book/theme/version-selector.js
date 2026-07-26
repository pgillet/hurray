// Version selector + "dev" banner for the Hurray book (ADR-028).
//
// Reads /versions.json at the deployed site root and injects a dropdown into the mdBook
// menu bar so readers can switch between the dev build and every released version. The site
// root is derived from the URL (everything before "/docs/<version>/"), so this works whether
// the site is served at a domain root or a GitHub Pages project sub-path.
//
// Degrades silently when versions.json is absent (e.g. a local `mdbook build`).

(function () {
  "use strict";

  var path = window.location.pathname;
  var marker = "/docs/";
  var i = path.indexOf(marker);
  if (i < 0) { return; }

  var root = path.slice(0, i);                     // "" or e.g. "/hurray"
  var current = path.slice(i + marker.length).split("/")[0]; // "dev" | "0.1.0" | "stable"

  function versionUrl(id) { return root + "/docs/" + id + "/"; }

  fetch(root + "/versions.json", { cache: "no-store" })
    .then(function (r) { if (!r.ok) { throw new Error("no manifest"); } return r.json(); })
    .then(function (data) {
      var versions = (data && data.versions) || [];

      var select = document.createElement("select");
      select.className = "hurray-version-select";
      select.setAttribute("aria-label", "Documentation version");

      // If we are on /docs/stable/, surface it as a distinct, selected option.
      if (current === "stable") {
        var stableOpt = document.createElement("option");
        stableOpt.value = "stable";
        stableOpt.textContent = "stable";
        stableOpt.selected = true;
        select.appendChild(stableOpt);
      }

      versions.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.label || v.id;
        if (v.id === current) { opt.selected = true; }
        select.appendChild(opt);
      });

      select.addEventListener("change", function () {
        window.location.href = versionUrl(select.value);
      });

      var bar = document.querySelector("#menu-bar .right-buttons") ||
                document.querySelector("#menu-bar");
      if (bar) {
        var wrap = document.createElement("div");
        wrap.className = "hurray-version-wrap";
        wrap.appendChild(select);
        bar.insertBefore(wrap, bar.firstChild);
      }

      // Unreleased banner on the dev build.
      if (current === "dev") {
        var banner = document.createElement("div");
        banner.className = "hurray-dev-banner";
        banner.innerHTML = "You are reading the <strong>unreleased development</strong> documentation.";
        if (data.stable) {
          banner.innerHTML += ' <a href="' + versionUrl(data.stable) +
            '">View the latest release &rarr;</a>';
        }
        var content = document.querySelector("#content main") ||
                      document.querySelector("#content") || document.body;
        content.insertBefore(banner, content.firstChild);
      }
    })
    .catch(function () { /* manifest unavailable: no selector, no banner */ });
})();
