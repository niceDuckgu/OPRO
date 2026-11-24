/* Project page behaviour: equation rendering, task tabs, and BibTeX copy. */
(function () {
  "use strict";

  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  // Task tabs: without JavaScript all panels stay visible.
  const tabs = Array.prototype.slice.call(document.querySelectorAll('.tabs [role="tab"]'));
  function selectTab(tab) {
    tabs.forEach(function (t) {
      const active = t === tab;
      t.setAttribute("aria-selected", String(active));
      t.tabIndex = active ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !active;
    });
  }
  tabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () { selectTab(tab); });
    tab.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next); next.focus();
    });
  });
  if (tabs.length) selectTab(tabs[0]);

  const copyButton = document.getElementById("copy-bibtex");
  if (copyButton && navigator.clipboard && navigator.clipboard.writeText) {
    copyButton.addEventListener("click", function () {
      const text = document.getElementById("bibtex-text").textContent;
      navigator.clipboard.writeText(text).then(function () {
        copyButton.textContent = "Copied";
        setTimeout(function () { copyButton.textContent = "Copy"; }, 1600);
      }, function () {});
    });
  }
})();
