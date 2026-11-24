/* Project page behaviour: equation rendering and BibTeX copy. */
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
