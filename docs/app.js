/* Project page behaviour: example pickers, task tabs, and BibTeX copy.
   Without JavaScript the page shows the default example and every task panel. */
(function () {
  "use strict";

  const $ = (root, sel) => root.querySelector(sel);
  const $$ = (root, sel) => Array.prototype.slice.call(root.querySelectorAll(sel));

  function press(buttons, active) {
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === active)));
  }

  // ---------- Teaser: MagicBrush examples (UNO, paper Fig. 6) ----------
  const EXAMPLES = {
    logo: "Replace the logo on the sign with a smiley face.",
    flowers: "Let the flowers in the vase be blue.",
  };
  const teaser = document.getElementById("teaser");
  if (teaser) {
    const exampleButtons = $$(teaser, "[data-example]");
    function show(example) {
      const stem = "static/magicbrush/" + example;
      $(teaser, ".js-input").src = stem + "_input.jpg";
      $(teaser, ".js-before").src = stem + "_uno.jpg";
      $(teaser, ".js-after").src = stem + "_uno_opro.jpg";
      $(teaser, ".js-prompt").textContent = EXAMPLES[example];
      press(exampleButtons, exampleButtons.find((b) => b.dataset.example === example));
    }
    exampleButtons.forEach((b) => b.addEventListener("click", () => show(b.dataset.example)));
  }

  // ---------- Instructional editing gallery ----------
  const SHARED = {
    clothes: "Edit clothes into blue cardigan layered with white shirt.",
    style: "Transfer the image into sketch-like photo.",
  };
  const PROMPTS = {
    p1: { color: "Modify color of the knit sweater into blue.", hair: "Modify her hairstyle into glam blonde.", text: "Render text “OPRO” on her sweater." },
    p2: { color: "Modify color of the knit sweater into blue.", hair: "Modify her hairstyle into glam blonde.", text: "Render text “OPRO” on her sweater." },
    p3: { color: "Modify color of the hoodie into pink.", hair: "Modify his hairstyle into straightened black.", text: "Render text “OPRO” on his hoodie." },
  };
  const edits = document.getElementById("edits");
  if (edits) {
    const state = { person: "p1", edit: "clothes" };
    const personButtons = $$(edits, "[data-person]");
    const editButtons = $$(edits, "[data-edit]");
    function render() {
      $(edits, ".js-original").src = "static/edits/" + state.person + "_original.jpg";
      $(edits, ".js-edited").src = "static/edits/" + state.person + "_" + state.edit + ".jpg";
      $(edits, ".js-prompt").textContent = SHARED[state.edit] || PROMPTS[state.person][state.edit];
      press(personButtons, personButtons.find((b) => b.dataset.person === state.person));
      press(editButtons, editButtons.find((b) => b.dataset.edit === state.edit));
    }
    personButtons.forEach((b) => b.addEventListener("click", () => { state.person = b.dataset.person; render(); }));
    editButtons.forEach((b) => b.addEventListener("click", () => { state.edit = b.dataset.edit; render(); }));
    render();
  }

  // ---------- Task tabs ----------
  const tabs = $$(document, '.tabs [role="tab"]');
  function selectTab(tab) {
    tabs.forEach((t) => {
      const active = t === tab;
      t.setAttribute("aria-selected", String(active));
      t.tabIndex = active ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !active;
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      selectTab(next);
      next.focus();
    });
  });
  if (tabs.length) selectTab(tabs[0]);

  // ---------- BibTeX copy ----------
  const copyButton = document.getElementById("copy-bibtex");
  if (copyButton) {
    copyButton.addEventListener("click", () => {
      const code = document.getElementById("bibtex-text");
      const done = () => {
        copyButton.textContent = "Copied";
        setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
      };
      const selectText = () => {
        const range = document.createRange();
        range.selectNodeContents(code);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code.textContent).then(done, selectText);
      } else {
        selectText();
      }
    });
  }
})();
