function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatInline(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function typesetMath(container) {
  if (!container || !window.MathJax || !window.MathJax.typesetPromise) return;
  if (window.MathJax.typesetClear) {
    window.MathJax.typesetClear([container]);
  }
  window.MathJax.typesetPromise([container]).catch(() => {});
}

function renderTutorResponse(container, rawText) {
  const text = (rawText || "")
    .replace(/\r\n/g, "\n")
    .replace(/\\\\\(/g, "\\(")
    .replace(/\\\\\)/g, "\\)")
    .replace(/\\\\\[/g, "\\[")
    .replace(/\\\\\]/g, "\\]");
  const lines = text.split("\n");
  const html = [];
  let inList = false;
  let inOrderedList = false;

  function closeLists() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
    if (inOrderedList) {
      html.push("</ol>");
      inOrderedList = false;
    }
  }

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      closeLists();
      continue;
    }

    // Preserve display math blocks across multiple lines.
    if (trimmed === "\\[" || trimmed === "$$") {
      closeLists();
      const endToken = trimmed === "\\[" ? "\\]" : "$$";
      const mathLines = [];
      i += 1;
      while (i < lines.length && lines[i].trim() !== endToken) {
        mathLines.push(lines[i]);
        i += 1;
      }
      const mathContent = escapeHtml(mathLines.join("\n").trim());
      const open = trimmed === "\\[" ? "\\[" : "$$";
      const close = endToken;
      html.push(`<div class="math-block">${open}${mathContent}${close}</div>`);
      continue;
    }

    const markdownHeading = trimmed.match(/^#{1,3}\s+(.+)$/);
    if (markdownHeading) {
      closeLists();
      html.push(`<h3 class="response-heading">${formatInline(markdownHeading[1])}</h3>`);
      continue;
    }

    const boldHeading = trimmed.match(/^\*\*(.+)\*\*:?$/);
    if (boldHeading) {
      closeLists();
      const heading = boldHeading[1].replace(/:\s*$/, "");
      html.push(`<h3 class="response-heading">${formatInline(heading)}</h3>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        closeLists();
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatInline(bullet[1])}</li>`);
      continue;
    }

    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      if (!inOrderedList) {
        closeLists();
        html.push("<ol>");
        inOrderedList = true;
      }
      html.push(`<li>${formatInline(ordered[1])}</li>`);
      continue;
    }

    closeLists();
    html.push(`<p>${formatInline(trimmed)}</p>`);
  }

  closeLists();
  container.innerHTML = html.join("");
  typesetMath(container);
}

function renderSources(container, sources) {
  const items = Array.isArray(sources) ? sources : [];
  if (!items.length) {
    container.innerHTML = '<p class="muted">No sources returned for this response.</p>';
    return;
  }

  const cards = items
    .map((source) => {
      const score = Number(source.score ?? 0).toFixed(3);
      return `
        <article class="source-item">
          <div class="source-meta">
            <span class="source-doc">${escapeHtml(source.pdf)}</span>
            <span class="source-badge">p${escapeHtml(source.page)}</span>
            <span class="source-badge">score ${score}</span>
          </div>
          <p class="source-preview">${escapeHtml(source.preview || "")}</p>
          <code class="source-id">${escapeHtml(source.chunk_id || "")}</code>
        </article>
      `;
    })
    .join("");

  container.innerHTML = cards;
}

function setActiveNav() {
  const path = window.location.pathname;
  const links = document.querySelectorAll(".nav-link[data-path]");
  links.forEach((link) => {
    const expected = link.getAttribute("data-path");
    if (expected && path.startsWith(expected)) {
      link.classList.add("is-active");
    }
  });
}

window.renderTutorResponse = renderTutorResponse;
window.renderSources = renderSources;
window.typesetMath = typesetMath;

document.addEventListener("DOMContentLoaded", setActiveNav);
