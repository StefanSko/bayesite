"use strict";

(() => {
  const MAX_MANIFEST_BYTES = 1024 * 1024;
  const MAX_OBJECT_BYTES = 64 * 1024 * 1024;
  const MAX_RECORDS = 256;
  const MAX_DEPTH = 64;
  const SNAPSHOT_DOMAIN = new TextEncoder().encode("bayesite-investigation-snapshot-v0\0");
  const digestPattern = /^[0-9a-f]{64}$/;

  class StrictJsonParser {
    constructor(text) { this.text = text; this.i = 0; }
    error(message) { throw new Error(`${message} at character ${this.i}`); }
    whitespace() { while (/\s/.test(this.text[this.i] || "")) this.i += 1; }
    parse() {
      const value = this.value(0);
      this.whitespace();
      if (this.i !== this.text.length) this.error("Trailing JSON content");
      return value;
    }
    value(depth) {
      if (depth > MAX_DEPTH) this.error(`JSON exceeds depth ${MAX_DEPTH}`);
      this.whitespace();
      const c = this.text[this.i];
      if (c === "{") return this.object(depth + 1);
      if (c === "[") return this.array(depth + 1);
      if (c === '"') return this.string();
      for (const [token, value] of [["true", true], ["false", false], ["null", null]]) {
        if (this.text.startsWith(token, this.i)) { this.i += token.length; return value; }
      }
      const match = this.text.slice(this.i).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if (match) {
        this.i += match[0].length;
        const number = Number(match[0]);
        if (!Number.isFinite(number)) this.error("Non-finite JSON number");
        return number;
      }
      this.error("Invalid JSON value");
    }
    string() {
      const start = this.i;
      this.i += 1;
      let escaped = false;
      while (this.i < this.text.length) {
        const c = this.text[this.i++];
        if (!escaped && c === '"') return JSON.parse(this.text.slice(start, this.i));
        if (!escaped && c === "\\") escaped = true;
        else escaped = false;
      }
      this.error("Unterminated JSON string");
    }
    object(depth) {
      this.i += 1;
      const object = Object.create(null);
      const names = new Set();
      this.whitespace();
      if (this.text[this.i] === "}") { this.i += 1; return object; }
      let count = 0;
      while (true) {
        this.whitespace();
        if (this.text[this.i] !== '"') this.error("Object key must be a string");
        const name = this.string();
        if (names.has(name)) this.error(`Duplicate JSON field ${JSON.stringify(name)}`);
        names.add(name);
        this.whitespace();
        if (this.text[this.i++] !== ":") this.error("Object key needs ':'");
        object[name] = this.value(depth);
        count += 1;
        if (count > MAX_RECORDS * 16) this.error("JSON object has too many fields");
        this.whitespace();
        const delimiter = this.text[this.i++];
        if (delimiter === "}") return object;
        if (delimiter !== ",") this.error("Object needs ',' or '}'");
      }
    }
    array(depth) {
      this.i += 1;
      const array = [];
      this.whitespace();
      if (this.text[this.i] === "]") { this.i += 1; return array; }
      while (true) {
        array.push(this.value(depth));
        if (array.length > MAX_RECORDS * 16) this.error("JSON array is too large to display");
        this.whitespace();
        const delimiter = this.text[this.i++];
        if (delimiter === "]") return array;
        if (delimiter !== ",") this.error("Array needs ',' or ']'");
      }
    }
  }

  function parseJsonStrict(text) { return new StrictJsonParser(text).parse(); }
  function bytesHex(bytes) { return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join(""); }
  async function sha256(bytes) {
    if (!globalThis.crypto || !globalThis.crypto.subtle) return null;
    return bytesHex(new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes)));
  }
  function joinBytes(left, right) {
    const result = new Uint8Array(left.length + right.length);
    result.set(left); result.set(right, left.length); return result;
  }
  async function readLimited(response, limit, label) {
    if (!response.ok) throw new Error(`Cannot load ${label}: HTTP ${response.status}`);
    const stated = Number(response.headers.get("content-length") || 0);
    if (stated > limit) throw new Error(`${label} exceeds ${limit} bytes`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length > limit) throw new Error(`${label} exceeds ${limit} bytes`);
    return bytes;
  }
  function decode(bytes, label) {
    try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
    catch (_) { throw new Error(`${label} is not UTF-8`); }
  }
  function requireObject(value, label) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
    return value;
  }
  function requireArray(value, label) {
    if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
    if (value.length > MAX_RECORDS) throw new Error(`${label} exceeds ${MAX_RECORDS} records`);
    return value;
  }
  function text(value) { return value === null || value === undefined ? "—" : String(value); }
  function safeIdentifier(value, label) {
    if (typeof value !== "string" || value.length < 1 || value.length > 128 || !/^[A-Za-z0-9._-]+$/.test(value)) {
      throw new Error(`${label} must use only 1–128 ASCII letters, digits, '.', '_' or '-'`);
    }
    return value;
  }
  function el(tag, content, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = text(content);
    return node;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function facts(entries) {
    const list = el("dl", undefined, "card");
    for (const [name, value] of entries) {
      const row = el("div", undefined, "fact");
      row.append(el("dt", name), el("dd", value)); list.append(row);
    }
    return list;
  }
  function table(headers, rows) {
    const result = el("table");
    const head = el("thead"); const headRow = el("tr");
    headers.forEach(header => headRow.append(el("th", header))); head.append(headRow); result.append(head);
    const body = el("tbody");
    rows.slice(0, MAX_RECORDS).forEach(row => {
      const tr = el("tr"); row.forEach(cell => tr.append(el("td", cell))); body.append(tr);
    });
    result.append(body);
    if (rows.length > MAX_RECORDS) result.after(el("p", `Truncated ${rows.length - MAX_RECORDS} rows.`, "small"));
    return result;
  }
  function referencePath(reference) {
    requireObject(reference, "artifact reference");
    if (!digestPattern.test(reference.sha256 || "")) throw new Error("Artifact digest is not lowercase SHA-256");
    if (!Number.isSafeInteger(reference.bytes) || reference.bytes < 0 || reference.bytes > MAX_OBJECT_BYTES) {
      throw new Error("Artifact byte length is outside viewer limits");
    }
    return `bundle/objects/sha256/${reference.sha256}`;
  }
  async function loadObject(reference, label) {
    const bytes = await readLimited(await fetch(referencePath(reference), { cache: "no-store" }), MAX_OBJECT_BYTES, label);
    if (bytes.length !== reference.bytes) throw new Error(`${label} byte length failed integrity check`);
    const actual = await sha256(bytes);
    if (actual !== null && actual !== reference.sha256) throw new Error(`${label} SHA-256 integrity failure`);
    return { bytes, verified: actual !== null };
  }
  function outputFor(manifest, evidence) {
    const execution = requireArray(manifest.executions, "executions").find(item => item.id === evidence.execution);
    if (!execution || execution.outcome !== "completed" || !execution.output) return null;
    const recipe = requireArray(manifest.recipes, "recipes").find(item => item.id === execution.recipe);
    return recipe ? { execution, recipe, output: execution.output } : null;
  }
  function diagnosticRows(report) {
    const rhat = requireObject(report.rhat, "diagnostics rhat");
    const ess = requireObject(report.ess, "diagnostics ess");
    const parameters = Array.from(new Set([...Object.keys(rhat), ...Object.keys(ess)]));
    if (parameters.length > MAX_RECORDS) throw new Error(`diagnostics exceeds ${MAX_RECORDS} parameters`);
    return parameters.map(parameter => ({ parameter, rhat: rhat[parameter], ess: ess[parameter] }));
  }
  function evidenceRows(manifest, ancestors) {
    const rows = requireArray(manifest.evidence, "evidence").map(evidence => ({
      name: evidence.name,
      status: evidence.status,
      origin: "current snapshot",
      manifest,
      linked: outputFor(manifest, evidence),
    }));
    for (const ancestor of ancestors) {
      for (const evidence of requireArray(ancestor.manifest.evidence, "source evidence")) {
        rows.push({
          name: evidence.name,
          status: "historical",
          origin: `source sha256:${ancestor.snapshotId}`,
          manifest: ancestor.manifest,
          linked: outputFor(ancestor.manifest, evidence),
        });
        if (rows.length > MAX_RECORDS) throw new Error(`combined evidence exceeds ${MAX_RECORDS} records`);
      }
    }
    return rows;
  }
  function recorded(value) {
    return value !== null && typeof value === "object" ? JSON.stringify(value) : value;
  }
  function prettyExpression(value) { return JSON.stringify(value); }

  async function loadAncestry(manifest, rootSnapshotId) {
    const ancestors = [];
    const seen = new Set([rootSnapshotId]);
    let current = manifest;
    let verified = true;
    while (current.source) {
      if (ancestors.length >= 15) throw new Error("Snapshot ancestry exceeds the 16-manifest viewer limit");
      const source = requireObject(current.source, "source");
      if (!digestPattern.test(source.snapshot_id || "")) throw new Error("Source snapshot ID is not lowercase SHA-256");
      if (seen.has(source.snapshot_id)) throw new Error("Snapshot ancestry cycle detected");
      const loaded = await loadObject(source.manifest, "source manifest");
      verified = verified && loaded.verified;
      const parent = parseJsonStrict(decode(loaded.bytes, "source manifest"));
      if (parent.investigation_snapshot !== "v0-provisional") throw new Error("Unsupported source investigation snapshot format");
      const computed = await sha256(joinBytes(SNAPSHOT_DOMAIN, loaded.bytes));
      if (computed !== null && computed !== source.snapshot_id) throw new Error("Source snapshot identity mismatch");
      seen.add(source.snapshot_id);
      ancestors.push({ snapshotId: source.snapshot_id, manifest: parent });
      current = parent;
    }
    return { ancestors, verified };
  }

  async function renderModel(manifest) {
    const root = document.getElementById("model-summary"); clear(root);
    const evidence = requireArray(manifest.evidence, "evidence").find(item => {
      const linked = outputFor(manifest, item); return item.status === "current" && linked && linked.recipe.operation === "inspect";
    });
    if (!evidence) { root.append(el("p", "No current effective-model inspection is recorded.", "badge warn")); return false; }
    const linked = outputFor(manifest, evidence);
    const loaded = await loadObject(linked.output, "inspection report");
    const report = parseJsonStrict(decode(loaded.bytes, "inspection report"));
    if (report.inspection_format !== "v0-provisional") throw new Error("Unsupported inspection report");
    const slots = requireArray(report.free_slots, "inspection free_slots");
    const factors = requireArray(report.density_factors, "inspection density_factors");
    const metadata = requireObject(report.execution_metadata, "inspection execution_metadata");
    const grid = el("div", undefined, "grid");
    grid.append(facts([
      ["Free metadata", metadata.free_values],
      ["Factor metadata", metadata.stochastic_sites],
      ["Unconstrained coordinates", report.unconstrained_parameter_count],
      ["Transform Jacobians", requireObject(report.density_accounting, "density accounting").transform_jacobians],
    ]));
    grid.append(facts(requireArray(report.data, "inspection data").map(item => [
      `${item.role}: ${item.name}`, `[${(item.bound_shape || []).join(", ")}]${item.bound_integer ? " integer" : ""}`,
    ])));
    root.append(grid, el("h3", "Ordered free-state layout"));
    root.append(table(["Name", "Shape", "Offset", "Length", "Resolved transform"], slots.map(slot => [
      slot.name, `[${(slot.shape || []).join(", ")}]`, slot.offset, slot.length,
      prettyExpression(slot.resolved_constraint),
    ])));
    root.append(el("h3", "Actual log-density factors"));
    root.append(table(["#", "Name", "Distribution", "Value expression"], factors.map(factor => [
      factor.index, factor.name, prettyExpression(factor.distribution), prettyExpression(factor.value_expression),
    ])));
    const discrepancies = requireArray(report.structural_discrepancies, "inspection discrepancies");
    root.append(el("h3", "Declaration / execution differences"));
    root.append(discrepancies.length
      ? table(["Name", "Structural fact", "Note"], discrepancies.map(item => [item.name, item.kind, item.note || "—"]))
      : el("p", "No structural discrepancy was recorded for same-name declarations and factors."));
    const link = el("a", "Download raw inspection report"); link.href = referencePath(linked.output); link.download = "inspection.json"; root.append(link);
    return loaded.verified;
  }

  async function renderEvidence(manifest, ancestors) {
    const root = document.getElementById("evidence"); clear(root);
    const evidence = evidenceRows(manifest, ancestors);
    let allVerified = true;
    root.append(table(["Evidence", "Operation", "Status", "Origin", "Outcome"], evidence.map(row => [
      row.name, row.linked ? row.linked.recipe.operation : "unavailable", row.status,
      row.origin, row.linked ? row.linked.execution.outcome : "missing",
    ])));
    for (const row of evidence) {
      const linked = row.linked;
      if (!linked || !["diagnose", "posterior-check"].includes(linked.recipe.operation)) continue;
      const loaded = await loadObject(linked.output, `${linked.recipe.operation} output`); allVerified = allVerified && loaded.verified;
      const report = parseJsonStrict(decode(loaded.bytes, `${linked.recipe.operation} output`));
      const card = el("div", undefined, "card");
      card.append(el("h3", `${row.name} · ${row.status}`), el("p", row.origin, "small"));
      if (linked.recipe.operation === "diagnose") {
        card.append(facts([
          ["Source draws", report.source_draw_count], ["Chains", report.source_chain_count],
          ["R-hat definition", report.rhat_statistic], ["ESS definition", report.ess_statistic],
        ]));
        card.append(table(["Parameter", "R-hat value", "ESS value"], diagnosticRows(report).map(item => [
          item.parameter, recorded(item.rhat), recorded(item.ess),
        ])));
      } else {
        const checks = requireArray(report.checks, "posterior checks");
        card.append(table(["Site", "Statistic", "Observed", "Replicated mean", "Replicated range"], checks.map(check => {
          const summary = requireObject(check.summary, "check summary");
          return [check.site, check.statistic, summary.observed, summary.replicated_mean,
            `${text(summary.replicated_min)} … ${text(summary.replicated_max)}`];
        })));
        card.append(el("p", "These are factual discrepancy summaries, not a pass/fail verdict.", "small"));
      }
      root.append(card);
    }
    return allVerified;
  }

  function renderHistory(manifest) {
    const root = document.getElementById("history"); clear(root);
    root.append(facts([["Current interpretation", manifest.interpretation], ["Unresolved questions", (manifest.unresolved_questions || []).join(" · ")]]));
    const list = el("ol", undefined, "timeline");
    requireArray(manifest.decisions, "decisions").forEach(decision => {
      const item = el("li"); item.append(el("strong", decision.id), el("div", decision.reason),
        el("div", `${decision.kind}; parent: ${decision.parent || "none"}; cites ${decision.cites.length} artifact(s)`, "small")); list.append(item);
    }); root.append(list);
    root.append(el("h3", "Operation outcomes"));
    root.append(table(["Execution", "Recipe", "Outcome", "Retained error"], requireArray(manifest.executions, "executions").map(item => [
      item.id, item.recipe, item.outcome, item.error || "—",
    ])));
  }

  function renderContinuation(manifest, ancestors) {
    const root = document.getElementById("continuation"); clear(root);
    if (!manifest.source) {
      root.append(el("p", "This snapshot is an investigation root. A continuation can branch from any recorded decision."));
      return;
    }
    const source = requireObject(manifest.source, "source");
    root.append(facts([
      ["Source snapshot", `sha256:${source.snapshot_id}`],
      ["Branch decision", source.decision],
      ["Current model", `sha256:${manifest.inputs.model.sha256}`],
      ["Current data", `sha256:${manifest.inputs.data.sha256}`],
    ]));
    const comparison = evidenceRows(manifest, ancestors).map(row => [
      row.name, row.status, row.origin, row.linked ? row.linked.recipe.operation : "unavailable",
    ]);
    root.append(el("h3", "New evidence beside retained evidence"));
    root.append(table(["Evidence", "Status", "Snapshot", "Operation"], comparison));
    root.append(el("p", "Parent evidence is retained as historical lineage, not silently promoted to current evidence for changed inputs.", "small"));
  }

  function renderReproduce(manifest, entry) {
    const root = document.getElementById("reproduce"); clear(root);
    const sample = manifest.recipes.find(recipe => recipe.operation === "sample");
    const decision = manifest.decisions[0];
    const sampleId = sample ? safeIdentifier(sample.id, "sample recipe ID") : null;
    const decisionId = decision ? safeIdentifier(decision.id, "decision ID") : null;
    const commands = [
      "bayesite investigation verify bundle/",
      sampleId ? `bayesite investigation replay bundle/ --recipe ${sampleId} --out replay/` : "# No sample recipe is recorded for replay",
      decisionId ? `bayesite investigation fork bundle/ --at ${decisionId} --out alternative/` : "# No branch decision is recorded",
      "# Edit alternative/inputs/model.json and alternative/investigation.json",
      "bayesite investigation inspect alternative/",
      "# Run only the explicit alternative recipes you add, then:",
      "bayesite investigation snapshot alternative/ --out continuation/",
    ].join("\n");
    const engine = requireObject(entry.engine, "entry engine");
    root.append(facts([
      ["Pinned engine target", engine.target], ["Pinned engine profile", engine.profile],
      ["Pinned executable", `sha256:${engine.sha256}`], ["Bundle", "bundle/ (copy the directory exactly)"],
    ]));
    const download = el("a", "Download pinned engine for the published target"); download.href = "downloads/bayesite-engine"; download.download = "bayesite";
    const protocol = el("a", "Read the investigation-specific recipient protocol"); protocol.href = "PROTOCOL.md";
    const guide = el("a", "Read continuation instructions"); guide.href = "CONTINUING.md";
    const ir = el("a", "Read the raw Bayeswire format"); ir.href = "IR-FORMAT.md";
    const tags = el("a", "Read the node-tag reference"); tags.href = "IR-TAGS.md";
    root.append(download, el("span", " · "), protocol, el("span", " · "), guide,
      el("span", " · "), ir, el("span", " · "), tags, el("pre", commands));
    root.append(el("p", "Public-data confirmation records a publication choice; it is not a privacy scan, author signature, or scientific approval.", "small"));
  }

  async function start() {
    const integrity = document.getElementById("integrity");
    const entryBytes = await readLimited(await fetch("entry.json", { cache: "no-store" }), MAX_MANIFEST_BYTES, "entry metadata");
    const entry = parseJsonStrict(decode(entryBytes, "entry metadata"));
    if (entry.publication_format !== "v0-provisional") throw new Error("Unsupported publication format");
    const manifestBytes = await readLimited(await fetch("bundle/manifest.json", { cache: "no-store" }), MAX_MANIFEST_BYTES, "manifest");
    const manifest = parseJsonStrict(decode(manifestBytes, "manifest"));
    if (manifest.investigation_snapshot !== "v0-provisional") throw new Error("Unsupported investigation snapshot format");
    requireObject(manifest.inputs, "inputs"); requireArray(manifest.recipes, "recipes");
    requireArray(manifest.executions, "executions"); requireArray(manifest.evidence, "evidence");
    requireArray(manifest.decisions, "decisions").forEach((decision, index) => {
      safeIdentifier(decision.id, `decisions[${index}].id`);
      if (decision.parent !== null) safeIdentifier(decision.parent, `decisions[${index}].parent`);
    });
    manifest.recipes.forEach((recipe, index) => safeIdentifier(recipe.id, `recipes[${index}].id`));
    manifest.executions.forEach((execution, index) => {
      safeIdentifier(execution.id, `executions[${index}].id`);
      safeIdentifier(execution.recipe, `executions[${index}].recipe`);
    });
    const digest = await sha256(joinBytes(SNAPSHOT_DOMAIN, manifestBytes));
    const computed = digest === null ? null : `sha256:${digest}`;
    const supplied = new URLSearchParams(location.search).get("snapshot") || entry.snapshot_id;
    if (computed !== null && supplied !== computed) throw new Error(`Snapshot identity mismatch: computed ${computed}, expected ${supplied}`);
    const rootSnapshotId = (computed || supplied || "").replace(/^sha256:/, "");
    if (!digestPattern.test(rootSnapshotId)) throw new Error("Expected snapshot ID is not lowercase SHA-256");
    const ancestry = await loadAncestry(manifest, rootSnapshotId);

    document.getElementById("question").textContent = text(manifest.question);
    document.getElementById("estimand").textContent = text(requireObject(manifest.estimand, "estimand").description);
    document.getElementById("snapshot-id").textContent = computed || `supplied ${supplied}`;
    const modelVerified = await renderModel(manifest);
    const evidenceVerified = await renderEvidence(manifest, ancestry.ancestors);
    renderHistory(manifest); renderContinuation(manifest, ancestry.ancestors); renderReproduce(manifest, entry);
    if (computed !== null && modelVerified && evidenceVerified && ancestry.verified) {
      integrity.textContent = "Manifest identity + displayed objects verified";
      integrity.className = "badge good";
    } else {
      integrity.textContent = "Displayed without browser cryptographic verification — use CLI verify";
      integrity.className = "badge warn";
    }
    integrity.title = "This is partial viewer coverage, not full bundle verification or author authentication.";
    document.getElementById("content").hidden = false;
  }

  globalThis.BAYESITE_VIEWER_TEST = {
    parseJsonStrict, referencePath, safeIdentifier, diagnosticRows, evidenceRows,
  };
  if (typeof document !== "undefined") {
    start().catch(error => {
      const status = document.getElementById("integrity"); status.textContent = "Integrity/display failure"; status.className = "badge bad";
      const node = document.getElementById("error"); node.textContent = error instanceof Error ? error.message : String(error); node.hidden = false;
    });
  }
})();
