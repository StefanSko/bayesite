import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
require("../demo/investigation/viewer.js");
const viewer = globalThis.BAYESITE_VIEWER_TEST;

const parsed = viewer.parseJsonStrict('{"reason":"<script>alert(1)</script>"}');
assert.equal(parsed.reason, "<script>alert(1)</script>");
for (const malformed of ['{"x":1,"x":2}', '{"x":[1,2,]}', '{"x":NaN}']) {
  assert.throws(() => viewer.parseJsonStrict(malformed));
}
for (const active of ["$(touch PWNED)", "x; rm -rf /", "name with space", "`id`"]) {
  assert.throws(() => viewer.safeIdentifier(active, "test ID"));
}

const diagnostics = JSON.parse(
  readFileSync(new URL("../examples/investigation-counts/evidence/diagnostics.json", import.meta.url), "utf8"),
);
const diagnosticRows = viewer.diagnosticRows(diagnostics);
const meanRow = diagnosticRows.find(row => row.parameter === "mean_daily_count");
assert.ok(meanRow, "numerical diagnostics include mean_daily_count");
assert.equal(typeof meanRow.rhat, "number");
assert.equal(typeof meanRow.ess, "number");
assert.equal(meanRow.rhat, diagnostics.rhat.mean_daily_count);
assert.equal(meanRow.ess, diagnostics.ess.mean_daily_count);

const current = {
  recipes: [{ id: "new-check", operation: "posterior-check" }],
  executions: [{ id: "new-exec", recipe: "new-check", outcome: "completed", output: { sha256: "a".repeat(64), bytes: 1 } }],
  evidence: [{ name: "new evidence", execution: "new-exec", status: "current" }],
};
const parent = {
  recipes: [{ id: "old-check", operation: "posterior-check" }],
  executions: [{ id: "old-exec", recipe: "old-check", outcome: "completed", output: { sha256: "b".repeat(64), bytes: 1 } }],
  evidence: [{ name: "old evidence", execution: "old-exec", status: "current" }],
};
const rows = viewer.evidenceRows(current, [{ snapshotId: "c".repeat(64), manifest: parent }]);
assert.deepEqual(
  rows.map(row => [row.name, row.status, row.origin]),
  [
    ["new evidence", "current", "current snapshot"],
    ["old evidence", "historical", `source sha256:${"c".repeat(64)}`],
  ],
);

console.log("investigation viewer parsing, diagnostics, and ancestry checks passed");
