const proposalsElement = document.querySelector("#proposals");
const recordElement = document.querySelector("#record");

function status(review, attempts) {
  const last = attempts.at(-1);
  if (last?.outcome === "completed") return "executed";
  if (last?.outcome === "failed") return "failed";
  if (last) return "incomplete";
  if (review?.decision === "approved") return "approved";
  if (review?.decision === "rejected") return "rejected";
  return "pending";
}

async function optionalJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return await response.json();
}

async function loadRecord(entry) {
  const base = `proposals/${entry.proposal_id}`;
  const proposal = await optionalJson(`${base}/proposal.json`);
  const review = await optionalJson(`${base}/review.json`);
  const attempts = [];
  for (let number = 1; number <= entry.attempt_count; number += 1) {
    const attempt = await optionalJson(`${base}/attempts/${number}.json`);
    if (attempt) attempts.push(attempt);
  }
  return {
    proposal_id: entry.proposal_id,
    status: status(review, attempts),
    action: proposal?.action ?? null,
    rationale: proposal?.rationale ?? null,
    cites: proposal?.cites ?? null,
    preconditions: proposal?.preconditions ?? null,
    review,
    attempts,
  };
}

async function show(entry) {
  recordElement.textContent = JSON.stringify(await loadRecord(entry), null, 2);
}

try {
  const index = await optionalJson("index.json");
  proposalsElement.textContent = "";
  for (const entry of index?.proposals ?? []) {
    const record = await loadRecord(entry);
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${entry.proposal_id} — ${record.status}`;
    button.addEventListener("click", () => void show(entry).catch(reportError));
    proposalsElement.append(button);
  }
  if (!(index?.proposals?.length > 0)) proposalsElement.textContent = "No proposals recorded.";
} catch (error) {
  reportError(error);
}

function reportError(error) {
  recordElement.textContent = `Could not load proposal records: ${error instanceof Error ? error.message : String(error)}`;
}
