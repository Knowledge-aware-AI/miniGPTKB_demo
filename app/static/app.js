// ---- Shared (single-instance) state and DOM refs ----
const runsBody = document.querySelector("#runs-body");
const refreshRunsButton = document.querySelector("#refresh-runs");
const runTabsContainer = document.querySelector("#run-tabs");
const runTabPanesContainer = document.querySelector("#run-tab-panes");
const addTabButton = document.querySelector("#add-tab-button");
const runTabTemplate = document.querySelector("#run-tab-template");

const SCADS_ENDPOINT = "https://llm.scads.ai/v1";
// NER/NED models on ScaDS.AI labelled "recommended" in the model pickers
// (Oct 2026): Qwen3.8-27B made no NED errors on decisions from test runs and
// has a high rate limit; gemma-4 is accurate on NER, fast, and has its own
// rate-limit quota. There is deliberately no default main model: that choice
// is left to the user.
const SCADS_RECOMMENDED_MODELS = {
  ner_model: "google/gemma-4-26B-A4B-it",
  ned_model: "Qwen/Qwen3.8-27B",
};
const KNOWN_ENDPOINTS = [
  SCADS_ENDPOINT,
  "https://api.openai.com/v1",
  "https://openrouter.ai/api/v1",
];

// Elicitation instruction texts taken from the prompt library of the Beyond
// Questions elicitation experiments -- there they are full OpenAI Batch API
// request templates for a different pipeline, so only their system-message
// content (the actual instructions) is reused here; picking
// one just loads its text into the Elicitation Prompt textarea, which already
// fully overrides the auto-generated prompt when non-empty.
const PROMPT_TEMPLATES = [
  {
    id: "GPTKB_2x_repetition",
    label: "GPTKB, 2x repetition",
    text: `I want to construct a knowledge graph on the topic of the ancient city of Babylon. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important: 
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list.
- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.
- If the subject is a named entity, include at least one triple where predicate is "instanceOf".
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.
- Don't generate any other text except for the triples in JSON format.
Let me repeat that:
I want to construct a knowledge graph on the topic of the ancient city of Babylon. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important: 
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list.
- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.
- If the subject is a named entity, include at least one triple where predicate is "instanceOf".
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.
- Don't generate any other text except for the triples in JSON format.`,
  },
  {
    id: "GPTKB_3x_repetition",
    label: "GPTKB, 3x repetition",
    text: `I want to construct a knowledge graph on the topic of the ancient city of Babylon. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important: 
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list.
- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.
- If the subject is a named entity, include at least one triple where predicate is "instanceOf".
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.
- Don't generate any other text except for the triples in JSON format.
Let me repeat that:
I want to construct a knowledge graph on the topic of the ancient city of Babylon. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important: 
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list.
- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.
- If the subject is a named entity, include at least one triple where predicate is "instanceOf".
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.
- Don't generate any other text except for the triples in JSON format.
Let me repeat that one more time:
I want to construct a knowledge graph on the topic of the ancient city of Babylon. Given a subject entity, return all facts that you know for the subject as a list of subject, predicate, object triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important: 
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list.
- If the subject does not belong to the topic of the ancient city of Babylon, return an empty list.
- If the subject is a named entity, include at least one triple where predicate is "instanceOf".
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Format the output as a structured JSON, a list of dictionaries with 3 keys "s", "p", and "o" each, and the respective values.
- Don't generate any other text except for the triples in JSON format.`,
  },
  {
    id: "schemaorg_schema",
    label: "Schema.org (fixed predicate list)",
    text: `You are a knowledge base construction expert. Given a subject entity, return all facts that you know for the subject as a list of (subject, predicate, object) triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important:
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Predicates MUST be chosen only from the allowed predicate list below.
- Choose only the predicates relevant for the subject entity.
- Do not invent new predicates.
Allowed predicates:
instanceOf
name
givenName
familyName
alternateName
birthDate
deathDate
gender
jobTitle
memberOf
affiliation
nationality
address
location
founder
parentOrganization
startDate
endDate
dateCreated
datePublished
organizer
author
creator
worksFor
genre`,
  },
  {
    id: "schemaorg_schema_no_constraints",
    label: "Schema.org (unconstrained)",
    text: `You are a knowledge base construction expert. Given a subject entity, return all facts that you know for the subject as a list of (subject, predicate, object) triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important:
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Predicates MUST be chosen only from the schema.org schema.
- Choose only the predicates relevant for the subject entity.
- Do not invent new predicates.`,
  },
  {
    id: "wikidata_schema",
    label: "Wikidata (fixed predicate list)",
    text: `You are a knowledge base construction expert. Given a subject entity, return all facts that you know for the subject as a list of (subject, predicate, object) triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important:
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Predicates MUST be chosen only from the allowed predicate list below.
- Choose only the predicates relevant for the subject entity.
- Do not invent new predicates.
 Allowed predicates:
instance of
subclass of
part of
has part(s)
title
short name
named after
inception
start time
end time
point in time
publication date
country
country of citizenship
located in the administrative territorial entity
location
coordinate location
place of birth
place of death
headquarters location
occupation
educated at
award received
member of
employer
languages spoken, written or signed
sex or gender
date of birth
date of death
founded by
parent organization or unit
child organization or unit
author
composer
director
publisher
genre
based on
follows
followed by
participant
organizer
parent taxon
taxon rank
found in taxon
population
number of pages
country of origin
given name
family name`,
  },
  {
    id: "wikidata_schema_no_constraints",
    label: "Wikidata (unconstrained)",
    text: `You are a knowledge base construction expert. Given a subject entity, return all facts that you know for the subject as a list of (subject, predicate, object) triples. The number of facts may be very high, between 50 to 100 or more, for very popular subjects. For less popular subjects, the number of facts can be very low, like 5 or 10.

Important:
- If you don't know the subject, return an empty list.
- If the subject is not a named entity, return an empty list
- Do not get too wordy.
- Separate several objects into multiple triples with one object.
- Predicates MUST be chosen only from the Wikidata schema.
- Choose only the predicates relevant for the subject entity.
- Do not invent new predicates.`,
  },
];

const TABS_STORAGE_KEY = "miniGPTKB.openTabs";

const tabs = new Map(); // tabId -> tab controller
let tabOrder = [];
let activeTabId = null;
let tabIdCounter = 0;
let cachedRunsList = [];
let cachedFoldersList = [];
let languagesPromise = null;

async function loadFolders() {
  try {
    const data = await fetch("/api/folders").then((response) => response.json());
    cachedFoldersList = data.folders || [];
  } catch (error) {
    // Leave the previous cache in place if the endpoint is briefly unreachable.
  }
  return cachedFoldersList;
}

function refreshFolderSelectAllTabs() {
  tabs.forEach((tab) => tab.refreshFolderOptions && tab.refreshFolderOptions());
}

function getLanguages() {
  if (!languagesPromise) {
    languagesPromise = fetch("/api/languages")
      .then((response) => response.json())
      .then((data) => data.languages || [])
      .catch(() => []);
  }
  return languagesPromise;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = Math.floor(total % 60);
  if (hours) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function formatCost(cost) {
  if (cost === null || cost === undefined || cost === "") return "";
  const value = Number(cost);
  if (!Number.isFinite(value)) return "";
  if (value > 0 && value < 0.01) return "<$0.01";
  return `$${value.toFixed(2)}`;
}

function shortDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text ?? "";
  return td;
}

// Stops of the Entity merging slider: the minimum label similarity a
// candidate needs to be offered to the NED model. 0 = Standard (no floor, as
// in GPTKB 2.0);
// the useful range starts around 0.8, since an entity's top-5 neighbours are
// rarely less similar than that.
const MERGE_SIMILARITY_FLOORS = [0, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96];

function mergeSliderLabel(index) {
  const floor = MERGE_SIMILARITY_FLOORS[index] || 0;
  return floor ? `${Math.round(floor * 100)}% similar` : "Standard";
}

function sliderName(value) {
  if (value <= 0.5) return "Few";
  if (value >= 1.5) return "Many";
  return "Medium";
}

function normalizeUrl(value) {
  return value.trim().replace(/\/+$/, "");
}

async function postJSON(url, body = null) {
  const response = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : null,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

async function deleteJSON(url) {
  const response = await fetch(url, { method: "DELETE" });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

async function patchJSON(url, body) {
  const response = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

// ---- Run tabs: strip rendering, switching, persistence ----

function labelForTab(tab) {
  if (tab.pendingLabel) return tab.pendingLabel;
  if (!tab.currentRunId) return "New Run";
  const info = cachedRunsList.find((run) => run.run_id === tab.currentRunId);
  if (info && (info.topic || info.seed_entity)) return info.topic || info.seed_entity;
  return tab.currentRunId.slice(0, 8);
}

function renderTabStrip() {
  runTabsContainer.innerHTML = "";
  tabOrder.forEach((tabId) => {
    const tab = tabs.get(tabId);
    if (!tab) return;
    const chip = document.createElement("div");
    const statusClass = ["running", "paused"].includes(tab.getStatus()) ? ` ${tab.getStatus()}` : "";
    chip.className = "run-tab-chip" + (tabId === activeTabId ? " active" : "") + statusClass;
    chip.dataset.tabId = tabId;

    const labelButton = document.createElement("button");
    labelButton.type = "button";
    labelButton.className = "run-tab-label";
    labelButton.textContent = labelForTab(tab);
    labelButton.addEventListener("click", () => activateTab(tabId));
    chip.appendChild(labelButton);

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "run-tab-close";
    closeButton.textContent = "×";
    closeButton.title = "Close tab";
    closeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      closeTab(tabId);
    });
    chip.appendChild(closeButton);

    runTabsContainer.appendChild(chip);
  });
}

function activateTab(tabId) {
  if (!tabs.has(tabId)) return;
  activeTabId = tabId;
  tabOrder.forEach((id) => {
    const tab = tabs.get(id);
    if (tab) tab.root.classList.toggle("active", id === tabId);
  });
  renderTabStrip();
  persistTabs();
}

function closeTab(tabId) {
  const tab = tabs.get(tabId);
  if (!tab) return;
  if (tab.isActiveRun()) {
    const confirmed = window.confirm(
      "This run is still active and will keep running on the server -- closing the tab just stops watching it here. Close anyway?"
    );
    if (!confirmed) return;
  }
  tab.destroy();
  tab.root.remove();
  tabs.delete(tabId);
  tabOrder = tabOrder.filter((id) => id !== tabId);

  if (activeTabId === tabId) {
    activeTabId = null;
    const next = tabOrder[tabOrder.length - 1];
    if (next) {
      activateTab(next);
    } else {
      createTab();
      return; // createTab() already activates and persists.
    }
  }
  renderTabStrip();
  persistTabs();
}

function persistTabs() {
  try {
    const data = tabOrder.map((tabId) => ({ runId: tabs.get(tabId)?.currentRunId || null }));
    localStorage.setItem(
      TABS_STORAGE_KEY,
      JSON.stringify({ tabs: data, activeIndex: tabOrder.indexOf(activeTabId) })
    );
  } catch (error) {
    // Storage can be unavailable (private browsing, quota) -- tabs just won't survive a reload.
  }
}

function loadPersistedTabs() {
  try {
    const raw = localStorage.getItem(TABS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.tabs) || !parsed.tabs.length) return null;
    return parsed;
  } catch (error) {
    return null;
  }
}

// Opens (or focuses, if already open) a tab watching a specific run_id --
// used by the Runs table's Watch button.
function openRunInTab(runId) {
  const existing = tabOrder.find((tabId) => tabs.get(tabId)?.currentRunId === runId);
  if (existing) {
    activateTab(existing);
    return;
  }
  createTab({ runId, activate: true });
}

// ---- One run tab: its own form, status panel, log, and live state ----

function createTab({ runId = null, isNewRun = false, activate = true } = {}) {
  const tabId = `tab-${++tabIdCounter}`;
  const fragment = runTabTemplate.content.cloneNode(true);
  const root = fragment.firstElementChild;
  runTabPanesContainer.appendChild(root);

  const tab = buildTabController(tabId, root);
  tabs.set(tabId, tab);
  tabOrder.push(tabId);

  if (activate) {
    activateTab(tabId);
  } else {
    renderTabStrip();
    persistTabs();
  }

  if (runId) {
    tab.watchRun(runId, { isNewRun });
  }

  return tab;
}

function buildTabController(tabId, root) {
  const form = root.querySelector('[data-role="run-form"]');
  const logOutput = root.querySelector('[data-role="log-output"]');
  const logFilter = root.querySelector('[data-role="log-filter"]');
  const statusText = root.querySelector('[data-role="status-text"]');
  const currentSubjectEl = root.querySelector('[data-role="current-subject"]');
  const factsSlider = root.querySelector('[data-role="facts-slider"]');
  const tripleLabel = root.querySelector('[data-role="triple-label"]');
  const mergeSlider = root.querySelector('[data-role="merge-slider"]');
  const mergeLabel = root.querySelector('[data-role="merge-label"]');
  const schemaDetails = root.querySelector('[data-role="schema-details"]');
  const schemaFile = root.querySelector('[data-role="schema-file"]');
  const endpointSelect = root.querySelector('[data-role="endpoint-select"]');
  const llmUrlLabel = root.querySelector('[data-role="llm-url-label"]');
  const llmUrl = root.querySelector('[data-role="llm-url"]');
  const apiKeyInput = root.querySelector('input[name="api_key"]');
  const scadsKeyNote = root.querySelector('[data-role="scads-key-note"]');
  const modelSelect = root.querySelector('[data-role="model-select"]');
  const modelInput = root.querySelector('[data-role="model-input"]');
  // The elicitation model comes from the chosen endpoint (a free-text input
  // for a custom endpoint); reasoning controls follow it. NER and NED can use
  // ScaDS.AI models -- always available through the shared key -- as well as
  // the elicitation endpoint's own models, so their options carry the
  // endpoint along with the model name.
  // With a custom elicitation endpoint, NER/NED can also use a model typed
  // in for that endpoint (the CUSTOM_ROLE_MODEL option shows a text input).
  const CUSTOM_ROLE_MODEL = "__custom__";
  const rolePickers = [
    {
      select: root.querySelector('[data-role="ner-model-select"]'),
      input: root.querySelector('[data-role="ner-model-custom"]'),
      name: "ner",
      userChose: false,
    },
    {
      select: root.querySelector('[data-role="ned-model-select"]'),
      input: root.querySelector('[data-role="ned-model-custom"]'),
      name: "ned",
      userChose: false,
    },
  ];
  let scadsModels = [];
  let endpointModels = []; // the elicitation endpoint's models, when it isn't ScaDS.AI
  const reasoningToggleField = root.querySelector('[data-role="reasoning-toggle-field"]');
  const reasoningToggleInput = root.querySelector('[data-role="reasoning-toggle-input"]');
  const reasoningToggleLabel = root.querySelector('[data-role="reasoning-toggle-label"]');
  const reasoningEffortField = root.querySelector('[data-role="reasoning-effort-field"]');
  const reasoningEffortSlider = root.querySelector('[data-role="reasoning-effort-slider"]');
  const reasoningEffortLabel = root.querySelector('[data-role="reasoning-effort-label"]');
  const reasoningEffortScale = root.querySelector('[data-role="reasoning-effort-scale"]');
  const languageSelect = root.querySelector('[data-role="language-select"]');
  const topicInput = root.querySelector('input[name="topic"]');
  const folderSelect = root.querySelector('[data-role="folder-select"]');
  const newFolderNameInput = root.querySelector('[data-role="new-folder-name"]');
  // The new-folder name field sits under the folder picker and is shown only for "+ New folder…".
  const newFolderNameLabel = newFolderNameInput;
  const schemaTextArea = root.querySelector('textarea[name="schema_text"]');
  const promptPreview = root.querySelector('[data-role="prompt-preview"]');
  const promptTemplateSelect = root.querySelector('[data-role="prompt-template-select"]');
  const htmlIndexLink = root.querySelector('[data-role="html-index-link"]');
  const elapsedTimeDisplay = root.querySelector('[data-role="elapsed-time"]');
  const startButton = root.querySelector('[data-role="start-button"]');
  const pauseButton = root.querySelector('[data-role="pause-button"]');
  const stopButton = root.querySelector('[data-role="stop-button"]');
  const statusDotEl = root.querySelector('[data-role="status-dot"]');

  const logs = [];
  // Matches the server's own MAX_LOG_EVENTS retention cap -- without this,
  // a very long run's log array grows unboundedly in this tab's memory, and
  // renderLogs()'s filter pass over the whole array gets slower with every
  // single new line for the rest of the run.
  const MAX_CLIENT_LOG_EVENTS = 20000;
  // Lines kept in the log panel (the full history up to the cap above stays
  // in memory and comes back when the filter changes).
  const MAX_VISIBLE_LOG_LINES = 5000;
  // Event ids already shown: a reconnecting event stream or a snapshot can
  // deliver the same event twice.
  const seenLogIds = new Set();
  function pushLog(event) {
    if (event.id !== undefined && event.id !== null) {
      if (seenLogIds.has(event.id)) return false;
      seenLogIds.add(event.id);
    }
    logs.push(event);
    if (logs.length > MAX_CLIENT_LOG_EVENTS) {
      logs.splice(0, logs.length - MAX_CLIENT_LOG_EVENTS);
    }
    return true;
  }
  function resetLogs() {
    logs.length = 0;
    seenLogIds.clear();
  }
  let paused = false;
  let modelsRequestToken = 0;
  let modelsDebounceTimer = null;
  let promptPreviewDebounceTimer = null;
  let currentRunId = null;
  let eventSource = null;
  let lastKnownStatus = null;
  // Elapsed time is wall-clock-since-`started_at`, a timestamp the server
  // persists to runConfig.json at launch (see /api/run and /status) --  not
  // a client-side stopwatch. That means it reads correctly the instant a
  // tab opens or reopens on a run already in progress, and it survives the
  // browser/tab being closed and reopened later, since it's derived from
  // data the server already keeps regardless of who's watching.
  let runStartedAtMs = null;
  let elapsedTimerHandle = null;

  const tab = { tabId, root, pendingLabel: null };
  Object.defineProperty(tab, "currentRunId", { get: () => currentRunId });

  function tickElapsed() {
    if (runStartedAtMs === null) {
      elapsedTimeDisplay.textContent = "-";
      return;
    }
    elapsedTimeDisplay.textContent = formatDuration(Math.floor((Date.now() - runStartedAtMs) / 1000));
  }

  function clearElapsedInterval() {
    if (elapsedTimerHandle) {
      clearInterval(elapsedTimerHandle);
      elapsedTimerHandle = null;
    }
  }

  function startElapsedTicking(startedAt) {
    const parsed = startedAt ? new Date(startedAt).getTime() : NaN;
    runStartedAtMs = Number.isNaN(parsed) ? Date.now() : parsed;
    clearElapsedInterval();
    tickElapsed();
    elapsedTimerHandle = setInterval(tickElapsed, 1000);
  }

  function stopElapsedTicking() {
    clearElapsedInterval();
    tickElapsed();
  }

  function optionalNumber(formData, enabledName, valueName) {
    if (!formData.has(enabledName)) return null;
    const raw = formData.get(valueName);
    return raw === "" ? null : Number(raw);
  }

  function payloadFromForm() {
    const data = new FormData(form);
    const isNewFolder = folderSelect.value === "__new__";
    return {
      llm_url: data.get("llm_url"),
      api_key: data.get("api_key"),
      model: data.get("model"),
      ...rolePayload(rolePickers[0]),
      ...rolePayload(rolePickers[1]),
      topic: data.get("topic"),
      seed_entity: data.get("seed_entity"),
      language: data.get("language") || "en",
      schema_mode: data.get("schema_mode"),
      schema_text: data.get("schema_text") || "",
      prompt_template: data.get("prompt_template") || "",
      facts_multiplier: Number(data.get("facts_multiplier")),
      ned_similarity_floor: MERGE_SIMILARITY_FLOORS[Number(mergeSlider.value)] || 0,
      ned_candidates: Number(data.get("ned_candidates")) || 5,
      max_entities: optionalNumber(data, "use_entities", "max_entities"),
      max_bfs_depth: optionalNumber(data, "use_bfs", "max_bfs_depth"),
      max_runtime_seconds: optionalNumber(data, "use_runtime", "max_runtime_seconds"),
      no_time_limit: data.has("no_time_limit"),
      max_cost: optionalNumber(data, "use_cost", "max_cost"),
      nthreads: Number(data.get("nthreads")),
      max_requests_per_minute: data.get("max_requests_per_minute") === "" ? null : Number(data.get("max_requests_per_minute")),
      folder_id: isNewFolder ? null : folderSelect.value || null,
      new_folder_name: isNewFolder ? newFolderNameInput.value.trim() : null,
      reasoning_effort: reasoningEffortField.classList.contains("hidden")
        ? null
        : currentReasoningLevels[Number(reasoningEffortSlider.value)] || null,
      reasoning_enabled: reasoningToggleField.classList.contains("hidden") ? null : reasoningToggleInput.checked,
    };
  }

  function setStatus(status) {
    statusText.textContent = status ? status[0].toUpperCase() + status.slice(1) : "Idle";
    statusDotEl.className = "status-dot" + (status ? ` ${status}` : "");
  }

  function updateProgress(status) {
    lastKnownStatus = status.status;
    setStatus(status.status);
    root.querySelector('[data-role="processed"]').textContent = Number(status.processed || 0);
    root.querySelector('[data-role="queued"]').textContent = Number(status.queued || 0);
    root.querySelector('[data-role="triples"]').textContent = status.triples || 0;
    root.querySelector('[data-role="failed-subjects"]').textContent = status.failed_subjects || 0;
    root.querySelector('[data-role="literals"]').textContent = status.literals || 0;
    root.querySelector('[data-role="live-cost"]').textContent =
      status.cost_usd === null || status.cost_usd === undefined ? "-" : formatCost(status.cost_usd) || "$0.00";
  }

  function logLine(entry) {
    const line = document.createElement("div");
    line.className = entry.level === "error" ? "log-error" : entry.level === "warning" ? "log-warning" : "";
    line.textContent = `${entry.time || "--:--:--"} [${entry.level}] ${entry.message}`;
    if (entry.url) {
      line.append(" ");
      const link = document.createElement("a");
      link.href = entry.url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = entry.url;
      line.appendChild(link);
    }
    return line;
  }

  // Follow new lines only while the reader is at the bottom; scrolled up,
  // the view stays where it is.
  function logAtBottom() {
    return logOutput.scrollHeight - logOutput.scrollTop - logOutput.clientHeight < 40;
  }

  function renderLogs() {
    const filter = logFilter.value;
    const fragment = document.createDocumentFragment();
    logs
      .filter((entry) => filter === "all" || entry.level === filter)
      .slice(-MAX_VISIBLE_LOG_LINES)
      .forEach((entry) => fragment.appendChild(logLine(entry)));
    logOutput.replaceChildren(fragment);
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  // Adds one live line without rebuilding the panel.
  function appendLog(event) {
    if (!pushLog(event)) return;
    const filter = logFilter.value;
    if (filter !== "all" && event.level !== filter) return;
    const follow = logAtBottom();
    logOutput.appendChild(logLine(event));
    while (logOutput.childElementCount > MAX_VISIBLE_LOG_LINES) logOutput.firstElementChild.remove();
    if (follow) logOutput.scrollTop = logOutput.scrollHeight;
  }

  function setControlsEnabled(enabled) {
    pauseButton.disabled = !enabled;
    stopButton.disabled = !enabled;
  }

  function setHtmlIndexLink(url) {
    if (!url) return;
    htmlIndexLink.href = url;
    htmlIndexLink.classList.remove("hidden");
  }

  function connectEvents(runId) {
    if (eventSource) {
      eventSource.close();
    }
    eventSource = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    eventSource.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "log") {
        appendLog(event);
      } else if (event.type === "status") {
        updateProgress(event);
        loadRunsDebounced();
        if (event.status === "running") {
          paused = false;
          pauseButton.textContent = "Pause";
          setControlsEnabled(true);
        } else if (event.status === "paused") {
          paused = true;
          pauseButton.textContent = "Resume";
          setControlsEnabled(true);
        } else if (["stopped", "finished", "failed"].includes(event.status)) {
          stopElapsedTicking();
          setControlsEnabled(false);
        }
      } else if (event.type === "current_subject") {
        currentSubjectEl.textContent = `Processing ${event.subject}`;
      }
    };
  }

  // Switches this tab to a specific run_id. `isNewRun` is true only right
  // after this tab itself started/resumed it (so the elapsed timer counts
  // from a known zero); otherwise (restored from storage, or opened via
  // Watch) an existing snapshot is fetched first so the panel doesn't sit
  // empty until the next live event happens to fire.
  async function watchRun(runId, { isNewRun = false, startedAt = null } = {}) {
    currentRunId = runId;
    tab.pendingLabel = null;
    resetLogs();
    currentSubjectEl.textContent = "No active subject";
    paused = false;
    pauseButton.textContent = "Pause";
    renderTabStrip();
    persistTabs();

    if (isNewRun) {
      startElapsedTicking(startedAt);
      setControlsEnabled(true);
      renderLogs();
    } else {
      runStartedAtMs = null;
      tickElapsed();
      setControlsEnabled(false);
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/status`);
        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? "No live data for this run (the server may have restarted since it last ran)."
              : `Could not load run status (${response.status}).`
          );
        }
        const snapshot = await response.json();
        const active = ["running", "paused", "starting"].includes(snapshot.status);
        if (snapshot.latest) {
          updateProgress(snapshot.latest);
          setControlsEnabled(active);
          if (snapshot.status === "paused") {
            paused = true;
            pauseButton.textContent = "Resume";
          }
        }
        if (active) {
          startElapsedTicking(snapshot.started_at);
        } else if (snapshot.started_at) {
          const startedMs = new Date(snapshot.started_at).getTime();
          const endedMs = snapshot.ended_at ? new Date(snapshot.ended_at).getTime() : NaN;
          if (!Number.isNaN(startedMs)) {
            const seconds = Math.max(0, Math.floor(((Number.isNaN(endedMs) ? Date.now() : endedMs) - startedMs) / 1000));
            elapsedTimeDisplay.textContent = formatDuration(seconds);
          }
        }
        renderLogs();
      } catch (error) {
        pushLog({ level: "error", time: new Date().toLocaleTimeString(), message: error.message });
        renderLogs();
        return;
      }
    }

    connectEvents(runId);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const payload = payloadFromForm();
      const response = await postJSON("/api/run", payload);
      tab.pendingLabel = payload.topic || payload.seed_entity || null;
      setHtmlIndexLink(response.html_index_url);
      await watchRun(response.run_id, { isNewRun: true, startedAt: response.started_at });
      loadRuns();
      if (payload.new_folder_name) {
        await loadFolders();
        refreshFolderSelectAllTabs();
      }
    } catch (error) {
      pushLog({ level: "error", time: new Date().toLocaleTimeString(), message: error.message });
      renderLogs();
    }
  });

  // The Start button now lives in the hero strip, outside <form>, so it can't
  // rely on being a native type="submit" nested inside it -- requestSubmit()
  // still runs the form's own required-field validation and fires the same
  // "submit" event the handler above listens for.
  startButton.addEventListener("click", () => form.requestSubmit());

  pauseButton.addEventListener("click", async () => {
    if (!currentRunId) return;
    try {
      if (paused) {
        await postJSON(`/api/runs/${encodeURIComponent(currentRunId)}/unpause`);
      } else {
        await postJSON(`/api/runs/${encodeURIComponent(currentRunId)}/pause`);
      }
    } catch (error) {
      pushLog({ level: "error", time: new Date().toLocaleTimeString(), message: error.message });
      renderLogs();
    }
  });

  stopButton.addEventListener("click", async () => {
    if (!currentRunId) return;
    try {
      await postJSON(`/api/runs/${encodeURIComponent(currentRunId)}/stop`);
    } catch (error) {
      pushLog({ level: "error", time: new Date().toLocaleTimeString(), message: error.message });
      renderLogs();
    }
  });

  factsSlider.addEventListener("input", () => {
    tripleLabel.textContent = sliderName(Number(factsSlider.value));
  });

  mergeSlider.addEventListener("input", () => {
    mergeLabel.textContent = mergeSliderLabel(Number(mergeSlider.value));
  });

  function syncEndpointSelectionFromUrl() {
    const normalized = normalizeUrl(llmUrl.value);
    endpointSelect.value = KNOWN_ENDPOINTS.includes(normalized) ? normalized : "custom";
  }

  function setModelOptions(models) {
    const previousValue = modelInput.value || modelSelect.value;
    modelSelect.innerHTML = "";
    // No default elicitation model: the empty first option keeps the
    // (required) select invalid until the user picks one.
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "Choose a model…";
    modelSelect.appendChild(blank);
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    });
    const nextValue = models.includes(previousValue) ? previousValue : "";
    modelSelect.value = nextValue;
    modelInput.value = nextValue;
    if (normalizeUrl(llmUrl.value) === SCADS_ENDPOINT) {
      scadsModels = models;
      endpointModels = [];
    } else {
      endpointModels = models;
    }
    renderRolePickers();
    refreshReasoningCapability();
  }

  function setModelPlaceholder(message) {
    modelSelect.innerHTML = "";
    const option = document.createElement("option");
    option.value = "";
    option.textContent = message;
    modelSelect.appendChild(option);
    modelInput.value = "";
    endpointModels = [];
    renderRolePickers();
    hideReasoningControls();
  }

  function endpointName(url) {
    if (url === SCADS_ENDPOINT) return "ScaDS.AI";
    if (url === "https://api.openai.com/v1") return "OpenAI";
    if (url === "https://openrouter.ai/api/v1") return "OpenRouter";
    return url.replace(/^https?:\/\//, "").split("/")[0];
  }

  function roleOptionValue(url, model) {
    return JSON.stringify([url, model]);
  }

  function rolePayload(picker) {
    if (!picker.select.value) return { [`${picker.name}_model`]: "", [`${picker.name}_llm_url`]: "" };
    if (picker.select.value === CUSTOM_ROLE_MODEL) {
      return { [`${picker.name}_model`]: picker.input.value.trim(), [`${picker.name}_llm_url`]: normalizeUrl(llmUrl.value) };
    }
    const [url, model] = JSON.parse(picker.select.value);
    return { [`${picker.name}_model`]: model, [`${picker.name}_llm_url`]: url };
  }

  // NER/NED options (under Advanced): "same as main model", then the main
  // endpoint's models (if it isn't ScaDS.AI and its models are loaded) -- or,
  // for a custom endpoint, a typed-in model on it -- then ScaDS.AI's. The
  // default is the main model; the ScaDS.AI models we recommend for the task
  // are only labelled as such.
  function renderRolePickers() {
    const mainUrl = normalizeUrl(llmUrl.value);
    const groups = [];
    const custom = endpointSelect.value === "custom";
    if (mainUrl !== SCADS_ENDPOINT && endpointModels.length) groups.push([mainUrl, endpointModels]);
    if (scadsModels.length) groups.push([SCADS_ENDPOINT, scadsModels]);
    rolePickers.forEach((picker) => {
      const previousValue = picker.select.value;
      picker.select.innerHTML = "";
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "Same as main model";
      picker.select.appendChild(blank);
      const values = new Set([""]);
      if (custom && mainUrl && mainUrl !== SCADS_ENDPOINT) {
        const group = document.createElement("optgroup");
        group.label = endpointName(mainUrl);
        const option = document.createElement("option");
        option.value = CUSTOM_ROLE_MODEL;
        option.textContent = "Type a model name…";
        group.appendChild(option);
        picker.select.appendChild(group);
        values.add(CUSTOM_ROLE_MODEL);
      }
      groups.forEach(([url, models]) => {
        const group = document.createElement("optgroup");
        group.label = endpointName(url);
        models.forEach((model) => {
          const option = document.createElement("option");
          option.value = roleOptionValue(url, model);
          const recommended = url === SCADS_ENDPOINT && model === SCADS_RECOMMENDED_MODELS[`${picker.name}_model`];
          option.textContent = recommended ? `${model} (recommended)` : model;
          group.appendChild(option);
          values.add(option.value);
        });
        picker.select.appendChild(group);
      });
      picker.select.value = picker.userChose && values.has(previousValue) ? previousValue : "";
      syncRoleCustomInput(picker);
    });
    updateFreeKeyNote();
  }

  // Runs on the shared ScaDS.AI key stop after 60 minutes unless the user
  // sets a runtime limit (enforced by the server; this note only explains
  // it). The shared key is used when the main endpoint is ScaDS.AI without a
  // key of the user's own, or when NER/NED run on ScaDS.AI next to another
  // main endpoint.
  function usesSharedKey() {
    const mainUrl = normalizeUrl(llmUrl.value);
    if (mainUrl === SCADS_ENDPOINT && !apiKeyInput.value.trim()) return true;
    return rolePickers.some((picker) => {
      const value = picker.select.value;
      if (!value || value === CUSTOM_ROLE_MODEL) return false;
      return normalizeUrl(JSON.parse(value)[0]) === SCADS_ENDPOINT && mainUrl !== SCADS_ENDPOINT;
    });
  }
  // The note and its "No time limit" box show only for shared-key runs
  // without a runtime limit of their own.
  function updateFreeKeyNote() {
    const freeKeyRuntime = root.querySelector('[data-role="free-key-runtime"]');
    if (!freeKeyRuntime) return;
    const runtimeSet = form.elements.use_runtime.checked && form.elements.max_runtime_seconds.value !== "";
    const show = usesSharedKey() && !runtimeSet;
    freeKeyRuntime.classList.toggle("hidden", !show);
    if (!show) form.elements.no_time_limit.checked = false;
  }

  function syncRoleCustomInput(picker) {
    const typed = picker.select.value === CUSTOM_ROLE_MODEL;
    picker.input.classList.toggle("hidden", !typed);
    picker.input.required = typed;
    picker.input.disabled = !typed;
  }

  async function loadScadsModels() {
    try {
      const response = await postJSON("/api/models", { llm_url: SCADS_ENDPOINT, api_key: "" });
      scadsModels = response.models || [];
    } catch (error) {
      scadsModels = [];
    }
    renderRolePickers();
  }

  async function refreshModelsForEndpoint(endpointUrl) {
    const apiKey = apiKeyInput.value.trim();
    // A blank key is only valid against the ScaDS endpoint, where the
    // backend fills in the project's shared key -- everywhere else a key
    // is required, same as before.
    const usingScadsDefault = !apiKey && normalizeUrl(endpointUrl) === SCADS_ENDPOINT;
    const requestToken = ++modelsRequestToken;

    if (!apiKey && !usingScadsDefault) {
      setModelPlaceholder("Enter an API key to load models");
      return;
    }

    setModelPlaceholder("Loading models...");
    try {
      const response = await postJSON("/api/models", { llm_url: endpointUrl, api_key: apiKey });
      if (requestToken !== modelsRequestToken) return;
      const models = response.models || [];
      if (!models.length) {
        setModelPlaceholder("No models returned by endpoint");
        return;
      }
      setModelOptions(models);
    } catch (error) {
      if (requestToken !== modelsRequestToken) return;
      setModelPlaceholder(`Could not load models: ${error.message}`);
    }
  }

  // Which reasoning controls (if any) apply to the currently-selected model
  // -- re-checked whenever the model, endpoint, or API key changes (the
  // server's answer can depend on all three: OpenRouter's support varies per
  // model and needs a key to look up; ScaDS/OpenAI/Custom depend only on the
  // model+endpoint). See get_reasoning_capability in runner.py for what's
  // actually being asked here.
  let reasoningRequestToken = 0;
  let currentReasoningLevels = [];

  function hideReasoningControls() {
    reasoningToggleField.classList.add("hidden");
    reasoningEffortField.classList.add("hidden");
    currentReasoningLevels = [];
  }

  function titleCase(value) {
    return value.length ? value.charAt(0).toUpperCase() + value.slice(1) : value;
  }

  function applyReasoningCapability(capability) {
    const toggle = capability && capability.toggle;
    const effort = capability && capability.effort;

    reasoningToggleField.classList.toggle("hidden", !toggle);
    if (toggle) {
      reasoningToggleInput.checked = !!toggle.default_on;
      reasoningToggleLabel.textContent = reasoningToggleInput.checked ? "Enabled" : "Disabled";
    }

    reasoningEffortField.classList.toggle("hidden", !effort);
    if (effort) {
      currentReasoningLevels = effort.levels;
      reasoningEffortSlider.min = "0";
      reasoningEffortSlider.max = String(effort.levels.length - 1);
      reasoningEffortSlider.step = "1";
      const defaultIndex = Math.max(0, effort.levels.indexOf(effort.default));
      reasoningEffortSlider.value = String(defaultIndex);
      reasoningEffortScale.innerHTML = "";
      effort.levels.forEach((level) => {
        const span = document.createElement("span");
        span.textContent = titleCase(level);
        reasoningEffortScale.appendChild(span);
      });
      reasoningEffortLabel.textContent = titleCase(effort.levels[defaultIndex]);
    } else {
      currentReasoningLevels = [];
    }
  }

  async function refreshReasoningCapability() {
    const model = (modelInput.value || modelSelect.value || "").trim();
    const llmUrlValue = llmUrl.value.trim();
    if (!model || !llmUrlValue) {
      hideReasoningControls();
      return;
    }
    const requestToken = ++reasoningRequestToken;
    let capability;
    try {
      capability = await postJSON("/api/reasoning-capability", {
        llm_url: llmUrlValue,
        model: model,
        api_key: apiKeyInput.value.trim(),
      });
    } catch (error) {
      capability = { toggle: null, effort: null };
    }
    if (requestToken !== reasoningRequestToken) return;
    applyReasoningCapability(capability);
  }

  reasoningToggleInput.addEventListener("change", () => {
    reasoningToggleLabel.textContent = reasoningToggleInput.checked ? "Enabled" : "Disabled";
  });
  reasoningEffortSlider.addEventListener("input", () => {
    const level = currentReasoningLevels[Number(reasoningEffortSlider.value)];
    if (level) reasoningEffortLabel.textContent = titleCase(level);
  });

  function updateApiKeyRequirement() {
    const usingScads = normalizeUrl(llmUrl.value) === SCADS_ENDPOINT;
    apiKeyInput.required = !usingScads;
    apiKeyInput.placeholder = usingScads ? "sk-... (optional)" : "sk-...";
    scadsKeyNote.classList.toggle("hidden", !usingScads);
  }

  function updateModelControl() {
    const custom = endpointSelect.value === "custom";
    if (!custom) llmUrl.value = endpointSelect.value;
    llmUrlLabel.classList.toggle("hidden", !custom);
    modelSelect.classList.toggle("hidden", custom);
    modelInput.classList.toggle("hidden", !custom);
    // Only the visible control is submitted (and validated as required).
    modelSelect.disabled = custom;
    modelInput.disabled = !custom;
    if (custom) {
      modelInput.name = "model";
      modelSelect.removeAttribute("name");
      endpointModels = [];
      renderRolePickers();
    } else {
      modelSelect.name = "model";
      modelInput.removeAttribute("name");
      refreshModelsForEndpoint(endpointSelect.value);
    }
    // NER/NED can always use ScaDS.AI models; load them once if the
    // elicitation endpoint's list won't include them.
    if (normalizeUrl(llmUrl.value) !== SCADS_ENDPOINT && !scadsModels.length) loadScadsModels();
    updateApiKeyRequirement();
  }

  endpointSelect.addEventListener("change", updateModelControl);
  llmUrl.addEventListener("input", () => {
    syncEndpointSelectionFromUrl();
    updateModelControl();
  });
  modelSelect.addEventListener("change", () => {
    modelInput.value = modelSelect.value;
    refreshReasoningCapability();
  });
  form.addEventListener("input", updateFreeKeyNote);
  form.addEventListener("change", updateFreeKeyNote);
  rolePickers.forEach((picker) => {
    picker.select.addEventListener("change", () => {
      picker.userChose = true;
      syncRoleCustomInput(picker);
      if (picker.select.value === CUSTOM_ROLE_MODEL) picker.input.focus();
    });
  });
  let reasoningModelInputDebounceTimer = null;
  modelInput.addEventListener("input", () => {
    clearTimeout(reasoningModelInputDebounceTimer);
    reasoningModelInputDebounceTimer = setTimeout(refreshReasoningCapability, 400);
  });
  apiKeyInput.addEventListener("input", () => {
    if (endpointSelect.value === "custom") {
      // No dropdown to repopulate, but a custom URL could still point at
      // OpenRouter, whose reasoning support needs a key to look up.
      clearTimeout(reasoningModelInputDebounceTimer);
      reasoningModelInputDebounceTimer = setTimeout(refreshReasoningCapability, 500);
      return;
    }
    clearTimeout(modelsDebounceTimer);
    modelsDebounceTimer = setTimeout(() => {
      refreshModelsForEndpoint(endpointSelect.value);
    }, 500);
  });
  syncEndpointSelectionFromUrl();
  updateModelControl();

  function populateFolderSelect() {
    const previousValue = folderSelect.value;
    folderSelect.innerHTML = "";
    const noneOption = document.createElement("option");
    noneOption.value = "";
    noneOption.textContent = "No folder";
    folderSelect.appendChild(noneOption);
    cachedFoldersList.forEach((folder) => {
      const option = document.createElement("option");
      option.value = folder.id;
      option.textContent = folder.name;
      folderSelect.appendChild(option);
    });
    const newOption = document.createElement("option");
    newOption.value = "__new__";
    newOption.textContent = "+ New folder…";
    folderSelect.appendChild(newOption);

    const stillValid = previousValue === "__new__" || previousValue === "" || cachedFoldersList.some((folder) => folder.id === previousValue);
    folderSelect.value = stillValid ? previousValue : "";
    newFolderNameLabel.classList.toggle("hidden", folderSelect.value !== "__new__");
  }

  folderSelect.addEventListener("change", () => {
    const isNew = folderSelect.value === "__new__";
    newFolderNameLabel.classList.toggle("hidden", !isNew);
    if (isNew) newFolderNameInput.focus();
  });

  populateFolderSelect();
  tab.refreshFolderOptions = populateFolderSelect;

  getLanguages().then((languages) => {
    if (!languages.length) return;
    const previousValue = languageSelect.value || "en";
    languageSelect.innerHTML = "";
    languages.forEach((language) => {
      const option = document.createElement("option");
      option.value = language.code;
      option.textContent = language.name;
      languageSelect.appendChild(option);
    });
    languageSelect.value = languages.some((language) => language.code === previousValue) ? previousValue : "en";
  });

  root.querySelectorAll('input[name="schema_mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      schemaDetails.classList.toggle("hidden", radio.value === "open" && radio.checked);
      refreshPromptPreview();
    });
  });

  schemaFile.addEventListener("change", async () => {
    const file = schemaFile.files[0];
    if (!file) return;
    schemaTextArea.value = await file.text();
    refreshPromptPreview();
  });

  PROMPT_TEMPLATES.forEach((template) => {
    const option = document.createElement("option");
    option.value = template.id;
    option.textContent = template.label;
    promptTemplateSelect.appendChild(option);
  });

  // Loading a template is a one-time "fill the box now" action, same as the
  // schema-file upload above -- it doesn't change the auto-refresh behavior,
  // so touching Language/Desired triples/Topic/Schema afterward will still
  // regenerate the textarea and replace it, same as it would a manual edit.
  promptTemplateSelect.addEventListener("change", () => {
    const template = PROMPT_TEMPLATES.find((item) => item.id === promptTemplateSelect.value);
    if (template) {
      promptPreview.value = template.text;
    } else {
      refreshPromptPreview();
    }
  });

  async function refreshPromptPreview() {
    try {
      const response = await postJSON("/api/prompt-preview", {
        language: languageSelect.value || "en",
        topic: topicInput.value,
        facts_multiplier: Number(factsSlider.value),
        schema_mode: root.querySelector('input[name="schema_mode"]:checked')?.value || "open",
        schema_text: schemaTextArea.value || "",
      });
      promptPreview.value = response.prompt;
    } catch (error) {
      // Leave whatever is currently in the box untouched if the preview call fails.
    }
  }

  function refreshPromptPreviewDebounced() {
    clearTimeout(promptPreviewDebounceTimer);
    promptPreviewDebounceTimer = setTimeout(refreshPromptPreview, 400);
  }

  languageSelect.addEventListener("change", refreshPromptPreview);
  topicInput.addEventListener("input", refreshPromptPreviewDebounced);
  factsSlider.addEventListener("input", refreshPromptPreview);
  schemaTextArea.addEventListener("input", refreshPromptPreviewDebounced);
  refreshPromptPreview();

  logFilter.addEventListener("change", renderLogs);

  setControlsEnabled(false);
  setStatus(null);

  initPaneSplitter(root);

  tab.watchRun = watchRun;
  tab.getPayload = payloadFromForm;
  tab.reportValid = () => form.reportValidity();
  tab.setHtmlIndexLink = setHtmlIndexLink;
  tab.isActiveRun = () => ["running", "starting", "paused"].includes(lastKnownStatus);
  tab.getStatus = () => lastKnownStatus;
  tab.destroy = () => {
    if (eventSource) eventSource.close();
    clearElapsedInterval();
  };

  return tab;
}

// ---- Runs history view (shared across all tabs) ----
// Runs are grouped into folders (a flat, non-nested list fetched from
// /api/folders) plus a catch-all "No folder" bucket for runs with no
// folder_id. Each group is one expandable/collapsible row followed by its
// indented run rows -- collapse state and run selection are UI-local to this
// table and don't need to survive a reload the way the folders themselves do.

const COLLAPSED_FOLDERS_KEY = "miniGPTKB.collapsedFolders";
const ACTIVE_RUN_STATUSES = ["running", "starting", "paused", "stopping"];

function loadCollapsedFolders() {
  try {
    const raw = localStorage.getItem(COLLAPSED_FOLDERS_KEY);
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch (error) {
    return new Set();
  }
}

function saveCollapsedFolders() {
  try {
    localStorage.setItem(COLLAPSED_FOLDERS_KEY, JSON.stringify([...collapsedFolders]));
  } catch (error) {
    // Storage can be unavailable -- collapse state just won't survive a reload.
  }
}

let collapsedFolders = loadCollapsedFolders();
let selectedRunIds = new Set();

function selectableRunIds(runs) {
  return runs.filter((run) => !ACTIVE_RUN_STATUSES.includes(run.status)).map((run) => run.run_id);
}

function updateBulkDeleteButton() {
  const button = document.querySelector("#delete-selected-runs");
  const count = selectedRunIds.size;
  button.classList.toggle("hidden", count === 0);
  button.textContent = `Delete selected (${count})`;
}

function updateMoveSelectedControl() {
  const select = document.querySelector("#move-selected-runs");
  select.classList.toggle("hidden", selectedRunIds.size === 0);
  // Options beyond the fixed "" placeholder and "__none__" are rebuilt from
  // the current folder list every render, same as the other folder pickers.
  while (select.options.length > 2) select.remove(2);
  [...cachedFoldersList]
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach((folder) => {
      const option = document.createElement("option");
      option.value = folder.id;
      option.textContent = folder.name;
      select.appendChild(option);
    });
  select.value = "";
}

function updateSelectAllCheckbox() {
  const checkbox = document.querySelector("#select-all-runs");
  const selectable = selectableRunIds(cachedRunsList);
  checkbox.checked = selectable.length > 0 && selectable.every((id) => selectedRunIds.has(id));
  checkbox.indeterminate = !checkbox.checked && selectedRunIds.size > 0;
}

function buildFolderRow(group, groupRuns, isCollapsed) {
  const row = document.createElement("tr");
  row.className = "folder-row";
  const groupCell = document.createElement("td");
  groupCell.colSpan = 14;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "folder-toggle";
  toggle.textContent = isCollapsed ? "▸" : "▾";
  toggle.setAttribute("aria-label", isCollapsed ? "Expand folder" : "Collapse folder");
  toggle.addEventListener("click", () => {
    const key = group.id || "__none__";
    if (collapsedFolders.has(key)) collapsedFolders.delete(key);
    else collapsedFolders.add(key);
    saveCollapsedFolders();
    renderRuns(cachedRunsList);
  });
  groupCell.appendChild(toggle);

  const nameSpan = document.createElement("span");
  nameSpan.className = "folder-name";
  nameSpan.textContent = group.name;
  groupCell.appendChild(nameSpan);

  const countSpan = document.createElement("span");
  countSpan.className = "folder-count";
  countSpan.textContent = `${groupRuns.length} run${groupRuns.length === 1 ? "" : "s"}`;
  groupCell.appendChild(countSpan);

  if (group.id) {
    const actions = document.createElement("span");
    actions.className = "folder-actions";

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "folder-rename-button";
    renameButton.textContent = "Rename";
    renameButton.addEventListener("click", async () => {
      const nextName = window.prompt("Rename folder", group.name);
      if (!nextName || !nextName.trim() || nextName.trim() === group.name) return;
      try {
        await patchJSON(`/api/folders/${encodeURIComponent(group.id)}`, { name: nextName.trim() });
        await loadFolders();
        refreshFolderSelectAllTabs();
        await loadRuns();
      } catch (error) {
        window.alert(`Could not rename folder: ${error.message}`);
      }
    });
    actions.appendChild(renameButton);

    const deleteFolderButton = document.createElement("button");
    deleteFolderButton.type = "button";
    deleteFolderButton.className = "folder-delete-button";
    deleteFolderButton.textContent = "Delete";
    deleteFolderButton.addEventListener("click", async () => {
      const confirmed = window.confirm(
        groupRuns.length
          ? `Delete folder "${group.name}"? Its ${groupRuns.length} run(s) will move to "No folder".`
          : `Delete folder "${group.name}"?`
      );
      if (!confirmed) return;
      try {
        await deleteJSON(`/api/folders/${encodeURIComponent(group.id)}`);
        await loadFolders();
        refreshFolderSelectAllTabs();
        await loadRuns();
      } catch (error) {
        window.alert(`Could not delete folder: ${error.message}`);
      }
    });
    actions.appendChild(deleteFolderButton);

    groupCell.appendChild(actions);
  }

  row.appendChild(groupCell);
  return row;
}

function buildRunRow(run) {
  const row = document.createElement("tr");
  row.className = "run-row";
  const isActive = ACTIVE_RUN_STATUSES.includes(run.status);

  const selectCell = document.createElement("td");
  selectCell.className = "select-cell";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.disabled = isActive;
  checkbox.checked = selectedRunIds.has(run.run_id);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) selectedRunIds.add(run.run_id);
    else selectedRunIds.delete(run.run_id);
    updateBulkDeleteButton();
    updateMoveSelectedControl();
    updateSelectAllCheckbox();
  });
  selectCell.appendChild(checkbox);
  row.appendChild(selectCell);

  const idCell = cell(run.run_id);
  idCell.className = "run-id-cell";
  row.appendChild(idCell);

  row.appendChild(cell(run.status));
  row.appendChild(cell(run.topic || ""));
  row.appendChild(cell(run.seed_entity || ""));
  const modelCell = cell(run.model || "");
  if (run.pipeline === "classic") {
    modelCell.title = "Classic run (no disambiguation)";
    modelCell.textContent = `${run.model || ""} · classic`;
  } else if (run.ner_model || run.ned_model) {
    const at = (url) => (url ? ` (${url.replace(/^https?:\/\//, "").split("/")[0]})` : "");
    modelCell.title = `Elicitation: ${run.model}\nNER: ${run.ner_model}${at(run.ner_llm_url)}\nNED: ${run.ned_model}${at(run.ned_llm_url)}`;
  }
  row.appendChild(modelCell);
  row.appendChild(cell(formatCost(run.cost_usd)));
  row.appendChild(cell(formatDuration(run.duration_seconds)));
  row.appendChild(cell(run.processed));
  row.appendChild(cell(run.queued));
  row.appendChild(cell(run.triples));


  row.appendChild(cell(formatBytes(run.size_bytes)));
  row.appendChild(cell(shortDate(run.modified_at)));

  const actionCell = document.createElement("td");
  actionCell.className = "run-actions";

  const folderMoveSelect = document.createElement("select");
  folderMoveSelect.className = "row-folder-select";
  folderMoveSelect.title = "Move to folder";
  const noneOption = document.createElement("option");
  noneOption.value = "";
  noneOption.textContent = "No folder";
  folderMoveSelect.appendChild(noneOption);
  cachedFoldersList.forEach((folder) => {
    const option = document.createElement("option");
    option.value = folder.id;
    option.textContent = folder.name;
    folderMoveSelect.appendChild(option);
  });
  folderMoveSelect.value = run.folder_id || "";
  folderMoveSelect.addEventListener("change", async () => {
    const nextFolderId = folderMoveSelect.value || null;
    try {
      await patchJSON(`/api/runs/${encodeURIComponent(run.run_id)}`, { folder_id: nextFolderId });
      await loadRuns();
    } catch (error) {
      window.alert(`Could not move run: ${error.message}`);
      folderMoveSelect.value = run.folder_id || "";
    }
  });
  // Opens the run's HTML results; disabled until a report exists.
  let exploreButton;
  if (run.html_index_url) {
    exploreButton = document.createElement("a");
    exploreButton.href = run.html_index_url;
    exploreButton.target = "_blank";
    exploreButton.rel = "noreferrer";
  } else {
    exploreButton = document.createElement("button");
    exploreButton.type = "button";
    exploreButton.disabled = true;
    exploreButton.title = "No results yet";
  }
  exploreButton.className = "explore-run-button";
  exploreButton.textContent = "Explore";
  actionCell.appendChild(exploreButton);

  actionCell.appendChild(folderMoveSelect);

  const watchButton = document.createElement("button");
  watchButton.type = "button";
  watchButton.className = "watch-run-button";
  watchButton.textContent = "Watch";
  watchButton.addEventListener("click", () => {
    document.querySelector('[data-view="dashboard-view"]').click();
    openRunInTab(run.run_id);
  });
  actionCell.appendChild(watchButton);

  const resumeButton = document.createElement("button");
  resumeButton.type = "button";
  resumeButton.className = "resume-run-button";
  resumeButton.textContent = "Resume";
  resumeButton.disabled = !(run.resumable ?? run.queued > 0) || isActive;
  resumeButton.addEventListener("click", async () => {
    const sourceTab = tabs.get(activeTabId) || tabs.get(tabOrder[0]);
    if (!sourceTab || !sourceTab.reportValid()) {
      window.alert("Fill in the model settings (endpoint, API key, model) in your current tab first, then Resume.");
      return;
    }
    const confirmed = window.confirm(
      `Resume run ${run.run_id} using the current tab's model settings, in a new tab?`
    );
    if (!confirmed) return;
    try {
      const response = await postJSON(
        `/api/runs/${encodeURIComponent(run.run_id)}/resume`,
        sourceTab.getPayload()
      );
      document.querySelector('[data-view="dashboard-view"]').click();
      const newTab = createTab();
      newTab.setHtmlIndexLink(response.html_index_url);
      await newTab.watchRun(run.run_id, { isNewRun: true, startedAt: response.started_at });
      await loadRuns();
    } catch (error) {
      window.alert(`Could not resume run: ${error.message}`);
    }
  });
  actionCell.appendChild(resumeButton);

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "delete-run-button";
  deleteButton.textContent = "Delete";
  deleteButton.disabled = isActive;
  deleteButton.addEventListener("click", async () => {
    const confirmed = window.confirm(`Delete run ${run.run_id}? This removes its files from the runs folder.`);
    if (!confirmed) return;
    try {
      await deleteJSON(`/api/runs/${encodeURIComponent(run.run_id)}`);
      selectedRunIds.delete(run.run_id);
      await loadRuns();
    } catch (error) {
      window.alert(`Could not delete run: ${error.message}`);
    }
  });
  actionCell.appendChild(deleteButton);
  row.appendChild(actionCell);

  return row;
}

function renderRuns(runs) {
  cachedRunsList = runs;
  selectedRunIds = new Set([...selectedRunIds].filter((id) => runs.some((run) => run.run_id === id)));
  runsBody.innerHTML = "";

  if (!runs.length && !cachedFoldersList.length) {
    const row = document.createElement("tr");
    const td = cell("No runs yet.");
    td.colSpan = 14;
    row.appendChild(td);
    runsBody.appendChild(row);
    renderTabStrip();
    updateBulkDeleteButton();
    updateMoveSelectedControl();
    updateSelectAllCheckbox();
    return;
  }

  const runsByFolder = new Map();
  runs.forEach((run) => {
    const key = run.folder_id || null;
    if (!runsByFolder.has(key)) runsByFolder.set(key, []);
    runsByFolder.get(key).push(run);
  });

  const sortedFolders = [...cachedFoldersList].sort((a, b) => a.name.localeCompare(b.name));
  const groups = [...sortedFolders.map((folder) => ({ id: folder.id, name: folder.name }))];
  const ungrouped = runsByFolder.get(null) || [];
  if (ungrouped.length) groups.push({ id: null, name: "No folder" });

  groups.forEach((group) => {
    const groupRuns = runsByFolder.get(group.id) || [];
    const isCollapsed = collapsedFolders.has(group.id || "__none__");
    runsBody.appendChild(buildFolderRow(group, groupRuns, isCollapsed));
    if (!isCollapsed) {
      groupRuns.forEach((run) => runsBody.appendChild(buildRunRow(run)));
    }
  });

  renderTabStrip();
  updateBulkDeleteButton();
  updateMoveSelectedControl();
  updateSelectAllCheckbox();
}

async function loadRuns() {
  try {
    const [runsData] = await Promise.all([fetch("/api/runs").then((response) => response.json()), loadFolders()]);
    renderRuns(runsData.runs || []);
  } catch (error) {
    runsBody.innerHTML = "";
    const row = document.createElement("tr");
    const td = cell(`Could not load runs: ${error.message}`);
    td.colSpan = 14;
    row.appendChild(td);
    runsBody.appendChild(row);
  }
}

// Every open tab's SSE status handler wants the Runs table (and tab labels)
// refreshed, but with several tabs each processing subjects at their own
// pace, calling loadRuns() straight from each one can fire far more full
// table-rebuild-plus-fetch cycles than needed and compete for the single JS
// thread with rendering each tab's own log -- coalesce bursts into one call.
let loadRunsDebounceTimer = null;
function loadRunsDebounced() {
  clearTimeout(loadRunsDebounceTimer);
  loadRunsDebounceTimer = setTimeout(loadRuns, 300);
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".app-view").forEach((view) => view.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.view}`).classList.add("active");
    if (button.dataset.view === "runs-view") {
      loadRuns();
    }
  });
});

refreshRunsButton.addEventListener("click", loadRuns);

document.querySelector("#new-folder-button").addEventListener("click", async () => {
  const name = window.prompt("New folder name");
  if (!name || !name.trim()) return;
  try {
    await postJSON("/api/folders", { name: name.trim() });
    await loadFolders();
    refreshFolderSelectAllTabs();
    await loadRuns();
  } catch (error) {
    window.alert(`Could not create folder: ${error.message}`);
  }
});

document.querySelector("#select-all-runs").addEventListener("change", (event) => {
  const selectable = selectableRunIds(cachedRunsList);
  if (event.target.checked) {
    selectable.forEach((id) => selectedRunIds.add(id));
  } else {
    selectable.forEach((id) => selectedRunIds.delete(id));
  }
  renderRuns(cachedRunsList);
});

document.querySelector("#move-selected-runs").addEventListener("change", async (event) => {
  const value = event.target.value;
  if (!value) return;
  const ids = [...selectedRunIds];
  const folderId = value === "__none__" ? null : value;
  try {
    const result = await postJSON("/api/runs/bulk-move", { run_ids: ids, folder_id: folderId });
    if (result.skipped && result.skipped.length) {
      window.alert(`${result.skipped.length} run(s) could not be moved.`);
    }
    await loadRuns();
  } catch (error) {
    window.alert(`Could not move selected runs: ${error.message}`);
    event.target.value = "";
  }
});

document.querySelector("#delete-selected-runs").addEventListener("click", async () => {
  const ids = [...selectedRunIds];
  if (!ids.length) return;
  const confirmed = window.confirm(`Delete ${ids.length} selected run(s)? This removes their files from the runs folder.`);
  if (!confirmed) return;
  try {
    const result = await postJSON("/api/runs/bulk-delete", { run_ids: ids });
    selectedRunIds.clear();
    if (result.skipped && result.skipped.length) {
      window.alert(`${result.skipped.length} run(s) could not be deleted (still active).`);
    }
    await loadRuns();
  } catch (error) {
    window.alert(`Could not delete selected runs: ${error.message}`);
  }
});

addTabButton.addEventListener("click", () => createTab());

// ---- Initial load: restore tabs from a previous session; otherwise, since
// runs are shared across everyone (not scoped to a browser/user), open a
// tab for whatever is currently running rather than defaulting to a single
// blank tab that would hide live activity from a first-time visitor. ----

loadFolders().then(refreshFolderSelectAllTabs);

loadRuns().then(() => {
  const persisted = loadPersistedTabs();
  if (persisted) {
    persisted.tabs.forEach((entry) => {
      createTab({ runId: entry.runId, activate: false });
    });
    const restoreIndex = persisted.activeIndex >= 0 ? persisted.activeIndex : 0;
    if (tabOrder[restoreIndex]) activateTab(tabOrder[restoreIndex]);
    return;
  }

  const activeRuns = cachedRunsList.filter((run) => ACTIVE_RUN_STATUSES.includes(run.status));
  if (activeRuns.length) {
    activeRuns.forEach((run, index) => createTab({ runId: run.run_id, activate: index === 0 }));
  } else {
    createTab();
  }
});

// ---- Theme toggle: explicit choice persisted per-browser; falls back to
// system preference (the un-stamped default) until the user picks one. ----
(function () {
  const root = document.documentElement;
  const opts = document.querySelectorAll("#theme-toggle .opt");
  if (!opts.length) return;
  const STORAGE_KEY = "miniGPTKB.theme";

  let stored = null;
  try { stored = localStorage.getItem(STORAGE_KEY); } catch (error) {}
  if (stored === "light" || stored === "dark") root.setAttribute("data-theme", stored);

  function currentChoice() {
    const explicit = root.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function paint() {
    const choice = currentChoice();
    opts.forEach((opt) => opt.classList.toggle("active", opt.dataset.themeChoice === choice));
  }
  opts.forEach((opt) => {
    opt.addEventListener("click", () => {
      const choice = opt.dataset.themeChoice;
      root.setAttribute("data-theme", choice);
      try { localStorage.setItem(STORAGE_KEY, choice); } catch (error) {}
      paint();
    });
  });
  paint();
})();

// ---- Resizable layout: sidebar width (and hiding it), the form/log split,
// and the Run Explorer's column widths. All remembered per browser. ----

function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch (error) {
    return null;
  }
}

function writeStored(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (error) {}
}

// Calls onMove(dx) while the handle is dragged; bodyClass sets the resize
// cursor and blocks text selection meanwhile.
function makeDraggable(handle, bodyClass, onStart, onMove, onEnd) {
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX;
    onStart();
    document.body.classList.add(bodyClass);
    handle.setPointerCapture(event.pointerId);
    const move = (moveEvent) => onMove(moveEvent.clientX - startX);
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      document.body.classList.remove(bodyClass);
      onEnd();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
}

const CONFIG_WIDTH_KEY = "miniGPTKB.configWidth";

// The form/log divider of one run tab; the width is shared by all tabs.
function initPaneSplitter(root) {
  const splitter = root.querySelector('[data-role="pane-splitter"]');
  const workspace = root.querySelector(".workspace");
  const configCol = root.querySelector(".config-col");
  if (!splitter || !workspace || !configCol) return;
  let startWidth = 0;
  makeDraggable(
    splitter,
    "resizing-panes",
    () => {
      startWidth = configCol.getBoundingClientRect().width;
    },
    (dx) => {
      const max = workspace.getBoundingClientRect().width - splitter.getBoundingClientRect().width - 320;
      const width = Math.round(Math.max(320, Math.min(max, startWidth + dx)));
      document.documentElement.style.setProperty("--config-width", `${width}px`);
    },
    () => writeStored(CONFIG_WIDTH_KEY, getComputedStyle(document.documentElement).getPropertyValue("--config-width").trim())
  );
}

(function () {
  const rootStyle = document.documentElement.style;
  const savedConfigWidth = readStored(CONFIG_WIDTH_KEY);
  if (savedConfigWidth) rootStyle.setProperty("--config-width", savedConfigWidth);

  // Sidebar: drag its right edge; « hides it, » (top left) brings it back.
  const RAIL_WIDTH_KEY = "miniGPTKB.railWidth";
  const RAIL_COLLAPSED_KEY = "miniGPTKB.railCollapsed";
  const rail = document.querySelector(".rail");
  const savedRailWidth = readStored(RAIL_WIDTH_KEY);
  if (savedRailWidth) rootStyle.setProperty("--rail-width", savedRailWidth);
  if (readStored(RAIL_COLLAPSED_KEY) === "1") document.body.classList.add("rail-collapsed");
  document.getElementById("rail-collapse")?.addEventListener("click", () => {
    document.body.classList.add("rail-collapsed");
    writeStored(RAIL_COLLAPSED_KEY, "1");
  });
  document.getElementById("rail-show")?.addEventListener("click", () => {
    document.body.classList.remove("rail-collapsed");
    writeStored(RAIL_COLLAPSED_KEY, "0");
  });
  const railResizer = document.getElementById("rail-resizer");
  if (rail && railResizer) {
    let startWidth = 0;
    makeDraggable(
      railResizer,
      "resizing-rail",
      () => {
        startWidth = rail.getBoundingClientRect().width;
      },
      (dx) => rootStyle.setProperty("--rail-width", `${Math.round(Math.max(180, Math.min(480, startWidth + dx)))}px`),
      () => writeStored(RAIL_WIDTH_KEY, rootStyle.getPropertyValue("--rail-width"))
    );
  }

  // Run Explorer: drag the right edge of a column header. The first drag
  // freezes every column at its current width (fixed table layout), so
  // changing one column doesn't make the browser redistribute the others.
  const COLUMN_WIDTHS_KEY = "miniGPTKB.runsColumnWidths";
  const table = document.querySelector(".runs-table");
  if (!table) return;
  const headers = Array.from(table.querySelectorAll("thead th"));
  function applyWidths(widths) {
    headers.forEach((th, index) => {
      th.style.width = `${widths[index]}px`;
    });
    table.style.width = `${widths.reduce((sum, width) => sum + width, 0)}px`;
    table.classList.add("fixed-cols");
  }
  let savedWidths = null;
  try {
    savedWidths = JSON.parse(readStored(COLUMN_WIDTHS_KEY) || "null");
  } catch (error) {}
  if (Array.isArray(savedWidths) && savedWidths.length === headers.length) applyWidths(savedWidths);
  headers.forEach((th, index) => {
    if (th.classList.contains("select-cell")) return;
    const handle = document.createElement("span");
    handle.className = "col-resizer";
    handle.title = "Drag to resize the column";
    th.appendChild(handle);
    let widths = [];
    makeDraggable(
      handle,
      "resizing-columns",
      () => {
        widths = headers.map((header) => Math.round(header.getBoundingClientRect().width));
        applyWidths(widths);
        widths = widths.slice();
      },
      (dx) => {
        const next = widths.slice();
        next[index] = Math.max(48, widths[index] + dx);
        applyWidths(next);
      },
      () => writeStored(COLUMN_WIDTHS_KEY, JSON.stringify(headers.map((header) => Math.round(header.getBoundingClientRect().width))))
    );
  });
})();
