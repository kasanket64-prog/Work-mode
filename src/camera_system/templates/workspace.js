'use strict';
const el = id => document.getElementById(id);
const node = (tag, value, className = '') => {
  const element = document.createElement(tag);
  element.textContent = value;
  element.className = className;
  return element;
};
let uploading = false, activeJob = false, online = false, refreshing = false;
let eventSource = null;
let knownCases = [];
let selectedCaseId = null;
const cards = new Map();
const statusNames = {processing: 'Analyzing', cancelling: 'Stopping', completed: 'Ready to review', failed: 'Needs attention', cancelled: 'Stopped', interrupted: 'Interrupted'};
const VALID_ROUTES = new Set(['overview', 'cases', 'search', 'demo', 'analyze', 'history', 'questions']);

function navigateTo(route, pushHash = true) {
  const target = route === 'features' ? 'overview' : (VALID_ROUTES.has(route) ? route : 'overview');
  document.querySelectorAll('.page-view').forEach(section => {
    section.classList.toggle('is-active', section.dataset.page === target);
  });
  document.querySelectorAll('.app-nav a[data-nav]').forEach(link => {
    const active = link.dataset.nav === target;
    link.classList.toggle('is-active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  if (pushHash && window.location.hash !== '#' + target) {
    history.replaceState(null, '', '#' + target);
  }
  window.scrollTo({top: 0, behavior: 'auto'});
  if (target === 'cases') {
    loadCases();
    loadAudit();
  }
}

function uploadError(file, mode) {
  if (!file || !file.size) return 'Choose a video or image that is not empty.';
  if (file.size > 512 * 1024 * 1024) return 'This file is too large. Choose a file up to 512 MB.';
  const dot = file.name.lastIndexOf('.');
  const extension = dot > 0 ? file.name.slice(dot + 1).toLowerCase() : '';
  if (!['mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png'].includes(extension)) return 'Choose an MP4, MOV, AVI, MKV, WebM, JPG or PNG file.';
  if (mode === 'plates' && ['jpg', 'jpeg', 'png'].includes(extension)) return 'Plate reading needs a video. Choose a video or switch to scene analysis.';
  return '';
}
function message(value, error = false) {
  el('message').textContent = value;
  el('message').classList.toggle('error', error);
}
function updateStart() {
  el('start').disabled = uploading || activeJob || !online;
}
function setUploadStep(step) {
  ['step-1', 'step-2', 'step-3'].forEach((id, idx) => {
    const item = el(id);
    if (item) item.classList.toggle('selected', idx + 1 === step);
  });
}
async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Something went wrong. Please try again.');
  return result;
}
async function loadSystemStatus() {
  const strip = el('system-status');
  if (!strip) return;
  try {
    const info = await jsonRequest('/api/system/status');
    strip.replaceChildren();
    const caps = info.capabilities || {};
    const mb = ((info.workspace_bytes || 0) / 1048576).toFixed(1);
    const items = [
      ['Detection', Boolean(caps.detection)],
      ['Relations', Boolean(caps.inference)],
      ['Video', Boolean(caps.video)],
      ['Plates', Boolean(caps.plates)],
      ['CUDA', Boolean(caps.cuda)],
      ['Warm worker', Boolean(info.warm_worker)],
    ];
    for (const [label, ready] of items) {
      strip.append(node('span', `${ready ? '●' : '○'} ${label}`, `status-pill${ready ? ' ready' : ''}`));
    }
    strip.append(node('span', `Storage: ${mb} MB · ${info.job_count || 0} saved`, 'status-pill'));
    if (el('kpi-storage')) el('kpi-storage').textContent = `${mb} MB`;
    if (el('overview-runtime')) {
      const readyModules = [caps.detection, caps.inference, caps.video, caps.plates, caps.cuda].filter(Boolean).length;
      el('overview-runtime').textContent = `${info.warm_worker ? 'Warm worker' : 'Subprocess'} · ${readyModules}/5 extras`;
    }
  } catch {
    strip.textContent = 'Local runtime status unavailable.';
    if (el('overview-runtime')) el('overview-runtime').textContent = 'Runtime offline';
  }
}
async function loadDemo() {
  try {
    const data = await jsonRequest('/assets/demo-result.json');
    const [width, height] = data.image_size;
    const boxes = new Map();
    const centers = new Map();
    for (const region of data.regions) {
      const [x1, y1, x2, y2] = region.box;
      const box = node('div', '', 'region');
      Object.assign(box.style, {left: `${x1 / width * 100}%`, top: `${y1 / height * 100}%`, width: `${(x2 - x1) / width * 100}%`, height: `${(y2 - y1) / height * 100}%`});
      box.append(node('span', region.label));
      el('demo-boxes').append(box);
      boxes.set(region.id, box);
      centers.set(region.id, [((x1 + x2) / 2) / width * 100, ((y1 + y2) / 2) / height * 100]);
    }
    const relationship = data.relations[0];
    const svg = el('demo-svg');
    if (svg && relationship && centers.has(relationship.subject_id) && centers.has(relationship.object_id)) {
      const ns = 'http://www.w3.org/2000/svg';
      const [sx, sy] = centers.get(relationship.subject_id);
      const [ox, oy] = centers.get(relationship.object_id);
      const path = document.createElementNS(ns, 'path');
      const mx = (sx + ox) / 2;
      const my = Math.min(sy, oy) - 10;
      path.setAttribute('d', `M ${sx.toFixed(2)} ${sy.toFixed(2)} Q ${mx.toFixed(2)} ${my.toFixed(2)} ${ox.toFixed(2)} ${oy.toFixed(2)}`);
      path.setAttribute('class', 'rel-line');
      const c1 = document.createElementNS(ns, 'circle');
      c1.setAttribute('cx', sx.toFixed(2)); c1.setAttribute('cy', sy.toFixed(2)); c1.setAttribute('r', '1.4'); c1.setAttribute('class', 'rel-node');
      const c2 = document.createElementNS(ns, 'circle');
      c2.setAttribute('cx', ox.toFixed(2)); c2.setAttribute('cy', oy.toFixed(2)); c2.setAttribute('r', '1.4'); c2.setAttribute('class', 'rel-node');
      svg.replaceChildren(path, c1, c2);
    }
    if (relationship && typeof relationship.score === 'number') {
      const pct = Math.max(0, Math.min(100, relationship.score * 100));
      if (el('demo-score-pct')) el('demo-score-pct').textContent = relationship.score.toFixed(3);
      if (el('demo-score-bar')) el('demo-score-bar').style.width = `${pct.toFixed(1)}%`;
    }
    const labels = new Map(data.regions.map(r => [r.id, r.label]));
    const notes = data.regions.map((r, i) => ({title: r.label[0].toUpperCase() + r.label.slice(1), subtitle: 'Manually marked region', icon: String(i + 1).padStart(2, '0'), ids: [r.id], description: `The ${r.label} region was supplied manually for this example. The relationship model uses this part of the image to understand the scene.`}));
    notes.push({title: `${labels.get(relationship.subject_id)} ${relationship.predicate} ${labels.get(relationship.object_id)}`, subtitle: 'Predicted relationship', icon: '↗', ids: [relationship.subject_id, relationship.object_id], description: 'The model connects these two regions with “' + relationship.predicate + '”. A relationship adds context beyond identifying individual objects.'});
    el('demo-findings').replaceChildren();
    const buttons = [];
    function select(index) {
      buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
      boxes.forEach((box, id) => {
        box.classList.toggle('dim', !notes[index].ids.includes(id));
        box.classList.toggle('active', notes[index].ids.includes(id));
      });
      if (svg) svg.hidden = notes[index].ids.length < 2;
      el('demo-description').textContent = notes[index].description;
      el('show-boxes').checked = true;
      el('demo-boxes').hidden = false;
    }
    notes.forEach((note, index) => {
      const button = node('button', '', 'finding');
      button.type = 'button';
      const label = node('span', '');
      label.append(node('strong', note.title), node('small', note.subtitle));
      button.append(node('span', note.icon, 'finding-icon'), label, node('span', '↗', 'arrow'));
      button.onclick = () => select(index);
      buttons.push(button);
      el('demo-findings').append(button);
    });
    select(notes.length - 1);
    el('demo-score').textContent = `Relationship score: ${relationship.score.toFixed(3)}. This is not an accuracy percentage. The original result also includes a lower-scoring reverse relationship.`;
  } catch {
    el('demo-findings').replaceChildren(node('p', 'The saved example could not load. Refresh the page to try again.', 'error'));
  }
}
el('show-boxes').addEventListener('change', () => {
  const show = el('show-boxes').checked;
  el('demo-boxes').hidden = !show;
  if (el('demo-svg')) el('demo-svg').style.display = show ? '' : 'none';
});

function syncCaseSelects() {
  for (const selId of ['case-select', 'search-case']) {
    const sel = el(selId);
    if (!sel) continue;
    const prev = sel.value;
    const defaultLabel = selId === 'case-select' ? 'Unassigned standalone clip' : 'All Cases & Unassigned Clips';
    sel.replaceChildren(node('option', defaultLabel));
    sel.firstChild.value = '';
    for (const c of knownCases) {
      const opt = node('option', `${c.case_number} · ${c.title}`);
      opt.value = c.id;
      sel.append(opt);
    }
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  }
}

function jobCard(job) {
  const card = node('article', '', 'job');
  card.dataset.jobId = job.id;
  const heading = node('div', '', 'job-heading');
  const title = node('div', '');
  title.append(node('span', statusNames[job.status] || job.status, 'status ' + job.status), node('strong', job.mode === 'scene' ? 'Scene analysis' : 'Plate reading'));
  if (job.evaluation_count > 0) {
    title.append(document.createTextNode(' '), node('span', `${job.evaluation_count} evaluation${job.evaluation_count === 1 ? '' : 's'}`, 'eval-badge'));
  }
  const date = node('time', new Date(job.created).toLocaleString());
  date.dateTime = job.created;
  heading.append(title, date);
  card.append(heading);
  if (job.camera_label || job.source_sha256 || job.case_id) {
    const custody = node('div', '', 'custody-strip');
    if (job.camera_label) custody.append(node('span', `Cam: ${job.camera_label}`, 'mono-chip'));
    if (job.source_sha256) {
      const shaChip = node('span', `SHA-256: ${job.source_sha256.slice(0, 12)}…`, 'mono-chip');
      shaChip.title = job.source_sha256;
      custody.append(shaChip);
    }
    if (job.case_id) {
      const foundCase = knownCases.find(c => c.id === job.case_id);
      custody.append(node('span', `Case: ${foundCase ? foundCase.case_number : job.case_id}`, 'mono-chip'));
    }
    card.append(custody);
  }
  if (job.status === 'processing' || job.status === 'cancelling') {
    const steps = node('div', '', 'processing-steps');
    steps.append(node('span', '✓ Upload ready'), node('span', job.status === 'cancelling' ? 'Stopping analysis…' : job.frames_processed ? '● Understanding your footage' : '● Preparing models', 'current'), node('span', '○ Review findings'));
    card.append(steps);
  }
  const metaParts = [`${job.frames_processed || 0} sampled frames processed`, `${(job.bytes / 1048576).toFixed(1)} MB`];
  if (job.stride) metaParts.push(`stride ${job.stride}`);
  if (job.reasoning_profile && job.reasoning_profile !== 'default') metaParts.push(job.reasoning_profile);
  card.append(node('p', metaParts.join(' · ')));
  if (job.status === 'processing' && !job.frames_processed) card.append(node('p', 'The first run can take longer while models load. Keep this window and the local server running.'));
  const actions = node('div', '', 'job-actions');
  if (job.status === 'completed') {
    const review = node('a', 'Explore findings ↗', 'button small dark');
    review.href = '/jobs/' + job.id + '/review';
    const download = node('a', 'Download results');
    download.href = '/jobs/' + job.id + '/result';
    download.download = 'fieldnote-' + job.id + '.jsonl';
    actions.append(review, download);
  }
  if (job.status === 'processing') {
    const cancel = node('button', 'Stop analysis');
    cancel.type = 'button';
    cancel.onclick = async () => {
      cancel.disabled = true;
      try {
        await jsonRequest('/api/jobs/' + job.id + '/cancel', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        await refresh();
      } catch (error) {message(error.message, true); cancel.disabled = false;}
    };
    actions.append(cancel);
  }
  if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
    const retry = node('button', 'Retry saved upload');
    retry.type = 'button';
    retry.dataset.retry = 'true';
    retry.onclick = async () => {
      retry.disabled = true;
      try {
        await jsonRequest('/api/jobs/' + job.id + '/retry', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        message('A new analysis has started. Your previous run is preserved.');
        await refresh();
      } catch (error) {message(error.message, true); retry.disabled = false;}
    };
    actions.append(retry);
  }
  if (!['processing', 'cancelling'].includes(job.status)) {
    const remove = node('button', 'Delete', 'danger-btn');
    remove.type = 'button';
    let armed = false;
    let resetTimer = null;
    remove.onclick = async () => {
      if (!armed) {
        armed = true;
        remove.textContent = 'Confirm delete';
        remove.classList.add('confirming');
        resetTimer = setTimeout(() => {
          armed = false;
          remove.textContent = 'Delete';
          remove.classList.remove('confirming');
        }, 4000);
        return;
      }
      clearTimeout(resetTimer);
      remove.disabled = true;
      try {
        await jsonRequest('/api/jobs/' + job.id, {method: 'DELETE'});
        message('Analysis removed from your local workspace.');
        await Promise.all([refresh(), loadSystemStatus(), loadCases()]);
      } catch (error) {
        message(error.message, true);
        remove.disabled = false;
      }
    };
    actions.append(remove);
  }
  card.append(actions);
  if (knownCases.length && !['processing', 'cancelling'].includes(job.status)) {
    const assignRow = node('div', '', 'job-assign-row');
    const assignSelect = document.createElement('select');
    assignSelect.setAttribute('aria-label', 'Assign analysis to case');
    const un = node('option', 'Case: Unassigned'); un.value = '';
    assignSelect.append(un);
    for (const c of knownCases) {
      const opt = node('option', `${c.case_number} · ${c.title}`);
      opt.value = c.id;
      assignSelect.append(opt);
    }
    assignSelect.value = job.case_id || '';
    assignSelect.onchange = async () => {
      try {
        await jsonRequest(`/api/jobs/${job.id}/metadata`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({case_id: assignSelect.value || null}),
        });
        await Promise.all([refresh(), loadCases()]);
      } catch (error) {
        message(error.message, true);
      }
    };
    assignRow.append(assignSelect);
    card.append(assignRow);
  }
  if (job.error) card.append(node('p', job.error, 'error'));
  const details = node('details', '');
  details.append(node('summary', 'Coverage & processing details'), node('p', job.coverage || 'Coverage has not been recorded yet.'), node('p', `Sample limit: ${job.limit}. Images use a single result. Completion does not guarantee the whole recording was analyzed.`));
  if (Number.isFinite(job.last_timestamp_seconds)) details.append(node('p', `Last sampled moment: ${job.last_timestamp_seconds.toFixed(2)} seconds`));
  if (job.retry_of) details.append(node('p', 'Retry of analysis ' + job.retry_of));
  if (job.has_log) {
    const log = node('a', 'Open processing log ↗');
    log.href = '/jobs/' + job.id + '/log'; log.target = '_blank'; log.rel = 'noopener';
    details.append(log);
  }
  card.append(details);
  if (job.status === 'completed' && job.mode === 'scene') addEvaluation(card, job);
  return card;
}

function matchesJob(job, query, filter) {
  const statuses = {active: ['processing', 'cancelling'], attention: ['failed', 'cancelled', 'interrupted']};
  if (filter !== 'all' && !(statuses[filter] || [filter]).includes(job.status)) return false;
  const title = job.mode === 'scene' ? 'Scene analysis' : 'Plate reading';
  return `${title} ${job.camera_label || ''} ${job.id} ${job.created} ${new Date(job.created).toLocaleString()}`.toLowerCase().includes(query.trim().toLowerCase());
}
function applyFilters() {
  let visible = 0;
  const currentFilter = el('job-filter').value;
  document.querySelectorAll('.filter-chip').forEach(btn => {
    btn.classList.toggle('is-selected', btn.dataset.filter === currentFilter);
  });
  for (const {card, job} of cards.values()) {
    card.hidden = !matchesJob(job, el('job-search').value, currentFilter);
    if (!card.hidden) visible++;
  }
  el('filter-count').textContent = cards.size ? `${visible} of ${cards.size} analyses` : '';
  el('no-matches').hidden = !cards.size || visible > 0;
}
el('job-search').addEventListener('input', applyFilters);
el('job-filter').addEventListener('change', applyFilters);
document.querySelectorAll('.filter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    el('job-filter').value = chip.dataset.filter;
    applyFilters();
  });
});

function applyJobs(jobs) {
  online = true;
  const wasActive = activeJob;
  activeJob = jobs.some(j => ['processing', 'cancelling'].includes(j.status));
  const readyCount = jobs.filter(j => j.status === 'completed').length;
  const activeCount = jobs.filter(j => ['processing', 'cancelling'].includes(j.status)).length;
  if (el('nav-job-count')) el('nav-job-count').textContent = String(jobs.length);
  if (el('kpi-total')) el('kpi-total').textContent = String(jobs.length);
  if (el('kpi-ready')) el('kpi-ready').textContent = String(readyCount);
  if (el('kpi-active')) el('kpi-active').textContent = String(activeCount);
  if (el('nav-live-dot')) {
    el('nav-live-dot').classList.add('online');
    const txt = el('nav-live-dot').querySelector('span');
    if (txt) txt.textContent = activeJob ? 'Running' : 'Live SSE';
  }
  el('connection').textContent = activeJob ? 'Analysis in progress' : (eventSource && eventSource.readyState === 1 ? 'Live workspace connected' : 'Local workspace connected');
  el('connection').className = 'connection online';
  // Keep unchanged cards mounted so updates preserve annotation inputs and review controls.
  const liveIds = new Set(jobs.map(j => j.id));
  for (const [id, entry] of cards) if (!liveIds.has(id)) {entry.card.remove(); cards.delete(id);}
  const empty = el('jobs').querySelector('.empty-state');
  if (empty && jobs.length) empty.remove();
  let previous = null;
  const casesKey = knownCases.map(c => c.id).join(',');
  for (const job of jobs) {
    const signature = JSON.stringify(job) + '|' + casesKey;
    let entry = cards.get(job.id);
    if (!entry || entry.signature !== signature) {
      const card = jobCard(job);
      if (entry) entry.card.replaceWith(card);
      else el('jobs').insertBefore(card, previous ? previous.nextSibling : el('jobs').firstChild);
      entry = {card, job, signature};
      cards.set(job.id, entry);
    }
    const next = previous ? previous.nextSibling : el('jobs').firstChild;
    if (next !== entry.card) el('jobs').insertBefore(entry.card, next);
    previous = entry.card;
  }
  el('jobs').querySelectorAll('[data-retry]').forEach(button => {button.disabled = activeJob || uploading;});
  if (!jobs.length && (!empty || !empty.dataset.ready)) {
    el('jobs').replaceChildren();
    const blank = node('div', '', 'empty-state');
    blank.dataset.ready = 'true';
    const link = node('a', 'Start your first analysis ↗');
    link.href = '#analyze';
    link.dataset.go = 'analyze';
    blank.append(node('strong', 'Your next discovery starts here.'), node('span', 'Upload a recording in Analyze Studio and your findings will appear here.'), document.createElement('br'), link);
    el('jobs').append(blank);
  }
  applyFilters();
  const announcement = jobs.length ? `${jobs.length} analyses. ${activeJob ? 'One is running.' : 'No analysis is running.'}` : 'Your workspace is ready for its first upload.';
  if (el('job-announcement').textContent !== announcement) el('job-announcement').textContent = announcement;
  if (wasActive && !activeJob) {
    loadSystemStatus();
    if (el('file')?.files?.length) setUploadStep(3);
  }
  updateStart();
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const jobs = await jsonRequest('/api/jobs');
    applyJobs(jobs);
  } catch {
    online = false;
    el('connection').textContent = 'Disconnected · reconnecting automatically';
    el('connection').className = 'connection error';
    if (el('nav-live-dot')) el('nav-live-dot').classList.remove('online');
    updateStart();
  } finally {refreshing = false;}
}

function connectEvents() {
  if (typeof EventSource === 'undefined') return;
  try {
    eventSource = new EventSource('/api/events');
    eventSource.addEventListener('jobs', event => {
      try {
        applyJobs(JSON.parse(event.data));
      } catch {}
    });
    eventSource.onerror = () => {
      if (eventSource) {eventSource.close(); eventSource = null;}
    };
  } catch {
    eventSource = null;
  }
}

/* Detective Cases Command Center, Audit Chain & Cross-Clip Search */
async function loadAudit() {
  const pill = el('audit-badge');
  const txt = el('audit-verify-text');
  const list = el('audit-log-list');
  if (!pill) return;
  try {
    const data = await jsonRequest('/api/audit?limit=40');
    pill.className = `audit-pill ${data.chain_valid ? 'verified' : 'tampered'}`;
    pill.textContent = data.chain_valid
      ? `✓ Audit Chain Verified (${data.chain_count} entries)`
      : `⚠ Audit Integrity Warning (${data.chain_count} entries)`;
    if (txt) {
      txt.textContent = data.chain_valid
        ? `SHA-256 chain verified intact across ${data.chain_count} logged custody events.`
        : 'Tamper check detected a hash mismatch in audit_log.';
    }
    if (list) {
      list.replaceChildren();
      if (!data.entries.length) list.append(node('p', 'No custody events logged yet.', 'muted'));
      for (const item of data.entries.slice(0, 18)) {
        const row = node('div', '', 'audit-row');
        row.append(
          node('span', `${new Date(item.timestamp).toLocaleString()} · ${item.action}`),
          node('span', `${item.target_type}:${item.target_id.slice(0, 14)} · ${item.entry_hash.slice(0, 12)}…`),
        );
        list.append(row);
      }
    }
  } catch {
    pill.textContent = 'Audit chain unavailable';
  }
}

async function loadCaseDetail(caseId) {
  if (!caseId) return;
  const empty = el('case-empty');
  const panel = el('case-detail-panel');
  try {
    const data = await jsonRequest(`/api/cases/${encodeURIComponent(caseId)}`);
    const c = data.case;
    if (empty) empty.hidden = true;
    if (panel) panel.hidden = false;
    el('detail-case-num').textContent = c.case_number;
    el('detail-case-title').textContent = c.title;
    el('detail-case-investigator').textContent = `Investigator: ${c.investigator || 'Unassigned'} · Updated ${new Date(c.updated_at).toLocaleString()}`;
    el('detail-case-notes').textContent = c.notes || 'No investigator notes.';
    el('detail-case-status').value = c.status;
    el('case-report-link').href = `/cases/${encodeURIComponent(c.id)}/report`;

    const mediaHost = el('case-media-list');
    mediaHost.replaceChildren();
    if (!data.jobs.length) {
      mediaHost.append(node('p', 'No media assigned to this case yet. Assign clips in Workspace or upload via Analyze.', 'muted'));
    }
    for (const j of data.jobs) {
      const card = node('div', '', 'dossier-card');
      const top = node('div', '', 'dossier-top');
      top.append(
        node('strong', `${j.camera_label || j.original_filename || j.id.slice(0, 8)} (${j.mode})`),
        node('span', j.status, `status ${j.status}`),
      );
      const shaText = j.source_sha256 ? `SHA-256: ${j.source_sha256}` : 'SHA-256: not recorded';
      const calRow = node('div', '', 'calibration-row');
      const startIn = document.createElement('input');
      startIn.type = 'text';
      startIn.placeholder = 'Recorded start UTC ISO (e.g. 2026-11-22T14:30:00Z)';
      startIn.value = j.recorded_start_iso || '';
      const offIn = document.createElement('input');
      offIn.type = 'number';
      offIn.step = '0.1';
      offIn.placeholder = 'Offset (s)';
      offIn.value = String(j.clock_offset_seconds ?? 0);
      const saveCal = node('button', 'Save Calibration');
      saveCal.type = 'button';
      saveCal.onclick = async () => {
        saveCal.disabled = true;
        try {
          await jsonRequest(`/api/jobs/${j.id}/metadata`, {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              recorded_start_iso: startIn.value.trim() || null,
              clock_offset_seconds: Number(offIn.value || 0),
            }),
          });
          await Promise.all([loadCaseDetail(caseId), loadAudit(), refresh()]);
        } catch (err) {
          saveCal.textContent = err.message;
        } finally {
          saveCal.disabled = false;
        }
      };
      calRow.append(startIn, offIn, saveCal);
      const btns = node('div', '', 'page-actions');
      if (j.status === 'completed') {
        const rev = node('a', 'Forensic Review ↗', 'button small dark');
        rev.href = `/jobs/${j.id}/review`;
        btns.append(rev);
      }
      const unlink = node('button', 'Unassign', 'icon-btn');
      unlink.type = 'button';
      unlink.onclick = async () => {
        await jsonRequest(`/api/jobs/${j.id}/metadata`, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({case_id: null}),
        });
        await Promise.all([loadCases(), refresh(), loadAudit()]);
      };
      btns.append(unlink);
      card.append(top, node('span', shaText, 'mono-chip'), calRow, btns);
      mediaHost.append(card);
    }

    const bmHost = el('case-bookmarks-list');
    bmHost.replaceChildren();
    if (!data.bookmarks.length) {
      bmHost.append(node('p', 'No pinned bookmarks yet. Open any recording in Forensic Review and pin key frames.', 'muted'));
    }
    for (const bm of data.bookmarks) {
      const card = node('div', '', 'dossier-card');
      const top = node('div', '', 'dossier-top');
      top.append(
        node('span', `${bm.tag.toUpperCase()} · frame #${bm.frame_index}`, 'status'),
        node('span', bm.wall_clock || `t=${bm.timestamp_seconds.toFixed(2)}s`, 'mono-chip'),
      );
      const act = node('div', '', 'page-actions');
      const jump = node('a', 'Inspect Frame ↗', 'button small dark');
      jump.href = `/jobs/${bm.job_id}/review#frame=${bm.frame_index}`;
      const del = node('button', 'Remove', 'danger-btn icon-btn');
      del.type = 'button';
      del.onclick = async () => {
        await jsonRequest(`/api/bookmarks/${bm.id}`, {method: 'DELETE'});
        await Promise.all([loadCaseDetail(caseId), loadCases(), loadAudit()]);
      };
      act.append(jump, del);
      card.append(top, node('strong', bm.caption), act);
      bmHost.append(card);
    }

    const entHost = el('case-entities-list');
    entHost.replaceChildren();
    if (!data.entities.length) {
      entHost.append(node('p', 'No manually linked subject or vehicle dossiers yet.', 'muted'));
    }
    for (const ent of data.entities) {
      const card = node('div', '', 'dossier-card');
      const top = node('div', '', 'dossier-top');
      top.append(node('strong', ent.label), node('span', ent.kind.toUpperCase(), 'status'));
      const tracks = ent.linked_tracks.map(t => `${t.job_id.slice(0, 8)}:${t.track_id}`).join(', ') || 'None';
      const del = node('button', 'Delete', 'danger-btn icon-btn');
      del.type = 'button';
      del.onclick = async () => {
        await jsonRequest(`/api/entities/${ent.id}`, {method: 'DELETE'});
        await Promise.all([loadCaseDetail(caseId), loadAudit()]);
      };
      card.append(top, node('p', ent.description || 'No notes', 'muted'), node('span', `Linked tracks: ${tracks}`, 'mono-chip'), del);
      entHost.append(card);
    }
  } catch {
    if (empty) empty.hidden = false;
    if (panel) panel.hidden = true;
  }
}

async function loadCases() {
  const host = el('cases-list');
  try {
    const data = await jsonRequest('/api/cases');
    knownCases = data.cases || [];
    if (el('nav-case-count')) el('nav-case-count').textContent = String(knownCases.length);
    syncCaseSelects();
    if (!host) return;
    host.replaceChildren();
    if (!knownCases.length) {
      host.append(node('p', 'No cases opened yet. Use the form above to open your first case.', 'muted'));
      selectedCaseId = null;
      if (el('case-empty')) el('case-empty').hidden = false;
      if (el('case-detail-panel')) el('case-detail-panel').hidden = true;
      return;
    }
    if (!selectedCaseId || !knownCases.some(c => c.id === selectedCaseId)) {
      selectedCaseId = knownCases[0].id;
    }
    for (const c of knownCases) {
      const item = node('button', '', `case-item${c.id === selectedCaseId ? ' is-selected' : ''}`);
      item.type = 'button';
      const top = node('div', '', 'case-item-top');
      top.append(node('span', c.case_number), node('span', c.status.replace('_', ' ')));
      item.append(top, node('strong', c.title), node('small', `${c.job_count || 0} clips · ${c.bookmark_count || 0} bookmarks`));
      item.onclick = () => {
        selectedCaseId = c.id;
        host.querySelectorAll('.case-item').forEach(b => b.classList.remove('is-selected'));
        item.classList.add('is-selected');
        loadCaseDetail(c.id);
      };
      host.append(item);
    }
    if (selectedCaseId) await loadCaseDetail(selectedCaseId);
  } catch {
    if (host) host.textContent = 'Could not load cases.';
  }
}

if (el('case-create-form')) {
  el('case-create-form').addEventListener('submit', async evt => {
    evt.preventDefault();
    const msg = el('case-form-msg');
    try {
      const res = await jsonRequest('/api/cases', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          case_number: el('new-case-number').value.trim(),
          title: el('new-case-title').value.trim(),
          investigator: el('new-case-investigator').value.trim(),
          notes: el('new-case-notes').value.trim(),
        }),
      });
      selectedCaseId = res.case.id;
      el('case-create-form').reset();
      if (msg) msg.textContent = `Opened ${res.case.case_number}.`;
      await Promise.all([loadCases(), loadAudit(), refresh()]);
    } catch (err) {
      if (msg) msg.textContent = err.message;
    }
  });
}

if (el('detail-case-status')) {
  el('detail-case-status').addEventListener('change', async () => {
    if (!selectedCaseId) return;
    await jsonRequest(`/api/cases/${encodeURIComponent(selectedCaseId)}`, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({status: el('detail-case-status').value}),
    });
    await Promise.all([loadCases(), loadAudit()]);
  });
}

if (el('case-delete-btn')) {
  let deleteArmed = false;
  let deleteTimer = null;
  el('case-delete-btn').addEventListener('click', async () => {
    if (!selectedCaseId) return;
    const btn = el('case-delete-btn');
    if (!deleteArmed) {
      deleteArmed = true;
      btn.textContent = 'Confirm delete case';
      btn.classList.add('confirming');
      deleteTimer = setTimeout(() => {
        deleteArmed = false;
        btn.textContent = 'Delete Case';
        btn.classList.remove('confirming');
      }, 4000);
      return;
    }
    clearTimeout(deleteTimer);
    deleteArmed = false;
    btn.textContent = 'Delete Case';
    btn.classList.remove('confirming');
    await jsonRequest(`/api/cases/${encodeURIComponent(selectedCaseId)}`, {method: 'DELETE'});
    selectedCaseId = null;
    await Promise.all([loadCases(), refresh(), loadAudit()]);
  });
}

if (el('entity-form')) {
  el('entity-form').addEventListener('submit', async evt => {
    evt.preventDefault();
    if (!selectedCaseId) return;
    const msg = el('entity-msg');
    try {
      const links = el('entity-tracks').value
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)
        .map(part => {
          const [jid, tid] = part.split(':').map(x => x.trim());
          if (!jid || !tid) throw Error('Format tracks as jobId:trackId (comma-separated).');
          return {job_id: jid, track_id: tid};
        });
      await jsonRequest(`/api/cases/${encodeURIComponent(selectedCaseId)}/entities`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          label: el('entity-label').value.trim(),
          kind: el('entity-kind').value,
          linked_tracks: links,
          description: el('entity-desc').value.trim(),
        }),
      });
      el('entity-form').reset();
      if (msg) msg.textContent = 'Entity dossier link saved.';
      await Promise.all([loadCaseDetail(selectedCaseId), loadAudit()]);
    } catch (err) {
      if (msg) msg.textContent = err.message;
    }
  });
}

async function runLeadSearch(evt) {
  if (evt) evt.preventDefault();
  const params = new URLSearchParams();
  if (el('search-q')?.value.trim()) params.set('q', el('search-q').value.trim());
  if (el('search-case')?.value) params.set('case_id', el('search-case').value);
  if (el('search-label')?.value.trim()) params.set('label', el('search-label').value.trim());
  if (el('search-color')?.value.trim()) params.set('color', el('search-color').value.trim());
  if (el('search-plate')?.value.trim()) params.set('plate', el('search-plate').value.trim());
  if (el('search-event')?.value.trim()) params.set('event_type', el('search-event').value.trim());
  const host = el('search-results');
  const summary = el('search-summary');
  if (!host) return;
  try {
    const data = await jsonRequest('/api/search?' + params.toString());
    const results = data.results || [];
    if (summary) summary.textContent = `${results.length} matching observation${results.length === 1 ? '' : 's'}`;
    host.replaceChildren();
    if (!results.length) {
      const empty = node('div', '', 'empty-state');
      empty.append(node('strong', 'No matching leads found'), node('span', 'Try broader keywords, partial plate characters, or clear the case filter.'));
      host.append(empty);
      return;
    }
    for (const hit of results) {
      const card = node('article', '', 'search-card');
      if (hit.preview_url) {
        const img = document.createElement('img');
        img.src = hit.preview_url;
        img.alt = `Frame ${hit.frame_index}`;
        card.append(img);
      } else {
        card.append(node('div', `FRAME ${hit.frame_index}`, 'search-thumb-empty'));
      }
      const info = node('div', '');
      const top = node('div', '', 'dossier-top');
      top.append(node('span', hit.match_kind.toUpperCase(), 'status'), node('span', hit.wall_clock || `t=${hit.timestamp_seconds.toFixed(2)}s`, 'mono-chip'));
      const openLink = node('a', 'Inspect Frame ↗', 'button small dark');
      openLink.href = hit.review_url;
      openLink.style.marginTop = '8px';
      info.append(
        top,
        node('strong', hit.summary),
        node('p', `Camera: ${hit.camera_label}${hit.source_sha256 ? ' · SHA-256 ' + hit.source_sha256.slice(0, 10) + '…' : ''}`, 'muted'),
        openLink,
      );
      card.append(info);
      host.append(card);
    }
  } catch (err) {
    if (summary) summary.textContent = err.message;
  }
}

if (el('lead-search-form')) {
  el('lead-search-form').addEventListener('submit', runLeadSearch);
}
if (el('search-reset')) {
  el('search-reset').addEventListener('click', () => {
    el('lead-search-form').reset();
    if (el('search-summary')) el('search-summary').textContent = '';
    runLeadSearch();
  });
}

let previewUrl;
function updateModeUI() {
  const wrap = el('crops-wrap');
  if (wrap) wrap.hidden = el('mode').value !== 'plates';
}
function fileSelected() {
  updateModeUI();
  const video = el('preview-video'), image = el('preview-image');
  video.pause(); video.removeAttribute('src'); video.load(); image.removeAttribute('src');
  video.hidden = image.hidden = true;
  if (previewUrl) {URL.revokeObjectURL(previewUrl); previewUrl = null;}

  const file = el('file').files[0];
  el('file-title').textContent = file ? file.name : 'Drop your video or image here';
  el('file-caption').textContent = file ? `${(file.size / 1048576).toFixed(1)} MB · click to choose a different file` : 'or browse files on your computer';
  const error = file ? uploadError(file, el('mode').value) : '';
  message(error, Boolean(error));
  el('file-preview').hidden = !file || Boolean(error);
  setUploadStep(file && !error ? 2 : 1);
  if (file && !error) {
    previewUrl = URL.createObjectURL(file);
    const target = /\.(jpe?g|png)$/i.test(file.name) ? image : video;
    target.src = previewUrl; target.hidden = false;
    el('preview-note').textContent = 'Preview only. Analysis starts when you choose Ingest & Analyze Footage.';
  }
}
el('preview-video').addEventListener('error', () => {
  if (previewUrl) el('preview-note').textContent = 'Your browser cannot preview this video. You can still try analyzing it.';
});
el('clear-file').addEventListener('click', () => {el('file').value = ''; fileSelected(); el('file').focus();});
const presets = document.querySelectorAll('[name=preset]');
presets.forEach(input => input.addEventListener('change', () => {el('limit').value = input.value;}));
el('limit').addEventListener('input', () => presets.forEach(input => {input.checked = input.value === el('limit').value;}));
window.addEventListener('pagehide', () => {if (previewUrl) URL.revokeObjectURL(previewUrl); if (eventSource) eventSource.close();});
el('file').addEventListener('change', fileSelected);
el('mode').addEventListener('change', fileSelected);
el('dropzone').addEventListener('dragover', event => {event.preventDefault(); el('dropzone').classList.add('dragover');});
el('dropzone').addEventListener('dragleave', () => el('dropzone').classList.remove('dragover'));
el('dropzone').addEventListener('drop', event => {
  event.preventDefault(); el('dropzone').classList.remove('dragover');
  if (uploading) return;
  if (event.dataTransfer.files.length !== 1) {message('Choose one recording at a time.', true); return;}
  el('file').files = event.dataTransfer.files;
  fileSelected();
});
el('upload').addEventListener('submit', event => {
  event.preventDefault();
  if (uploading || activeJob || !online) return;
  const file = el('file').files[0], error = uploadError(file, el('mode').value);
  if (error) {message(error, true); return;}
  uploading = true; updateStart(); setUploadStep(2);
  const controlIds = ['file', 'mode', 'limit', 'clear-file', 'stride', 'profile', 'device', 'vehicle-crops', 'case-select', 'camera-label', 'recorded-start', 'clock-offset'];
  for (const id of controlIds) if (el(id)) el(id).disabled = true;
  presets.forEach(input => {input.disabled = true;});
  el('progress').value = 0; el('progress').hidden = false;
  message('Hashing SHA-256 & uploading to local workspace…');
  const request = new XMLHttpRequest();
  request.open('POST', '/api/jobs');
  request.setRequestHeader('Content-Type', 'application/octet-stream');
  const dot = file.name.lastIndexOf('.');
  const ext = dot > 0 ? file.name.slice(dot).toLowerCase() : '';
  request.setRequestHeader('X-File-Extension', ext);
  request.setRequestHeader('X-Analysis-Mode', el('mode').value);
  request.setRequestHeader('X-Frame-Limit', el('limit').value);
  request.setRequestHeader('X-Original-Filename', file.name.replace(/[^\x20-\x7e]/g, '_').slice(0, 160));
  if (el('stride')) request.setRequestHeader('X-Frame-Stride', el('stride').value);
  if (el('profile')) request.setRequestHeader('X-Reasoning-Profile', el('profile').value);
  if (el('device')) request.setRequestHeader('X-Device', el('device').value);
  if (el('vehicle-crops')) request.setRequestHeader('X-Vehicle-Crops', el('vehicle-crops').checked ? '1' : '0');
  if (el('case-select')?.value) request.setRequestHeader('X-Case-Id', el('case-select').value);
  if (el('camera-label')?.value.trim()) request.setRequestHeader('X-Camera-Label', el('camera-label').value.trim().replace(/[^\x20-\x7e]/g, '').slice(0, 100));
  if (el('recorded-start')?.value.trim()) request.setRequestHeader('X-Recorded-Start', el('recorded-start').value.trim().slice(0, 40));
  if (el('clock-offset')?.value) request.setRequestHeader('X-Clock-Offset', el('clock-offset').value);
  request.upload.onprogress = e => {if (e.lengthComputable) {el('progress').value = e.loaded / e.total * 100; if (e.loaded === e.total) message('Upload received. Starting your analysis…');}};
  request.onload = () => {
    try {
      const result = JSON.parse(request.responseText);
      if (request.status !== 202) throw Error(result.error || 'Upload failed. Please try again.');
      activeJob = true;
      setUploadStep(3);
      message('Your analysis has started. Opening your Workspace dashboard…');
      navigateTo('history');
    } catch (error) {message(error.message, true);}
  };
  request.onerror = () => message('The connection was interrupted. Check your workspace before retrying; the server may have received the upload.', true);
  request.onloadend = () => {
    uploading = false; el('progress').hidden = true;
    for (const id of controlIds) if (el(id)) el(id).disabled = false;
    presets.forEach(input => {input.disabled = false;});
    updateStart(); refresh(); loadSystemStatus(); loadCases();
  };
  request.send(file);
});

function initRouterAndTheme() {
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-nav], [data-go]');
    if (!link) return;
    const route = link.dataset.nav || link.dataset.go;
    if (route) {
      event.preventDefault();
      navigateTo(route);
    }
  });
  const syncFromHash = () => {
    const raw = (window.location.hash || '').replace(/^#/, '');
    navigateTo(raw || 'overview', false);
  };
  window.addEventListener('hashchange', syncFromHash);
  syncFromHash();

  const applyTheme = mode => {
    const isLight = mode === 'light';
    if (isLight) document.documentElement.setAttribute('data-theme', 'light');
    else document.documentElement.removeAttribute('data-theme');
    const toggle = el('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(isLight));
      toggle.textContent = isLight ? '☾ Dark' : '☀ Light';
    }
  };
  try {
    applyTheme(localStorage.getItem('fieldnote_theme') || 'dark');
  } catch {
    applyTheme('dark');
  }
  const toggleBtn = el('theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      applyTheme(next);
      try {localStorage.setItem('fieldnote_theme', next);} catch {}
    });
  }

  document.querySelectorAll('[data-copy]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const text = btn.dataset.copy;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = 'Copied ✓';
        setTimeout(() => {btn.textContent = 'Copy';}, 1800);
      } catch {
        btn.textContent = 'Select manually';
        setTimeout(() => {btn.textContent = 'Copy';}, 1800);
      }
    });
  });

  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const heroCard = el('hero-visual');
  if (heroCard && !reducedMotion) {
    heroCard.addEventListener('mousemove', e => {
      const rect = heroCard.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width - 0.5;
      const py = (e.clientY - rect.top) / rect.height - 0.5;
      heroCard.style.setProperty('--rx', `${(-py * 5).toFixed(2)}deg`);
      heroCard.style.setProperty('--ry', `${(px * 6).toFixed(2)}deg`);
    });
    heroCard.addEventListener('mouseleave', () => {
      heroCard.style.setProperty('--rx', '0deg');
      heroCard.style.setProperty('--ry', '-1.5deg');
    });
  }
}

initRouterAndTheme();
loadDemo();
loadSystemStatus();
loadCases();
loadAudit();
updateModeUI();
updateStart();
refresh();
connectEvents();
setInterval(() => {
  if (!eventSource || eventSource.readyState === 2) {
    refresh();
    if (!eventSource) connectEvents();
  }
}, 3000);

function addEvaluation(card,job){
const details=node('details','');details.append(node('summary','Evaluate against reviewed annotations'));
const help=node('p','Choose annotations JSON for this exact run. Scores compare emitted events and confirmed relations with your labels; they do not measure every detection.');
const label=node('label','Annotations JSON');const input=document.createElement('input');input.type='file';input.accept='.json';label.append(input);
const output=node('pre','');output.setAttribute('role','status');output.style.whiteSpace='pre-wrap';
const download=node('button','Download metrics');download.type='button';download.hidden=true;let metrics=null;
download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(metrics,null,2)],{type:'application/json'}));const link=node('a','');link.href=url;link.download='metrics-'+job.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
input.onchange=async()=>{const file=input.files[0];if(!file)return;download.hidden=true;if(file.size>262144){output.textContent='Annotations must be smaller than 256 KB.';return;}input.disabled=true;output.textContent='Evaluating…';try{const response=await fetch('/api/jobs/'+job.id+'/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:await file.text()});const result=await response.json();if(!response.ok)throw Error(result.error||'Evaluation failed');metrics=result;output.textContent=JSON.stringify(result,null,2);download.hidden=false;refresh();}catch(error){output.textContent=error.message;}finally{input.disabled=false;}};
details.append(help,label,output,download);card.append(details);
}
