'use strict';
const el = id => document.getElementById(id);
const node = (tag, value, className = '') => {
  const element = document.createElement(tag);
  element.textContent = value;
  element.className = className;
  return element;
};
let uploading = false, activeJob = false, online = false, refreshing = false;
const cards = new Map();
const statusNames = {processing: 'Analyzing', cancelling: 'Stopping', completed: 'Ready to review', failed: 'Needs attention', cancelled: 'Stopped', interrupted: 'Interrupted'};

function uploadError(file, mode) {
  if (!file || !file.size) return 'Choose a video or image that is not empty.';
  if (file.size > 512 * 1024 * 1024) return 'This file is too large. Choose a file up to 512 MB.';
  const extension = file.name.split('.').pop().toLowerCase();
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
async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Something went wrong. Please try again.');
  return result;
}
async function loadDemo() {
  try {
    const data = await jsonRequest('/assets/demo-result.json');
    const [width, height] = data.image_size;
    const boxes = new Map();
    for (const region of data.regions) {
      const [x1, y1, x2, y2] = region.box;
      const box = node('div', '', 'region');
      Object.assign(box.style, {left: `${x1 / width * 100}%`, top: `${y1 / height * 100}%`, width: `${(x2 - x1) / width * 100}%`, height: `${(y2 - y1) / height * 100}%`});
      box.append(node('span', region.label));
      el('demo-boxes').append(box);
      boxes.set(region.id, box);
    }
    const relationship = data.relations[0];
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
el('show-boxes').addEventListener('change', () => {el('demo-boxes').hidden = !el('show-boxes').checked;});

function jobCard(job) {
  const card = node('article', '', 'job');
  card.dataset.jobId = job.id;
  const heading = node('div', '', 'job-heading');
  const title = node('div', '');
  title.append(node('span', statusNames[job.status] || job.status, 'status ' + job.status), node('strong', job.mode === 'scene' ? 'Scene analysis' : 'Plate reading'));
  const date = node('time', new Date(job.created).toLocaleString());
  date.dateTime = job.created;
  heading.append(title, date);
  card.append(heading);
  if (job.status === 'processing' || job.status === 'cancelling') {
    const steps = node('div', '', 'processing-steps');
    steps.append(node('span', '✓ Upload ready'), node('span', job.status === 'cancelling' ? 'Stopping analysis…' : job.frames_processed ? '● Understanding your footage' : '● Preparing models', 'current'), node('span', '○ Review findings'));
    card.append(steps);
  }
  card.append(node('p', `${job.frames_processed || 0} sampled frames processed · ${(job.bytes / 1048576).toFixed(1)} MB`));
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
  card.append(actions);
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

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const jobs = await jsonRequest('/api/jobs');
    online = true;
    activeJob = jobs.some(j => ['processing', 'cancelling'].includes(j.status));
    el('connection').textContent = activeJob ? 'Analysis in progress' : 'Local workspace connected';
    el('connection').className = 'connection online';
    // Keep unchanged cards mounted so polling preserves annotation inputs and review controls.
    const liveIds = new Set(jobs.map(j => j.id));
    for (const [id, entry] of cards) if (!liveIds.has(id)) {entry.card.remove(); cards.delete(id);}
    const empty = el('jobs').querySelector('.empty-state');
    if (empty && jobs.length) empty.remove();
    let previous = null;
    for (const job of jobs) {
      const signature = JSON.stringify(job);
      let entry = cards.get(job.id);
      if (!entry || entry.signature !== signature) {
        const card = jobCard(job);
        if (entry) {card.querySelectorAll('details').forEach((detail, i) => {detail.open = entry.card.querySelectorAll('details')[i]?.open || false;}); entry.card.replaceWith(card);}
        entry = {signature, card}; cards.set(job.id, entry);
      }
      const next = previous ? previous.nextElementSibling : el('jobs').firstElementChild;
      if (next !== entry.card) el('jobs').insertBefore(entry.card, next);
      previous = entry.card;
    }
    el('jobs').querySelectorAll('[data-retry]').forEach(button => {button.disabled = activeJob || uploading;});
    if (!jobs.length && (!empty || !empty.dataset.ready)) {
      el('jobs').replaceChildren();
      const blank = node('div', '', 'empty-state');
      blank.dataset.ready = 'true';
      const link = node('a', 'Start your first analysis ↗'); link.href = '#analyze';
      blank.append(node('strong', 'Your next discovery starts here.'), node('span', 'Upload a recording and your findings will appear in this space.'), document.createElement('br'), link);
      el('jobs').append(blank);
    }
    const announcement = jobs.length ? `${jobs.length} analyses. ${activeJob ? 'One is running.' : 'No analysis is running.'}` : 'Your workspace is ready for its first upload.';
    if (el('job-announcement').textContent !== announcement) el('job-announcement').textContent = announcement;
  } catch {
    online = false;
    el('connection').textContent = 'Disconnected · reconnecting automatically';
    el('connection').className = 'connection error';
  } finally {refreshing = false; updateStart();}
}

function fileSelected() {
  const file = el('file').files[0];
  el('file-title').textContent = file ? file.name : 'Drop your video or image here';
  el('file-caption').textContent = file ? `${(file.size / 1048576).toFixed(1)} MB · click to choose a different file` : 'or browse files on your computer';
  const error = file ? uploadError(file, el('mode').value) : '';
  message(error, Boolean(error));
}
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
  uploading = true; updateStart();
  for (const id of ['file', 'mode', 'limit']) el(id).disabled = true;
  el('progress').value = 0; el('progress').hidden = false;
  message('Uploading to your local workspace…');
  const request = new XMLHttpRequest();
  request.open('POST', '/api/jobs');
  request.setRequestHeader('Content-Type', 'application/octet-stream');
  request.setRequestHeader('X-File-Extension', '.' + file.name.split('.').pop().toLowerCase());
  request.setRequestHeader('X-Analysis-Mode', el('mode').value);
  request.setRequestHeader('X-Frame-Limit', el('limit').value);
  request.upload.onprogress = e => {if (e.lengthComputable) {el('progress').value = e.loaded / e.total * 100; if (e.loaded === e.total) message('Upload received. Starting your analysis…');}};
  request.onload = () => {
    try {
      const result = JSON.parse(request.responseText);
      if (request.status !== 202) throw Error(result.error || 'Upload failed. Please try again.');
      activeJob = true;
      message('Your analysis has started. Follow its progress in your workspace below.');
      el('history').scrollIntoView({behavior: 'auto', block: 'start'});
    } catch (error) {message(error.message, true);}
  };
  request.onerror = () => message('The connection was interrupted. Check your workspace before retrying; the server may have received the upload.', true);
  request.onloadend = () => {
    uploading = false; el('progress').hidden = true;
    for (const id of ['file', 'mode', 'limit']) el(id).disabled = false;
    updateStart(); refresh();
  };
  request.send(file);
});
loadDemo();
updateStart();
refresh();
setInterval(refresh, 3000);

function addEvaluation(card,job){
const details=node('details','');details.append(node('summary','Evaluate against reviewed annotations'));
const help=node('p','Choose annotations JSON for this exact run. Scores compare emitted events and confirmed relations with your labels; they do not measure every detection.');
const label=node('label','Annotations JSON');const input=document.createElement('input');input.type='file';input.accept='.json';label.append(input);
const output=node('pre','');output.setAttribute('role','status');output.style.whiteSpace='pre-wrap';
const download=node('button','Download metrics');download.type='button';download.hidden=true;let metrics=null;
download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(metrics,null,2)],{type:'application/json'}));const link=node('a','');link.href=url;link.download='metrics-'+job.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
input.onchange=async()=>{const file=input.files[0];if(!file)return;download.hidden=true;if(file.size>262144){output.textContent='Annotations must be smaller than 256 KB.';return;}input.disabled=true;output.textContent='Evaluating…';try{const response=await fetch('/api/jobs/'+job.id+'/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:await file.text()});const result=await response.json();if(!response.ok)throw Error(result.error||'Evaluation failed');metrics=result;output.textContent=JSON.stringify(result,null,2);download.hidden=false;}catch(error){output.textContent=error.message;}finally{input.disabled=false;}};
details.append(help,label,output,download);card.append(details);
}
