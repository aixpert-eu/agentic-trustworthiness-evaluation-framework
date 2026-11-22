// PAGE BEHAVIOUR
// Builds the cards, filters and metric detail dialog from catalogue-data.js.
// Most text edits belong in index.html or catalogue-data.js.
// Display-name and evidence-label overrides are grouped near the top below.

const $ = id => document.getElementById(id);
let category = '',
  visible = [],
  current = null;
const el = (tag, text, className) => {
  const n = document.createElement(tag);
  n.textContent = text;
  if (className) n.className = className;
  return n
};
const names = {
  bleu: 'BLEU',
  rouge_l: 'ROUGE-L',
  meteor: 'METEOR',
  bertscore: 'BERTScore',
  ece: 'Expected calibration error',
  pass_at_k_agent: 'Pass@k (agent)',
  pass_hat_k_agent: 'Pass^k (agent)',
  judge_elo: 'Judge Elo'
};
const evidenceTypes = {
  common: 'Output + reference / checks',
  trajectory: 'Tool-call trace',
  reflective: 'Trials within one run',
  governance: 'Governance events + labeled cases',
  efficiency: 'Resource measurements + budget',
  human_centricity: 'Participant responses',
  probabilistic: 'Probabilities / confidence data',
  aggregate: 'Repeated runs / case sets',
  meta_evaluation: 'Evaluation scores / reference data'
};
const ioAggregate = ['unsafe_refusal_rate', 'safe_refusal_rate', 'abstention_rate', 'judge_elo'];
metrics.forEach((m, i) => {
  m.number = i + 1;
  m.name = names[m.id] || m.name;
  m.agent = m.cat === 'meta_evaluation' ? 'Evaluation process' : m.cat === 'efficiency' ?
    'System measurements' : m.cat === 'human_centricity' ? 'Human experience' : m.cat ===
    'governance' ? 'System measurements' : m.cat === 'common' || m.cat === 'probabilistic' ||
    ioAggregate.includes(m.id) ? 'Single-agent input/output' : 'Single-agent behaviour';
  m.evidence = evidenceTypes[m.cat];
  m.human =
    'No human responses required by the scorer; reference labels or checks may be prepared by people.';
  if (m.id === 'bias_language') {
    m.evidence = 'Output + bias classifier';
  }
  if (m.id === 'schema_violation_rate') m.evidence = 'Batch of structured outputs';
  if (m.id === 'judge_human_agreement') {
    m.evidence = 'Judge scores + human ratings';
    m.human = 'Human reference ratings required; scoring is automated.';
  }
  if (m.cat === 'human_centricity') m.human =
    'Human participant responses required; scoring is automated.';
  if (m.id === 'inter_rater_reliability') {
    m.evidence = 'Paired rater labels';
    m.human =
      'Requires two sets of rater labels; the scorer does not require that raters are human.';
  }
  if (m.id === 'perplexity') m.evidence = 'Token log-probabilities';
  if (m.id === 'ece') m.evidence = 'Confidence + correctness labels';
  if (m.id === 'judge_elo') m.evidence = 'Pairwise comparison outcomes';
  if (m.id === 'unsafe_action_rate') m.evidence = 'Action trace + safety labels';
  if (m.id === 'log_completeness') m.evidence = 'Tool calls + tool results';
  if (m.id === 'prompt_injection_resistance') m.summary =
    'Checks whether a governance wrapper detects prompt-injection attempts against labeled cases.';
  if (m.id === 'help_seeking') m.summary =
    'Measures how often a supplied detector identifies help-seeking, compared with an acceptable rate window.';
  if (m.id === 'abstention_rate') m.summary =
    'Measures how often a supplied detector identifies abstention, compared with an acceptable rate window.';
});
[...new Set(metrics.map(m => m.evidence))].sort().forEach(v => {
  const o = el('option', v);
  o.value = v;
  $('evidence').append(o)
});
$('stats').innerHTML =
  `<b>${metrics.length}</b> IMPLEMENTED METRICS &nbsp; · &nbsp; <b>${dimensions.length}</b> DIMENSIONS &nbsp; · &nbsp; <b>${Object.keys(categories).length}</b> CATEGORIES`;

function categoryButton(key, name, color) {
  const wrap = el('span', '', 'category-wrap'),
    b = el('button', '', 'filter');
  const tip = el('span', categoryHelp[key], 'category-tooltip');
  tip.id = 'category-help-' + (key || 'all');
  tip.setAttribute('role', 'tooltip');
  b.type = 'button';
  b.dataset.category = key;
  b.setAttribute('aria-pressed', String(key === category));
  b.setAttribute('aria-describedby', tip.id);
  if (color) {
    const swatch = el('span', '', 'swatch');
    swatch.style.setProperty('--color', color);
    b.append(swatch)
  }
  b.append(document.createTextNode(name));
  b.onclick = () => {
    category = key;
    render()
  };
  wrap.addEventListener('pointerenter', () => wrap.classList.remove('dismissed'));
  b.addEventListener('focus', () => wrap.classList.remove('dismissed'));
  b.addEventListener('keydown', e => {
    if (e.key === 'Escape') wrap.classList.add('dismissed')
  });
  wrap.append(b, tip);
  $('categories').append(wrap);
}
categoryButton('', 'All categories');
Object.entries(categories).forEach(([k, v]) => categoryButton(k, v[0], v[1]));
dimensions.forEach(d => {
  const o = el('option', d.name);
  o.value = d.name;
  $('dimension').append(o);
  const detail = el('details', '');
  detail.append(el('summary', d.name), el('p', d.description));
  $('dimensions').append(detail)
});

function render() {
  $('category-explanation').textContent = categoryHelp[category];
  const q = $('search').value.toLowerCase().trim(),
    dim = $('dimension').value,
    scope = $('scope').value,
    agent = $('agent').value,
    evidence = $('evidence').value;
  visible = metrics.filter(m => (!category || m.cat === category) && (!agent || m.agent ===
    agent) && (!evidence || m.evidence === evidence) && (!dim || m.dims.includes(dim)) && (!
      scope || (scope === 'single' ? m.single : !m.single)) && (!q || [m.name, m.id, m.summary, m
      .agent, m.evidence, ...m.requires, ...m.dims
    ].join(' ').toLowerCase().includes(q)));
  $('grid').replaceChildren();
  visible.forEach(m => {
    const b = el('button', '', 'cell');
    b.style.setProperty('--bg', categories[m.cat][1]);
    b.setAttribute('aria-label', m.name + ' — ' + categories[m.cat][0]);
    const row = el('span', '', 'row');
    row.append(el('span', String(m.number).padStart(2, '0')), el('span', categories[m.cat][0]));
    b.append(row, el('span', m.name, 'name'), el('span', m.agent, 'agent-label'), el('span', m
      .evidence, 'type'));
    b.onclick = () => show(m);
    $('grid').append(b)
  });
  $('empty').hidden = visible.length > 0;
  $('empty').textContent = agent === 'Multi-agent-specific' ?
    'No multi-agent-specific evaluator is included in the committed framework. Local-only benchmark additions are excluded from this catalogue.' :
    'No metrics match these filters. Try another dimension or reset the filters.';
  $('count').textContent = `${visible.length} / ${metrics.length} metrics`;
  document.querySelectorAll('[data-category]').forEach(b => b.setAttribute('aria-pressed', String(b
    .dataset.category === category)))
}

function show(m) {
  current = m;
  $('metric-number').textContent =
    `Metric ${String(m.number).padStart(2,'0')} / ${categories[m.cat][0]}`;
  $('metric-title').textContent = m.name;
  $('description').textContent = m.summary;
  $('tags').replaceChildren(...m.dims.map(d => el('span', d, 'tag')));
  $('facts').replaceChildren();
  for (const [title, value] of [
      ['Agent scope', m.agent],
      ['Evidence needed', m.evidence],
      ['Human involvement', m.human],
      ['Registration', m.registered ? 'Registered metric' : 'Implemented; not registered'],
      ['Harness support', !m.registered ?
        'Direct invocation only; not selected by the automatic harness' : m.single ?
        'One captured trajectory; subject to required fields and dependencies' :
        'Separate data / orchestration required'
      ],
      ['Required task fields', m.requires.length ? m.requires.join(', ') :
        'No extra TaskDefinition fields declared; see method description for input data.'
      ],
      ['Entry point', m.entry],
      ['Registry notes', m.notes || 'No additional registry notes.']
    ]) {
    const div = el('div', '');
    div.append(el('dt', title), el('dd', value));
    $('facts').append(div)
  }
  $('documentation').textContent = m.doc || m.summary;
  $('technical').textContent = [m.detail, m.method].filter(Boolean).join('\n\n');
  $('source').textContent = m.path + ' · ' + m.id;
  const i = visible.indexOf(m);
  $('position').textContent = `${i+1} of ${visible.length}`;
  $('previous').disabled = i <= 0;
  $('next').disabled = i >= visible.length - 1;
  if (!$('detail').open) $('detail').showModal();
  $('detail').scrollTop = 0;
}
$('previous').onclick = () => show(visible[visible.indexOf(current) - 1]);
$('next').onclick = () => show(visible[visible.indexOf(current) + 1]);
$('close').onclick = () => $('detail').close();
$('detail').addEventListener('click', e => {
  if (e.target === $('detail')) {
    const r = e.target.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
      e.target.close()
  }
});
['search', 'dimension', 'scope', 'agent', 'evidence'].forEach(id => $(id).addEventListener(id ===
  'search' ? 'input' : 'change', render));
$('reset').onclick = () => {
  category = '';
  $('search').value = '';
  $('dimension').value = '';
  $('scope').value = '';
  $('agent').value = '';
  $('evidence').value = '';
  render()
};
render();
