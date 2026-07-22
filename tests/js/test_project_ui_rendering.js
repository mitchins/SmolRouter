'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  const bodyStart = source.indexOf('{', start);
  let depth = 0;
  for (let index = bodyStart; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`unterminated function ${name}`);
}

function escapeMarkup(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function renderer(templatePath, dependencies) {
  const source = fs.readFileSync(templatePath, 'utf8');
  const context = {
    encodeURIComponent,
    document: {
      createElement() {
        let content = '';
        return {
          set textContent(value) { content = value == null ? '' : String(value); },
          get innerHTML() { return escapeMarkup(content); },
        };
      },
    },
  };
  vm.createContext(context);
  const names = [...dependencies, 'renderProjectCell'];
  vm.runInContext(`${names.map((name) => extractFunction(source, name)).join('\n')}\nthis.render = renderProjectCell;`, context);
  return context.render;
}

const indexRender = renderer('smolrouter/templates/index.html', ['escapeHtml', 'escapeAttribute', 'renderFilterButton']);
const clientRender = renderer('smolrouter/templates/client_dashboard.html', ['escapeHtml']);

{
  const html = indexRender({identity_kind: 'facade_key', identity_subject_id: 'team alpha/proj', identity_display_name: 'Team Alpha'});
  assert.match(html, />\s*Team Alpha\s*</);
  assert.equal((html.match(/data-filter-field="project"/g) || []).length, 2, 'friendly name and distinct ID remain project filters');
  assert.equal((html.match(/data-filter-value="team%20alpha%2Fproj"/g) || []).length, 2);
  assert.match(html, /href="\/projects\/team%20alpha%2Fproj"/);
  assert.match(html, /folder_open/);
  assert.match(html, /aria-hidden="true"/);
  assert.match(html, /aria-label="Open project Team Alpha"/);
  assert.match(html, /class="project-primary">[\s\S]*data-filter-field="project"[\s\S]*class="icon-link project-detail-link"[\s\S]*<\/div>\s*<div class="cell-secondary">/);
}

{
  const html = indexRender({identity_kind: 'facade_key', identity_subject_id: 'project-a', identity_display_name: null});
  assert.equal((html.match(/data-filter-field="project"/g) || []).length, 1, 'an ID-only project has one filter control');
  assert.doesNotMatch(html, /data-filter-field="identity"/);
}

{
  const html = indexRender({identity_kind: 'facade_key', identity_subject_id: 'bad/\"id', identity_display_name: '<img src=x onerror=alert(1)>'});
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /href="\/projects\/bad%2F%22id"/);
}

{
  const hostileName = 'Project " onmouseover="alert(1)\' data-attack=\'yes';
  const html = indexRender({identity_kind: 'facade_key', identity_subject_id: 'bad/" onfocus="attack', identity_display_name: hostileName});
  const navigationAnchor = html.match(/<a class="icon-link project-detail-link"[\s\S]*?<\/a>/)[0];
  assert.doesNotMatch(navigationAnchor, /" onmouseover="/);
  assert.doesNotMatch(navigationAnchor, /' data-attack='/);
  assert.match(html, /aria-label="Open project Project &quot; onmouseover=&quot;alert\(1\)&#39; data-attack=&#39;yes"/);
  assert.match(html, /href="\/projects\/bad%2F%22%20onfocus%3D%22attack"/);
}

for (const render of [indexRender, clientRender]) {
  const html = render({identity_kind: 'service_account', identity_subject_id: '<admin>', identity_display_name: 'Ignored'});
  assert.match(html, /service_account:&lt;admin&gt;/);
  assert.match(html, />Identity</);
  assert.doesNotMatch(html, /data-filter-field/);
  assert.doesNotMatch(html, /\/projects\//);
  assert.doesNotMatch(html, />Ignored</);
}

{
  const html = clientRender({identity_kind: 'facade_key', identity_subject_id: 'team/proj', identity_display_name: 'Team Project'});
  assert.match(html, /href="\/projects\/team%2Fproj"/);
  assert.match(html, />Team Project</);
  assert.match(html, />team\/proj</);
  assert.doesNotMatch(html, /token-button/);
  assert.doesNotMatch(html, /folder_open/);
}

{
  const html = clientRender({identity_kind: 'facade_key', identity_subject_id: 'same-id', identity_display_name: 'same-id'});
  assert.equal((html.match(/same-id/g) || []).length, 2, 'matching display name and ID render one visible label plus the URL');
}
