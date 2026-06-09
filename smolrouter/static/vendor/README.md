# Vendored third-party browser assets

These files are upstream, third-party, generated/minified assets checked in
verbatim so the Web UI makes **zero external browser requests** (SmolRouter is
a LAN/offline-first router). They are **not** first-party code: do not refactor
or hand-edit them. To update, re-download the pinned release from the upstream
source and update the version/hash below.

For the same reason, these exact files are the only assets excluded from
SonarCloud analysis (see `sonar-project.properties`). First-party templates,
CSS, and application code remain fully in scope.

## chart.umd.min.js

- Library: Chart.js
- Version: 4.5.1
- Source: https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js
- Integrity: sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ
- License: MIT (https://github.com/chartjs/Chart.js/blob/master/LICENSE.md)
- Used by: `smolrouter/templates/performance.html`
