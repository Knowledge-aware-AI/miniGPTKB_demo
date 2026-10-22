// Shared script of the disambiguated miniGPTKB HTML report (index and entity pages).
(function () {
    // Theme toggle -- same storage key as the miniGPTKB control panel (same
    // origin), so a choice made in one place is remembered in the other.
    var root = document.documentElement;
    var STORAGE_KEY = 'miniGPTKB.theme';
    var stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (error) {}
    if (stored === 'light' || stored === 'dark') root.setAttribute('data-theme', stored);

    function isDark() {
        var explicit = root.getAttribute('data-theme');
        return explicit === 'dark' || (!explicit && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }

    function escapeHtml(value) {
        return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    document.addEventListener('DOMContentLoaded', function () {
        var opts = document.querySelectorAll('#themeToggle button');
        function paint() {
            var choice = isDark() ? 'dark' : 'light';
            opts.forEach(function (opt) { opt.classList.toggle('active', opt.dataset.themeChoice === choice); });
        }
        opts.forEach(function (opt) {
            opt.addEventListener('click', function () {
                root.setAttribute('data-theme', opt.dataset.themeChoice);
                try { localStorage.setItem(STORAGE_KEY, opt.dataset.themeChoice); } catch (error) {}
                paint();
                if (window.__miniGPTKBUpdateGraphTheme) window.__miniGPTKBUpdateGraphTheme();
            });
        });
        paint();

        // Knowledge graph (index page only).
        var container = document.getElementById('graphCanvas');
        var nodesEl = document.getElementById('graph-nodes-data');
        var edgesEl = document.getElementById('graph-edges-data');
        if (container && nodesEl && edgesEl && typeof vis !== 'undefined') {
            var colors = function () {
                return isDark()
                    ? { font: '#e5eaef', edge: '#3a4a58', accent: '#4db8f0' }
                    : { font: '#1f2937', edge: '#c7cdd6', accent: '#0077cc' };
            };
            var network = new vis.Network(container, {
                nodes: new vis.DataSet(JSON.parse(nodesEl.textContent)),
                edges: new vis.DataSet(JSON.parse(edgesEl.textContent))
            }, {
                // Labels sit inside pastel nodes, so they stay dark in both themes.
                nodes: { font: { size: 13, face: 'Arial' }, margin: 8, borderWidth: 1.5, shapeProperties: { borderRadius: 6 } },
                edges: {
                    color: { color: colors().edge, highlight: colors().accent, hover: colors().accent },
                    smooth: { type: 'dynamic' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } }
                },
                physics: { solver: 'barnesHut', stabilization: { iterations: 150 } },
                interaction: { hover: true, tooltipDelay: 100, dragView: true, zoomView: true }
            });
            network.once('stabilizationIterationsDone', function () { network.setOptions({ physics: false }); });
            network.on('doubleClick', function (params) {
                if (params.nodes && params.nodes.length) window.location.href = params.nodes[0] + '.html';
            });
            window.__miniGPTKBUpdateGraphTheme = function () {
                var c = colors();
                network.setOptions({
                    edges: { color: { color: c.edge, highlight: c.accent, hover: c.accent } }
                });
            };
        }

        // Entity index filter.
        var input = document.getElementById('entitySearch');
        if (input) {
            var items = Array.prototype.slice.call(document.querySelectorAll('#entityList li'));
            var countEl = document.getElementById('entityCount');
            var trueTotal = countEl ? countEl.textContent : String(items.length);
            input.addEventListener('input', function () {
                var query = input.value.trim().toLowerCase();
                var shown = 0;
                items.forEach(function (item) {
                    var match = item.textContent.toLowerCase().indexOf(query) !== -1;
                    item.classList.toggle('hidden', !match);
                    if (match) shown++;
                });
                if (countEl) countEl.textContent = query ? (shown + ' / ' + items.length) : trueTotal;
            });
        }

        // Triples table, rendered on first expand.
        var details = document.getElementById('triplesDetails');
        var dataEl = document.getElementById('triple-data');
        var tbody = document.getElementById('tripleTableBody');
        if (details && dataEl && tbody) {
            var rendered = false;
            var link = function (label, id) {
                if (!id) return escapeHtml(label);
                return '<a href="' + escapeHtml(id) + '.html">' + escapeHtml(label) + '</a><span class="idref">' + escapeHtml(id) + '</span>';
            };
            details.addEventListener('toggle', function () {
                if (!details.open || rendered) return;
                rendered = true;
                // [subject label, subject id, predicate label, predicate id, object label, object id, object type]
                tbody.innerHTML = JSON.parse(dataEl.textContent).map(function (t) {
                    var predicate = escapeHtml(t[2]) + (t[3] ? '<span class="idref">' + escapeHtml(t[3]) + '</span>' : '');
                    var object = t[6] === 'ne' || t[5] ? link(t[4], t[5]) : escapeHtml(t[4]);
                    return '<tr><td>' + link(t[0], t[1]) + '</td><td>' + predicate + '</td><td>' + object +
                        ' <span class="tag">' + escapeHtml(t[6]) + '</span></td></tr>';
                }).join('');
            });
        }
    });
})();
