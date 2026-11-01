(function () {
    "use strict";

    const loading = document.getElementById("dossier-loading");
    const empty = document.getElementById("dossier-empty");
    const content = document.getElementById("dossier-content");
    const sourceBadge = document.getElementById("dossier-source");
    const disclaimer = document.getElementById("dossier-disclaimer");
    let requestToken = 0;
    let lifespanChart = null;

    function addLink(container, label, url) {
        if (!url) return;
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = label;
        container.appendChild(link);
    }

    function chip(container, label, value) {
        if (!value) return;
        const item = document.createElement("span");
        item.className = "chip";
        item.textContent = label + ": " + value;
        container.appendChild(item);
    }

    function render(profile) {
        loading.hidden = true;
        const photo = document.getElementById("dossier-photo");
        const caption = document.getElementById("dossier-photo-caption");
        photo.hidden = true;
        caption.hidden = true;
        const hasData = profile && (profile.summary || profile.scientific_name || profile.taxonomy);
        content.hidden = !hasData;
        content.classList.toggle("no-photo", !hasData || !profile.image_url);
        empty.hidden = Boolean(hasData);
        if (!hasData) return;

        document.getElementById("result-scientific-name").textContent =
            profile.scientific_name || "Scientific name unavailable";
        document.getElementById("dossier-summary").textContent =
            profile.summary || "No encyclopedia summary is available.";

        const habitat = document.getElementById("dossier-habitat");
        habitat.textContent = profile.habitat || "";
        habitat.hidden = !profile.habitat;

        const taxonomy = document.getElementById("dossier-taxonomy");
        taxonomy.replaceChildren();
        Object.entries(profile.taxonomy || {}).forEach(([rank, value]) => {
            chip(taxonomy, rank.charAt(0).toUpperCase() + rank.slice(1), value);
        });

        const range = document.getElementById("dossier-range");
        range.replaceChildren();
        (profile.native_range || []).forEach((region) => chip(range, "Range", region));

        const biology = document.getElementById("dossier-biology");
        biology.replaceChildren();
        chip(biology, "2n", profile.chromosome_count_2n);
        if (profile.diet) chip(biology, "Diet", profile.diet);

        const iucn = document.getElementById("result-iucn");
        if (profile.iucn_status) {
            const code = profile.iucn_code || "";
            iucn.textContent = "IUCN: " + profile.iucn_status;
            iucn.dataset.level = ["VU", "NT"].includes(code) ? "warning"
                : ["EN", "CR", "EW", "EX"].includes(code) ? "error" : "success";
            iucn.hidden = false;
        }

        if (profile.image_url) {
            photo.onerror = () => { photo.hidden = true; caption.hidden = true; };
            photo.src = profile.image_url;
            photo.alt = (profile.common_name || "Bird") + " reference photograph";
            photo.hidden = false;
            const credit = profile.image_attribution || "";
            const license = profile.image_license ? " (" + profile.image_license + ")" : "";
            caption.textContent = credit ? "Photo: " + credit + license : "Reference photograph";
            caption.hidden = false;
        }

        const links = document.getElementById("dossier-links");
        links.replaceChildren();
        addLink(links, "Wikipedia", profile.wikipedia_url);
        addLink(links, "iNaturalist", profile.inaturalist_url);
        addLink(links, "GBIF", profile.gbif_url);
        addLink(links, "NCBI Taxonomy", (profile.ncbi || {}).url);
        addLink(links, "GenBank", (profile.ncbi || {}).genbank_url);
        const assembly = (profile.ncbi || {}).reference_assembly;
        if (assembly) {
            const suffix = assembly.chromosomes ? " (" + assembly.chromosomes + " chromosomes)" : "";
            addLink(links, "Reference assembly " + assembly.accession + suffix, assembly.url);
        }
        if (profile.observations_count) {
            const note = document.createElement("span");
            note.className = "dossier-empty";
            note.textContent = Number(profile.observations_count).toLocaleString() + " research-grade observations on iNaturalist";
            links.appendChild(note);
        }

        renderProvenance(profile);
        renderLifespan(profile.lifespan);
    }

    function renderProvenance(profile) {
        const local = ["local-model", "hybrid"].includes(profile.source);
        sourceBadge.hidden = false;
        sourceBadge.textContent = local ? "Local AI dossier" : "Cached scientific sources";
        disclaimer.hidden = !local;
        disclaimer.textContent = local
            ? "Generated offline by " + (profile.model_id || "the local model") + ". Verify field-critical facts."
            : "";
    }

    function renderLifespan(lifespan) {
        const block = document.getElementById("lifespan-block");
        const canvas = document.getElementById("lifespan-chart");
        const wild = (lifespan || {}).wild_years;
        const captive = (lifespan || {}).captive_years;
        const rows = [];
        if (wild) rows.push({ label: "Wild", range: wild, color: "rgba(57, 173, 158, 0.85)" });
        if (captive) rows.push({ label: "Captive", range: captive, color: "rgba(226, 176, 39, 0.85)" });
        block.hidden = rows.length === 0 || typeof window.Chart === "undefined";
        if (block.hidden) return;
        if (lifespanChart) lifespanChart.destroy();
        lifespanChart = new window.Chart(canvas, {
            type: "bar",
            data: {
                labels: rows.map((row) => row.label),
                datasets: [{
                    label: "Years",
                    data: rows.map((row) => row.range),
                    backgroundColor: rows.map((row) => row.color),
                    borderRadius: 6,
                }],
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 600 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => context.raw[0] + "-" + context.raw[1] + " years",
                        },
                    },
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: "rgba(255,255,255,0.08)" },
                        ticks: { color: "#9aa89f" },
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#ece4d4" },
                    },
                },
            },
        });
    }

    function load(speciesUrl, classLabel) {
        const token = ++requestToken;
        loading.hidden = false;
        loading.textContent = "Local model is composing the scientific dossier...";
        empty.hidden = true;
        content.hidden = true;
        sourceBadge.hidden = true;
        disclaimer.hidden = true;
        const poll = (attempt) => {
            fetch(speciesUrl + "?name=" + encodeURIComponent(classLabel))
                .then(async (response) => {
                    if (token !== requestToken) return null;
                    if (response.status === 202) {
                        if (attempt >= 30) throw new Error("Dossier generation timed out");
                        window.setTimeout(() => poll(attempt + 1), 2000);
                        return null;
                    }
                    if (!response.ok) throw new Error("Enrichment unavailable");
                    return response.json();
                })
                .then((profile) => {
                    if (profile && token === requestToken) render(profile);
                })
                .catch(() => {
                    if (token !== requestToken) return;
                    loading.hidden = true;
                    content.hidden = true;
                    empty.hidden = false;
                });
        };
        poll(0);
    }

    window.FeatheredDossier = { load };
})();
