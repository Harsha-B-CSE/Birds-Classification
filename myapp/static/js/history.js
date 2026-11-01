(function () {
    "use strict";

    const root = document.getElementById("history-root");
    if (!root) return;

    const apiUrl = root.dataset.historyApi;
    const list = document.getElementById("history-list");
    const empty = document.getElementById("history-empty");
    const stats = document.getElementById("history-stats");
    const countLabel = document.getElementById("history-count");
    const clearButton = document.getElementById("btn-clear-history");
    const errorCard = document.getElementById("history-error");

    function showError(message) {
        errorCard.textContent = message;
        errorCard.style.display = "block";
    }

    function formatDate(record) {
        const raw = record.created_at;
        let date = raw ? new Date(raw) : null;
        if ((!date || Number.isNaN(date.getTime())) && record.timestamp) {
            date = new Date(Number(record.timestamp) * 1000);
        }
        return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : "Time unavailable";
    }

    function render(records) {
        list.replaceChildren();
        records.sort((a, b) => Number(b.timestamp || 0) - Number(a.timestamp || 0));

        const unique = new Set(records.map((item) => item.species || item.display_name));
        const confidences = records
            .map((item) => Number(item.confidence))
            .filter((value) => Number.isFinite(value));

        document.getElementById("stat-total").textContent = records.length;
        document.getElementById("stat-species").textContent = unique.size;
        document.getElementById("stat-confidence").textContent = confidences.length
            ? (confidences.reduce((sum, value) => sum + value, 0) / confidences.length * 100).toFixed(1) + "%"
            : "0%";
        stats.hidden = records.length === 0;
        empty.hidden = records.length !== 0;
        clearButton.hidden = records.length === 0;
        countLabel.textContent = records.length + (records.length === 1 ? " sighting" : " sightings");

        records.forEach((record) => {
            const item = document.createElement("article");
            item.className = "history-item";

            const body = document.createElement("div");
            const species = document.createElement("div");
            species.className = "history-species";
            species.textContent = FeatheredUI.readableName(record.display_name || record.species);

            const meta = document.createElement("div");
            meta.className = "history-meta";
            const parts = [formatDate(record)];
            if (record.latency_ms) parts.push(Math.round(record.latency_ms) + " ms inference");
            parts.forEach((part) => {
                const span = document.createElement("span");
                span.textContent = part;
                meta.appendChild(span);
            });
            body.append(species, meta);

            const confidence = document.createElement("span");
            confidence.className = "history-confidence";
            confidence.textContent = (Number(record.confidence || 0) * 100).toFixed(1) + "%";
            item.append(body, confidence);
            list.appendChild(item);
        });
    }

    function loadHistory() {
        const deviceId = window.FEATHERED_DEVICE_ID || "";
        if (!deviceId) {
            showError("This browser has no device identity yet. Identify a bird first.");
            return;
        }
        fetch(apiUrl + "?device_id=" + encodeURIComponent(deviceId))
            .then((response) => response.ok ? response.json() : Promise.reject(new Error("Unable to load field log")))
            .then((data) => render(data.history || []))
            .catch(() => showError("The field log could not be loaded. Please refresh and try again."));
    }

    clearButton.addEventListener("click", () => {
        if (clearButton.dataset.confirm !== "true") {
            clearButton.dataset.confirm = "true";
            clearButton.textContent = "Click again to confirm";
            return;
        }
        const deviceId = window.FEATHERED_DEVICE_ID || "";
        fetch(apiUrl + "?device_id=" + encodeURIComponent(deviceId), {
            method: "DELETE",
            headers: { "X-CSRFToken": FeatheredUI.getCookie("csrftoken") },
        })
            .then((response) => response.ok ? response.json() : Promise.reject(new Error("Clear failed")))
            .then(() => {
                clearButton.dataset.confirm = "false";
                clearButton.textContent = "Clear field log";
                render([]);
            })
            .catch(() => showError("The field log could not be cleared. Please try again."));
    });

    loadHistory();
})();
