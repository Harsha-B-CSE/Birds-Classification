(function () {
    "use strict";

    function getCookie(name) {
        const match = document.cookie.match(new RegExp("(^|;\\s*)" + name + "=([^;]*)"));
        return match ? decodeURIComponent(match[2]) : "";
    }

    function readableName(value) {
        if (!value) return "Unknown species";
        return value.split(".").pop().replace(/_/g, " ");
    }

    function renderTop5(rows) {
        const container = document.getElementById("top5-rows-container");
        if (!container) return;
        container.replaceChildren();
        (rows || []).forEach((row) => {
            const item = document.createElement("div");
            item.className = "top5-row";
            const labels = document.createElement("div");
            labels.className = "top5-labels";
            const name = document.createElement("span");
            name.textContent = readableName(row.display_name || row.class);
            const pct = document.createElement("span");
            pct.className = "top5-pct";
            pct.textContent = Number(row.percentage || 0).toFixed(1) + "%";
            labels.append(name, pct);
            const track = document.createElement("div");
            track.className = "meter-track";
            const fill = document.createElement("div");
            fill.className = "meter-fill";
            track.appendChild(fill);
            item.append(labels, track);
            container.appendChild(item);
            window.setTimeout(() => {
                fill.style.width = Math.max(1, Number(row.percentage || 0)) + "%";
            }, 60);
        });
    }

    window.FeatheredUI = { getCookie, readableName, renderTop5 };
})();
