(function () {
    "use strict";

    const dashboard = document.getElementById("predict-dashboard");
    if (!dashboard) return;

    const predictUrl = dashboard.dataset.predictUrl;
    const speciesUrl = dashboard.dataset.speciesUrl;
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("bird-file-input");
    const form = document.getElementById("upload-form");
    const previewBox = document.getElementById("preview-box");
    const previewImg = document.getElementById("image-preview");
    const scanBeam = document.getElementById("scan-beam");
    const pipeline = document.getElementById("pipeline-status-card");
    const resultCard = document.getElementById("result-card");
    const errorCard = document.getElementById("error-card");
    const MAX_BYTES = 15 * 1024 * 1024;
    const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

    let processing = false;
    let timers = [];

    function setStep(key, state, detail) {
        const step = document.getElementById("step-" + key);
        const text = document.getElementById("step-" + key + "-detail");
        if (step) step.dataset.state = state;
        if (text) text.textContent = detail;
    }

    function resetSteps() {
        setStep("detector", "pending", "Queued");
        setStep("classifier", "pending", "Queued");
        setStep("ood", "pending", "Queued");
    }

    function clearTimers() {
        timers.forEach((timer) => window.clearTimeout(timer));
        timers = [];
    }

    function setBusy(busy) {
        processing = busy;
        dropzone.classList.toggle("busy", busy);
    }

    function showError(message) {
        errorCard.textContent = message;
        errorCard.style.display = "block";
        resultCard.style.display = "none";
    }

    function beginProgress() {
        errorCard.style.display = "none";
        resultCard.style.display = "none";
        pipeline.style.display = "block";
        scanBeam.style.display = "block";
        resetSteps();
        setStep("detector", "active", "Scanning for avian features...");
        timers.push(window.setTimeout(() => {
            setStep("detector", "done", "Bird presence confirmed");
            setStep("classifier", "active", "Extracting plumage features...");
        }, 350));
        timers.push(window.setTimeout(() => {
            setStep("classifier", "done", "Species logits generated");
            setStep("ood", "active", "Computing Helmholtz free energy...");
        }, 950));
    }

    function finishRejected(data, httpStatus) {
        pipeline.style.display = "block";
        scanBeam.style.display = "none";
        const status = data.status || "REJECTED";
        if (status === "REJECTED_NOT_A_BIRD") {
            setStep("detector", "failed", "No avian subject detected");
            setStep("classifier", "skipped", "Skipped");
            setStep("ood", "skipped", "Skipped");
        } else {
            setStep("detector", "done", "Bird presence confirmed");
            setStep("classifier", "done", "Species logits generated");
            setStep("ood", "failed", status.replace("REJECTED_", "").replace(/_/g, " ").toLowerCase());
        }
        showError(data.error || "The specimen could not be validated (HTTP " + httpStatus + ").");
    }

    function renderSuccess(data) {
        scanBeam.style.display = "none";
        const checks = data.diagnostics || {};
        const birdProb = ((checks.check_1 || {}).bird_probability || 0) * 100;
        const energy = ((checks.check_3 || {}).meta || {}).energy_score;
        setStep("detector", "done", "Bird probability " + birdProb.toFixed(1) + "%");
        setStep("classifier", "done", "Top-1 " + FeatheredUI.readableName(data.display_name || data.class));
        setStep("ood", "done", energy === undefined ? "In-distribution" : "Energy " + Number(energy).toFixed(2) + " - accepted");

        const confidence = Number(data.probability || 0) * 100;
        document.getElementById("result-species-name").textContent =
            FeatheredUI.readableName(data.display_name || data.class);
        document.getElementById("result-scientific-name").textContent = "Resolving taxonomy...";
        document.getElementById("result-confidence").textContent = confidence.toFixed(1) + "% match";
        document.getElementById("result-iucn").hidden = true;
        FeatheredUI.renderTop5(data.top5);

        resultCard.style.display = "block";
        FeatheredDossier.load(speciesUrl, data.class);
    }

    function renderFailure(data, httpStatus) {
        clearTimers();
        if (data.status && data.status !== "SUCCESS") {
            finishRejected(data, httpStatus);
        } else {
            setStep("detector", "failed", data.error || "Request failed");
            setStep("classifier", "skipped", "Skipped");
            setStep("ood", "skipped", "Skipped");
            showError(data.error || "Prediction failed. Please try again.");
        }
    }

    function submitImage(file) {
        if (processing) return;
        if (!ALLOWED_TYPES.includes(file.type)) {
            showError("Unsupported file type. Upload a JPEG, PNG, or WEBP photograph.");
            return;
        }
        if (file.size > MAX_BYTES) {
            showError("This image is larger than 15 MB. Please choose a smaller photograph.");
            return;
        }

        setBusy(true);
        beginProgress();
        const body = new FormData();
        body.append("image", file);
        if (window.FEATHERED_DEVICE_ID) body.append("device_id", window.FEATHERED_DEVICE_ID);

        fetch(predictUrl, {
            method: "POST",
            headers: { "X-CSRFToken": FeatheredUI.getCookie("csrftoken") },
            body: body,
        })
            .then(async (response) => {
                const data = await response.json().catch(() => ({}));
                return { ok: response.ok, status: response.status, data: data };
            })
            .then((payload) => {
                clearTimers();
                if (payload.ok && payload.data.status === "SUCCESS") {
                    renderSuccess(payload.data);
                } else {
                    renderFailure(payload.data, payload.status);
                }
            })
            .catch(() => {
                clearTimers();
                setStep("detector", "failed", "Network error");
                setStep("classifier", "skipped", "Skipped");
                setStep("ood", "skipped", "Skipped");
                showError("Could not reach the identification service. Check your connection and try again.");
            })
            .finally(() => setBusy(false));
    }

    function handleFile(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImg.src = event.target.result;
            previewBox.style.display = "block";
            submitImage(file);
        };
        reader.readAsDataURL(file);
    }

    function resetUploader() {
        clearTimers();
        form.reset();
        previewBox.style.display = "none";
        pipeline.style.display = "none";
        resultCard.style.display = "none";
        errorCard.style.display = "none";
        scanBeam.style.display = "none";
        resetSteps();
        setBusy(false);
    }

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInput.click();
        }
    });
    form.addEventListener("submit", (event) => event.preventDefault());
    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("drag-over");
        if (event.dataTransfer.files.length) {
            fileInput.files = event.dataTransfer.files;
            handleFile(event.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener("change", (event) => {
        if (event.target.files.length) handleFile(event.target.files[0]);
    });
    document.getElementById("btn-identify-another").addEventListener("click", resetUploader);
})();
