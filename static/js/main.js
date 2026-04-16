function getStatusClass(status) {
    if (!status) return 'status-stopped';
    const s = status.toLowerCase();
    if (s === 'running') return 'status-running pulse';
    if (s === 'paused') return 'status-paused';
    if (s === 'finished') return 'status-finished';
    if (s === 'error' || s === 'failed') return 'status-error';
    return 'status-stopped';
}

let openMenuProject = null;
let currentProjectDetailsProject = null;

const DEFAULT_LLM_ROLE_EXECUTION = {
    director: 'remote'
};

// Handle clicking outside the dropdown to close it
window.onclick = function (event) {
    if (event.target.closest && event.target.closest('.dropdown')) {
        return;
    }

    if (openMenuProject) {
        const menu = document.getElementById("menu-" + openMenuProject);
        if (menu) menu.classList.remove("show");
        openMenuProject = null;
    }
}

function toggleMenu(projectName) {
    if (openMenuProject === projectName) {
        document.getElementById("menu-" + projectName).classList.remove("show");
        openMenuProject = null;
    } else {
        if (openMenuProject) {
            const oldMenu = document.getElementById("menu-" + openMenuProject);
            if (oldMenu) oldMenu.classList.remove("show");
        }
        document.getElementById("menu-" + projectName).classList.add("show");
        openMenuProject = projectName;
    }
}

function getLogStatusClass(status) {
    if (!status) return '';
    const s = status.toLowerCase();
    if (s.includes('waiting')) return 'log-waiting';
    if (s === 'started') return 'log-started';
    if (s === 'completed') return 'log-completed';
    if (s === 'failed') return 'log-failed';
    return '';
}

function normalizeLlmRoleExecution(raw) {
    const normalized = { ...DEFAULT_LLM_ROLE_EXECUTION };
    if (!raw || typeof raw !== 'object') {
        return normalized;
    }

    for (const role of Object.keys(DEFAULT_LLM_ROLE_EXECUTION)) {
        const value = String(raw[role] || '').trim().toLowerCase();
        normalized[role] = (value === 'local') ? 'local' : 'remote';
    }
    return normalized;
}

function getCreateProjectLlmRoleExecution() {
    const directorSelect = document.getElementById('createDirectorBackend');

    return normalizeLlmRoleExecution({
        director: directorSelect ? directorSelect.value : 'remote'
    });
}

function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const d = new Date(isoString);
    return d.toLocaleTimeString() + ' ' + d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function formatRemainingDuration(ms) {
    const totalMinutes = Math.max(0, Math.floor(ms / 60000));
    const days = Math.floor(totalMinutes / (60 * 24));
    const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
    const minutes = totalMinutes % 60;

    if (days > 0) return `${days}d ${hours}h left`;
    if (hours > 0) return `${hours}h ${minutes}m left`;
    return `${minutes}m left`;
}

function buildRemainingTimeProgress(project) {
    const endMs = Date.parse(project.end_time_utc || '');
    if (Number.isNaN(endMs)) {
        return {
            label: 'No end time set',
            percentRemaining: 0,
            fillClass: 'time-progress-fill-urgent',
            expired: true,
            endLabel: 'N/A'
        };
    }

    let startMs = Date.parse(project.start_time_utc || '');
    if (Number.isNaN(startMs) || startMs >= endMs) {
        startMs = endMs - 24 * 60 * 60 * 1000;
    }

    const nowMs = Date.now();
    const totalMs = Math.max(1, endMs - startMs);
    const remainingMs = endMs - nowMs;
    const remainingRatio = clamp(remainingMs / totalMs, 0, 1);
    const percentRemaining = remainingRatio * 100;

    let fillClass = 'time-progress-fill-good';
    if (remainingRatio <= 0.2) {
        fillClass = 'time-progress-fill-urgent';
    } else if (remainingRatio <= 0.5) {
        fillClass = 'time-progress-fill-warn';
    }

    if (remainingMs <= 0) {
        return {
            label: 'Expired',
            percentRemaining: 0,
            fillClass: 'time-progress-fill-urgent',
            expired: true,
            endLabel: formatDate(project.end_time_utc)
        };
    }

    return {
        label: formatRemainingDuration(remainingMs),
        percentRemaining: percentRemaining,
        fillClass: fillClass,
        expired: false,
        endLabel: formatDate(project.end_time_utc)
    };
}

async function fetchData() {
    try {
        const resp = await fetch('/api/data');
        const data = await resp.json();

        // Update Projects
        const projContainer = document.getElementById('projects-container');
        if (data.projects.length === 0) {
            projContainer.innerHTML = '<div class="card empty-state" style="grid-column: 1 / -1;">No active projects found. Click New Project to begin.</div>';
        } else {
            let projHtml = '';
            for (let p of data.projects) {
                let showClass = (openMenuProject === p.project_name) ? "show" : "";
                const timeProgress = buildRemainingTimeProgress(p);
                const displayStatus = p.is_finished ? 'Finished' : p.status;
                const targetStatus = String(p.target_status || '').toLowerCase();
                const isTargetRunning = targetStatus === 'running';
                const isTargetStopped = targetStatus === 'stopped';

                const startDisabled = isTargetRunning
                    ? 'disabled style="opacity:0.5;cursor:not-allowed;"'
                    : '';
                const pauseDisabled = !isTargetRunning
                    ? 'disabled style="opacity:0.5;cursor:not-allowed;"'
                    : '';
                const stopDisabled = isTargetStopped
                    ? 'disabled style="opacity:0.5;cursor:not-allowed;"'
                    : '';

                const testSetButtonHtml = `<button class="btn-test-set" onclick="runOnTestSet('${p.project_name}', this)">Run on test set</button>`;

                const primaryControlsHtml = p.is_finished
                    ? `${testSetButtonHtml}`
                    : `
                        <button class="btn-start" onclick="sendCommand('${p.project_name}', 'start')" ${startDisabled}>▶ Start</button>
                        <button class="btn-pause" onclick="sendCommand('${p.project_name}', 'pause')" ${pauseDisabled}>⏸ Pause</button>
                        <button class="btn-stop" onclick="sendCommand('${p.project_name}', 'stop')" ${stopDisabled}>■ Stop</button>
                        ${testSetButtonHtml}
                        ${p.status === 'Running' ? `<button class="btn-secondary" onclick="openLiveTraining('${p.project_name}')" style="margin-left: 5px; color: var(--accent-green); border-color: var(--accent-green); background: rgba(46, 160, 67, 0.1);">Live Training</button>` : ''}
                    `;
                projHtml += `
                    <div class="card">
                        <h2>${p.project_name}</h2>
                        <div class="stat-row">
                            <span class="stat-label">System Status</span>
                            <span class="status-badge ${getStatusClass(displayStatus)}">${displayStatus}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Target State</span>
                            <span class="stat-value">${p.target_status}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Current Cycle</span>
                            <span class="stat-value">${p.current_cycle}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Active Step</span>
                            <span class="stat-value">${p.current_step}</span>
                        </div>
                        <div class="stat-row" style="font-size: 0.8rem; margin-top: 1rem; color: #8b949e;">
                            <span>Last Updated:</span>
                            <span>${formatDate(p.last_updated)}</span>
                        </div>
                        <div class="time-progress-block">
                            <div class="time-progress-header">
                                <span class="stat-label">Remaining Time</span>
                                <span class="time-progress-label ${timeProgress.expired ? 'expired' : ''}">${timeProgress.label}</span>
                            </div>
                            <div class="time-progress-track">
                                <div class="time-progress-fill ${timeProgress.fillClass}" style="width: ${timeProgress.percentRemaining.toFixed(1)}%;"></div>
                            </div>
                            <div class="time-progress-meta">
                                <span>Ends: ${timeProgress.endLabel}</span>
                                <span>${Math.round(timeProgress.percentRemaining)}%</span>
                            </div>
                        </div>
                        <div class="controls-row">
                            ${primaryControlsHtml}
                            <div class="dropdown" style="margin-left: auto; position: relative;">
                                <button class="btn-menu" onclick="toggleMenu('${p.project_name}')">⋮</button>
                                <div id="menu-${p.project_name}" class="dropdown-content ${showClass}">
                                    <a href="#" onclick="openProjectDetails('${p.project_name}'); return false;">Project details</a>
                                    <a href="#" onclick="openDataExplorer('${p.project_name}'); return false;">Explore Data</a>
                                    <a href="#" onclick="openModelExplorer('${p.project_name}'); return false;">Explore Models</a>
                                    <div class="dropdown-toggle-row" onclick="event.stopPropagation();">
                                        <label class="dropdown-checkbox-label" title="Require manual confirmation for completed/failed steps">
                                            <input class="dropdown-checkbox" type="checkbox" ${p.manual_verification_enabled ? 'checked' : ''} onchange="toggleManualVerification('${p.project_name}', this.checked, event)">
                                            <span>Manual verification</span>
                                        </label>
                                    </div>
                                    <a href="#" class="text-danger" onclick="deleteProject('${p.project_name}'); return false;">Delete</a>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
            projContainer.innerHTML = projHtml;
        }

        // Update Logs
        const logBody = document.getElementById('logs-body');
        if (data.logs.length === 0) {
            logBody.innerHTML = '<tr><td colspan="5" class="empty-state">No execution logs yet.</td></tr>';
        } else {
            let logHtml = '';
            for (let l of data.logs) {
                const shownStatus = l.display_status || l.status;
                const confirmButton = l.needs_confirmation
                    ? `<div style="margin-top: 6px;"><button class="btn-confirm-step" onclick="confirmStep('${l.project_name}', ${l.id}); return false;">Confirm step</button></div>`
                    : '';

                logHtml += `
                    <tr>
                        <td style="color: #8b949e; font-size: 0.9em;">${formatDate(l.timestamp)}</td>
                        <td><strong>${l.project_name}</strong></td>
                        <td><a href="#" style="color: var(--accent-blue); text-decoration: underline;" onclick="openCycleDetail('${l.project_name}', '${l.cycle}'); return false;">${l.cycle}</a></td>
                        <td>${l.step_name}</td>
                        <td class="${getLogStatusClass(shownStatus)}"><strong>${shownStatus}</strong>${confirmButton}</td>
                    </tr>
                `;
            }
            logBody.innerHTML = logHtml;
        }

        document.getElementById('last-update').innerText = 'Last polled: ' + new Date().toLocaleTimeString();

    } catch (e) {
        console.error("Failed to fetch dashboard data:", e);
    }
}

async function sendCommand(projectName, action) {
    try {
        await fetch(`/api/${action}/${projectName}`, { method: 'POST' });
        fetchData(); // immediately refresh
    } catch (e) {
        console.error(`Failed to ${action} ${projectName}:`, e);
    }
}

async function runOnTestSet(projectName, buttonEl) {
    const originalLabel = buttonEl ? buttonEl.innerText : null;
    if (buttonEl) {
        buttonEl.disabled = true;
        buttonEl.innerText = 'Running...';
        buttonEl.style.opacity = '0.7';
    }

    try {
        const response = await fetch(`/api/project/${encodeURIComponent(projectName)}/run_test_set`, { method: 'POST' });
        const data = await response.json();

        if (!response.ok || data.status !== 'success') {
            alert(`Failed to run test set: ${data.message || 'Unknown error'}`);
            return;
        }

        const sampleCount = Number.isFinite(Number(data.evaluated_samples)) ? Number(data.evaluated_samples) : 'N/A';
        const ensembleId = data.ensemble_candidate_id || 'baseline_ensemble';
        const title = `Test Set Results: ${projectName} (${sampleCount} samples, ${ensembleId})`;
        openEnsembleDetail(data.metrics || {}, data.class_map || {}, title);
    } catch (e) {
        console.error(`Failed to run test-set evaluation for ${projectName}:`, e);
        alert('Network error while running test-set evaluation.');
    } finally {
        if (buttonEl) {
            buttonEl.disabled = false;
            buttonEl.innerText = originalLabel || 'Run on test set';
            buttonEl.style.opacity = '1';
        }
    }
}

async function toggleManualVerification(projectName, enabled, event) {
    if (event) event.stopPropagation();
    try {
        const response = await fetch(`/api/project/${projectName}/manual_verification`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ enabled: enabled })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            alert(`Failed to update manual verification: ${data.message || 'Unknown error'}`);
        }
        fetchData();
    } catch (e) {
        console.error(`Failed to toggle manual verification for ${projectName}:`, e);
        alert('Network error while updating manual verification setting.');
    }
}

async function confirmStep(projectName, logId) {
    try {
        const response = await fetch(`/api/project/${projectName}/confirm_step`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ log_id: logId })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            alert(`Failed to confirm step: ${data.message || 'Unknown error'}`);
            return;
        }

        fetchData();
    } catch (e) {
        console.error(`Failed to confirm step ${logId} for ${projectName}:`, e);
        alert('Network error while confirming step.');
    }
}

async function deleteProject(projectName) {
    if (!confirm(`Are you sure you want to permanently delete project '${projectName}'?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/project/${projectName}`, { method: 'DELETE' });
        if (!response.ok) {
            const data = await response.json();
            alert(`Failed to delete: ${data.message || 'Unknown error'}`);
            return;
        }
        fetchData(); // refresh the dashboard
    } catch (e) {
        console.error(`Failed to delete project ${projectName}:`, e);
        alert("Network error while attempting to delete project.");
    }
}

// Modal handling
async function openModal() {
    document.getElementById('newProjectModal').style.display = 'block';
    document.getElementById('projectName').focus();

    // Reset wizard
    document.getElementById('wizardStep1').style.display = 'block';
    document.getElementById('wizardStep2').style.display = 'none';
    document.getElementById('dynamicSignalsContainer').innerHTML = '';
    document.getElementById('dynamicClassesContainer').innerHTML = '';
    document.getElementById('btnLoadData').style.display = 'block';
    document.getElementById('loadDataSpinner').style.display = 'none';

    const defaults = { ...DEFAULT_LLM_ROLE_EXECUTION };
    const directorSelect = document.getElementById('createDirectorBackend');
    if (directorSelect) directorSelect.value = defaults.director;
    const ensembleArchitectureSelect = document.getElementById('ensembleArchitecture');
    if (ensembleArchitectureSelect) ensembleArchitectureSelect.value = 'default';

    // Fetch available datasets
    const datasetSelect = document.getElementById('datasetPath');
    datasetSelect.innerHTML = '<option value="">Loading datasets...</option>';
    try {
        const resp = await fetch('/api/datasets');
        const data = await resp.json();
        if (data.status === 'success' && data.datasets.length > 0) {
            datasetSelect.innerHTML = '<option value="">-- Select a Dataset --</option>';
            data.datasets.forEach(folder => {
                datasetSelect.innerHTML += `<option value="${folder}">${folder}</option>`;
            });
        } else {
            datasetSelect.innerHTML = '<option value="">No valid datasets found</option>';
        }
    } catch (e) {
        console.error("Failed to load datasets:", e);
        datasetSelect.innerHTML = '<option value="">Error loading datasets</option>';
    }
}

function closeModal() {
    document.getElementById('newProjectModal').style.display = 'none';
    document.getElementById('projectName').value = '';
    document.getElementById('projectDesc').value = '';
    document.getElementById('datasetPath').value = '';
}

async function handleDatasetChange() {
    const datasetSelect = document.getElementById('datasetPath');
    const descBox = document.getElementById('projectDesc');
    const path = datasetSelect.value;

    if (!path) return;

    try {
        const resp = await fetch(`/api/dataset_description?dataset_path=${encodeURIComponent(path)}`);
        const data = await resp.json();

        if (data.status === 'success' && data.description) {
            descBox.value = data.description;
        }
    } catch (e) {
        console.error("Failed to fetch dataset description:", e);
    }
}

async function loadDatasetContext() {
    const datasetPath = document.getElementById('datasetPath').value.trim();
    if (!datasetPath) {
        alert("Please enter a valid dataset path.");
        return;
    }

    document.getElementById('btnLoadData').style.display = 'none';
    document.getElementById('loadDataSpinner').style.display = 'block';

    try {
        const response = await fetch(`/api/dataset_info?dataset_path=${encodeURIComponent(datasetPath)}`);
        const data = await response.json();

        if (!response.ok || data.status !== 'success') {
            alert(`Failed to load dataset: ${data.message}`);
            document.getElementById('btnLoadData').style.display = 'block';
            document.getElementById('loadDataSpinner').style.display = 'none';
            return;
        }

        // Build dynamic inputs for Signals
        const sigContainer = document.getElementById('dynamicSignalsContainer');
        sigContainer.innerHTML = '<h3 style="margin-bottom: 5px; font-size: 1.1em;">Signals (X)</h3>';
        data.signals.forEach(sig => {
            sigContainer.innerHTML += `
                <div class="form-group signal-group" data-sig="${sig}" style="margin-top: 5px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                    <label style="font-size: 0.85em; width: 60px; flex-shrink: 0;"><code>${sig}</code></label>
                    <input type="text" class="signal-modality" placeholder="Modality (e.g. EKG, EEG)" style="width: 30%;">
                    <input type="text" class="signal-desc" placeholder="Describe this signal..." style="width: 70%;">
                </div>
            `;
        });

        // Build dynamic inputs for Classes
        const classContainer = document.getElementById('dynamicClassesContainer');
        classContainer.innerHTML = '<h3 style="margin-bottom: 5px; font-size: 1.1em; margin-top: 20px;">Classes (Y)</h3>';
        data.classes.forEach(cls => {
            classContainer.innerHTML += `
                <div class="form-group" style="margin-top: 5px; margin-bottom: 10px;">
                    <label style="font-size: 0.85em; display: inline-block; width: 80px;">Class <code>${cls}</code></label>
                    <input type="text" class="class-desc" data-cls="${cls}" placeholder="Describe this class..." style="width: calc(100% - 90px); display: inline-block;">
                </div>
            `;
        });

        // Transition to Step 2
        document.getElementById('wizardStep1').style.display = 'none';
        document.getElementById('wizardStep2').style.display = 'block';

    } catch (e) {
        console.error("Failed to fetch dataset context:", e);
        alert("Network error. See console.");
        document.getElementById('btnLoadData').style.display = 'block';
        document.getElementById('loadDataSpinner').style.display = 'none';
    }
}

async function submitNewProject() {
    const projectNameField = document.getElementById('projectName');
    const projectDescField = document.getElementById('projectDesc');
    const datasetPathField = document.getElementById('datasetPath');

    const projectName = projectNameField.value.trim();
    const projectDesc = projectDescField.value.trim();
    const datasetPath = datasetPathField.value.trim();

    // Parse the train proportion
    const trainProportionField = document.getElementById('trainProportion');
    let trainProportion = 1.0;
    if (trainProportionField && trainProportionField.value) {
        trainProportion = parseFloat(trainProportionField.value);
        if (isNaN(trainProportion) || trainProportion <= 0 || trainProportion > 1.0) {
            alert("Training proportion must be a valid number between 0.01 and 1.00");
            return;
        }
    }

    const ensembleArchitectureField = document.getElementById('ensembleArchitecture');
    let ensembleArchitecture = 'default';
    if (ensembleArchitectureField && ensembleArchitectureField.value) {
        const value = String(ensembleArchitectureField.value).trim().toLowerCase();
        if (value === 'default' || value === 'simple') {
            ensembleArchitecture = value;
        }
    }

    if (!projectName) {
        alert("Please enter a valid project name.");
        return;
    }

    if (!projectDesc) {
        alert("Please provide a project description context.");
        return;
    }

    // Aggregrate Context
    const signalMetadata = {};
    document.querySelectorAll('.signal-group').forEach(group => {
        const sig = group.getAttribute('data-sig');
        const mod = group.querySelector('.signal-modality').value.trim() || 'Unspecified';
        const desc = group.querySelector('.signal-desc').value.trim() || 'No description provided.';
        signalMetadata[sig] = { modality: mod, description: desc };
    });

    const classMetadata = {};
    document.querySelectorAll('.class-desc').forEach(el => {
        classMetadata[el.getAttribute('data-cls')] = el.value.trim() || 'No description provided.';
    });

    try {
        const llmRoleExecution = getCreateProjectLlmRoleExecution();
        const payload = {
            dataset_path: datasetPath,
            project_description: projectDesc,
            train_proportion: trainProportion,
            ensemble_architecture: ensembleArchitecture,
            signal_metadata: signalMetadata,
            class_metadata: classMetadata,
            llm_role_execution: llmRoleExecution,
        };

        const response = await fetch(`/api/create/${projectName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            alert(`Failed to initialize: ${errorData.message}`);
            return;
        }

        closeModal();
        fetchData();
    } catch (e) {
        console.error("Failed to create project:", e);
        alert("Failed to reach server. See console for details.");
    }
}

// Data Explorer Logic
let currentCharts = [];

function closeDataExplorerModal() {
    document.getElementById('dataExplorerModal').style.display = 'none';
    currentCharts.forEach(c => c.destroy());
    currentCharts = [];
}

async function openDataExplorer(projectName) {
    document.getElementById('exploreProjectName').innerText = `Data Explore: ${projectName}`;
    document.getElementById('exploreProjectName').setAttribute('data-project', projectName);
    document.getElementById('exploreSampleIdx').innerText = '-';
    document.getElementById('exploreLabel').innerText = '-';

    document.getElementById('exploreLoading').style.display = 'block';
    const exploreContent = document.getElementById('exploreContent');
    exploreContent.style.display = 'none';
    document.getElementById('dataExplorerModal').style.display = 'block';

    if (currentCharts.length > 0) {
        currentCharts.forEach(c => c.destroy());
        currentCharts = [];
    }

    try {
        const response = await fetch(`/api/project/${projectName}/explore_data`);
        const data = await response.json();

        if (data.status !== "success") {
            document.getElementById('exploreLoading').innerText = `Error: ${data.message}`;
            return;
        }

        document.getElementById('exploreSampleIdx').innerText = data.sample_index;
        document.getElementById('exploreLabel').innerText = data.label;

        document.getElementById('exploreLoading').style.display = 'none';

        exploreContent.style.display = 'flex';
        exploreContent.innerHTML = ''; // clear any existing plot canvas elements

        // Prepare Chart.js datasets
        const signalNames = Object.keys(data.signals);

        // Generate nice distinct colors for up to 10 signals
        const colors = [
            '#58a6ff', '#3fb950', '#f85149', '#d29922',
            '#a371f7', '#e1e4e8', '#00ffff', '#ff00ff',
        ];

        Chart.defaults.color = '#8b949e'; // Default text color
        Chart.defaults.borderColor = '#30363d'; // Default grid lines

        let labels = [];
        let colorIdx = 0;

        for (const sig of signalNames) {
            const arr = data.signals[sig];

            // Generate basic index array for X axis if not done yet
            if (labels.length === 0 && Array.isArray(arr)) {
                labels = Array.from({ length: arr.length }, (_, i) => i);
            }

            // Build subplot container
            const wrapper = document.createElement('div');
            wrapper.style.width = '100%';
            wrapper.style.height = '150px'; // fixed height per subplot

            const canvas = document.createElement('canvas');
            wrapper.appendChild(canvas);
            exploreContent.appendChild(wrapper);

            const ctx = canvas.getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: sig,
                        data: arr,
                        borderColor: colors[colorIdx % colors.length],
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0, // hide dots, just show line
                        tension: 0.1 // slight smoothing
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#c9d1d9'
                            }
                        },
                        tooltip: {
                            enabled: true
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: { display: false } // omit x label for cleaner subplots
                        },
                        y: {
                            display: true,
                            title: { display: false }
                        }
                    }
                }
            });

            currentCharts.push(chart);
            colorIdx++;
        }

    } catch (e) {
        console.error("Data Explorer Error:", e);
        document.getElementById('exploreLoading').innerText = "Network Error. See Console.";
    }
}


// Initial fetch, then poll every 3 seconds
fetchData();
setInterval(fetchData, 3000);

let modelExplorerInterval = null;
let modelHistoryCharts = [];
let liveChartInstance = null;
let liveTrainingInterval = null;

async function fetchAndRenderModelExplorer(projectName) {
    const loading = document.getElementById('modelLoading');
    const content = document.getElementById('modelContent');
    const gridContainer = document.getElementById('modelGridContainer');

    try {
        const response = await fetch(`/api/project/${projectName}/models`);
        const data = await response.json();

        loading.style.display = "none";

        if (data.status === 'error') {
            gridContainer.innerHTML = `<div class="empty-state" style="color: var(--accent-red)">Error: ${data.message}</div>`;
            content.style.display = "flex";
            return;
        }

        const signals = data.signals;
        const classes = data.classes;
        const classMap = data.class_map;
        const matrix = data.matrix || {};

        if (!signals || signals.length === 0 || !classes || classes.length === 0) {
            gridContainer.innerHTML = `<div class="empty-state">No signal or class data found for this project.</div>`;
            content.style.display = "flex";
            return;
        }

        // Build grid dynamically based on M signals x N classes
        gridContainer.className = 'model-grid';
        gridContainer.style.gridTemplateColumns = `150px repeat(${classes.length}, 1fr)`;

        // Header Row
        let html = `<div class="grid-cell grid-header">Signal \\ Class</div>`;
        for (let c of classes) {
            html += `<div class="grid-cell grid-header">${classMap[c] || c}</div>`;
        }

        // Data Rows
        for (let sig of signals) {
            html += `<div class="grid-cell grid-header" style="justify-content: flex-start;">${sig}</div>`;
            for (let c of classes) {
                // Assume expert_matrix schema handles nested [modality][class_label]
                let cellData = matrix[sig] && matrix[sig][c.toString()];

                if (cellData) {
                    const val = typeof cellData.f1 === 'number' ? cellData.f1.toFixed(4) : cellData.f1;
                    const jsonStr = JSON.stringify(cellData).replace(/"/g, '&quot;');
                    html += `<div class="grid-cell grid-data-cell has-model" onclick='showModelDetail(${jsonStr})'>
                        ${val}<br><span style="font-size: 0.8em; color: var(--text-muted);">f1</span>
                    </div>`;
                } else {
                    html += `<div class="grid-cell grid-data-cell no-model">N/A</div>`;
                }
            }
        }

        gridContainer.innerHTML = html;

        // Render Ensemble if available
        const existingEnsemble = document.getElementById('ensembleMetricsContainer');
        if (existingEnsemble) { existingEnsemble.remove(); }

        if (data.ensemble_metrics) {
            const acc = (data.ensemble_metrics.accuracy * 100).toFixed(1);
            const jsonStr = JSON.stringify(data.ensemble_metrics).replace(/"/g, '&quot;');
            const ensembleHtml = `
            <div id="ensembleMetricsContainer" style="margin-top: 20px; padding: 15px; background: rgba(88, 166, 255, 0.1); border: 1px solid var(--accent-blue); border-radius: 6px; text-align: center; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='rgba(88, 166, 255, 0.2)'" onmouseout="this.style.background='rgba(88, 166, 255, 0.1)'" onclick='openEnsembleDetail(${jsonStr}, ${JSON.stringify(classMap)})'>
                <h4 style="margin: 0 0 10px 0; color: var(--accent-blue);">Ensemble Model</h4>
                <div style="font-size: 1.5em; font-weight: bold; color: var(--text-main);">${acc}% <span style="font-size: 0.5em; font-weight: normal; color: var(--text-muted);">Accuracy</span></div>
            </div>`;
            content.insertAdjacentHTML('beforeend', ensembleHtml);
        }

        content.style.display = "flex";

    } catch (e) {
        console.error("Failed to load models:", e);
        loading.style.display = "none";
        gridContainer.innerHTML = `<div class="empty-state" style="color: var(--accent-red)">Error loading models. Check console.</div>`;
        content.style.display = "flex";
    }
}

async function openModelExplorer(projectName) {
    const modal = document.getElementById('modelExplorerModal');
    const title = document.getElementById('modelProjectName');
    const loading = document.getElementById('modelLoading');
    const content = document.getElementById('modelContent');
    const gridContainer = document.getElementById('modelGridContainer');

    title.innerText = `Model Explorer - ${projectName}`;
    title.setAttribute('data-project', projectName);

    modal.style.display = "block";
    loading.style.display = "block";
    content.style.display = "none";
    gridContainer.innerHTML = '';

    // Initial fetch, then poll
    await fetchAndRenderModelExplorer(projectName);

    if (modelExplorerInterval) clearInterval(modelExplorerInterval);
    modelExplorerInterval = setInterval(() => fetchAndRenderModelExplorer(projectName), 3000);
}

function closeModelExplorer() {
    if (modelExplorerInterval) {
        clearInterval(modelExplorerInterval);
        modelExplorerInterval = null;
    }
    closeModelHistoryModal();
    document.getElementById('modelExplorerModal').style.display = "none";
}

function destroyModelHistoryCharts() {
    modelHistoryCharts.forEach(chart => {
        try {
            chart.destroy();
        } catch (e) {
            console.warn('Failed to destroy model history chart:', e);
        }
    });
    modelHistoryCharts = [];
}

function closeModelHistoryModal() {
    destroyModelHistoryCharts();
    const modal = document.getElementById('modelHistoryModal');
    const loading = document.getElementById('modelHistoryLoading');
    const content = document.getElementById('modelHistoryContent');
    const ensembleContainer = document.getElementById('modelHistoryEnsemble');
    const expertsGrid = document.getElementById('modelHistoryExpertsGrid');

    if (ensembleContainer) ensembleContainer.innerHTML = '';
    if (expertsGrid) expertsGrid.innerHTML = '';
    if (loading) loading.style.display = 'none';
    if (content) content.style.display = 'none';
    if (modal) modal.style.display = 'none';
}

function openModelHistoryFromExplorer() {
    const projectName = document.getElementById('modelProjectName').getAttribute('data-project');
    if (!projectName) {
        alert('Please open a project in Model Explorer first.');
        return;
    }
    openModelHistory(projectName);
}

function normalizeHistoryCycles(rawCycles, experts, ensembleSeries) {
    const cycleSet = new Set();

    if (Array.isArray(rawCycles)) {
        rawCycles.forEach(cycle => {
            const cycleNum = Number(cycle);
            if (Number.isFinite(cycleNum)) {
                cycleSet.add(cycleNum);
            }
        });
    }

    if (Array.isArray(experts)) {
        experts.forEach(expert => {
            if (!expert || !Array.isArray(expert.history)) return;
            expert.history.forEach(point => {
                const cycleNum = Number(point && point.cycle);
                if (Number.isFinite(cycleNum)) {
                    cycleSet.add(cycleNum);
                }
            });
        });
    }

    if (Array.isArray(ensembleSeries)) {
        ensembleSeries.forEach(point => {
            const cycleNum = Number(point && point.cycle);
            if (Number.isFinite(cycleNum)) {
                cycleSet.add(cycleNum);
            }
        });
    }

    return Array.from(cycleSet).sort((a, b) => a - b);
}

function buildCycleMetricMap(historyPoints, metricKey) {
    const metricMap = new Map();
    if (!Array.isArray(historyPoints)) {
        return metricMap;
    }

    historyPoints.forEach(point => {
        const cycleNum = Number(point && point.cycle);
        const metricVal = Number(point && point[metricKey]);
        if (Number.isFinite(cycleNum) && Number.isFinite(metricVal)) {
            metricMap.set(cycleNum, metricVal);
        }
    });

    return metricMap;
}

function buildTrainedCycleMetricMap(historyPoints) {
    const trainedMap = new Map();
    if (!Array.isArray(historyPoints)) {
        return trainedMap;
    }

    historyPoints.forEach(point => {
        const cycleNum = Number(point && point.cycle);
        if (!Number.isFinite(cycleNum)) {
            return;
        }

        const trainedF1 = Number(point && point.trained_f1);
        if (Number.isFinite(trainedF1)) {
            trainedMap.set(cycleNum, trainedF1);
            return;
        }

        // Backward compatibility for older payloads without trained_f1.
        const trainedInCycle = Boolean(point && point.trained_in_cycle);
        const modelChanged = Boolean(point && point.model_changed);
        const fallbackF1 = Number(point && point.f1);
        if ((trainedInCycle || modelChanged) && Number.isFinite(fallbackF1)) {
            trainedMap.set(cycleNum, fallbackF1);
        }
    });

    return trainedMap;
}

function buildChangedCycleSet(historyPoints) {
    const changedCycles = new Set();
    if (!Array.isArray(historyPoints)) {
        return changedCycles;
    }

    historyPoints.forEach(point => {
        const cycleNum = Number(point && point.cycle);
        const modelChanged = Boolean(point && point.model_changed);
        if (Number.isFinite(cycleNum) && modelChanged) {
            changedCycles.add(cycleNum);
        }
    });

    return changedCycles;
}

function compareHistoryAxisValues(a, b) {
    const aText = String(a ?? '').trim();
    const bText = String(b ?? '').trim();

    const aNum = Number(aText);
    const bNum = Number(bText);
    if (Number.isFinite(aNum) && Number.isFinite(bNum)) {
        return aNum - bNum;
    }

    return aText.localeCompare(bText, undefined, {
        numeric: true,
        sensitivity: 'base'
    });
}

function createModelHistoryChartCard(gridEl, title, cycles, metricMap, colorHex, metricLabel, extraClassName = '', options = {}) {
    const card = document.createElement('div');
    const showTitle = Boolean(!(options && options.showTitle === false));
    const compactClass = showTitle ? '' : 'compact-history-card';
    card.className = `model-history-card ${compactClass} ${extraClassName}`.trim();

    if (showTitle) {
        const heading = document.createElement('h4');
        heading.textContent = title;
        card.appendChild(heading);
    }

    const canvasWrap = document.createElement('div');
    canvasWrap.className = 'model-history-canvas-wrap';
    const canvas = document.createElement('canvas');
    canvasWrap.appendChild(canvas);
    card.appendChild(canvasWrap);

    gridEl.appendChild(card);

    const carryForward = Boolean(options && options.carryForward);
    const trainedMetricMap = options && options.trainedMetricMap instanceof Map ? options.trainedMetricMap : null;

    const yData = [];
    let lastValue = null;
    cycles.forEach(cycle => {
        if (metricMap.has(cycle)) {
            const val = metricMap.get(cycle);
            lastValue = val;
            yData.push(val);
            return;
        }

        if (carryForward && lastValue !== null) {
            yData.push(lastValue);
            return;
        }

        yData.push(null);
    });

    const datasets = [{
        label: metricLabel,
        data: yData,
        borderColor: colorHex,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 0,
        pointStyle: 'circle',
        tension: 0.2,
        spanGaps: false
    }];

    if (trainedMetricMap && trainedMetricMap.size > 0) {
        const trainedData = cycles.map(cycle => {
            if (!trainedMetricMap.has(cycle)) {
                return null;
            }
            const trainedVal = Number(trainedMetricMap.get(cycle));
            return Number.isFinite(trainedVal) ? trainedVal : null;
        });

        datasets.push({
            label: 'Trained Model F1',
            data: trainedData,
            borderColor: colorHex,
            backgroundColor: colorHex,
            showLine: false,
            borderWidth: 0,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointStyle: 'circle',
            pointBackgroundColor: colorHex,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1.2,
            spanGaps: false,
        });
    }

    const chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: cycles,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'nearest',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const cycle = items && items[0] ? items[0].label : '';
                            return `Cycle ${cycle}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Cycle',
                        color: '#8b949e'
                    },
                    ticks: {
                        color: '#8b949e'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.08)'
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: metricLabel,
                        color: '#8b949e'
                    },
                    ticks: {
                        color: '#8b949e'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.08)'
                    }
                }
            }
        }
    });

    modelHistoryCharts.push(chart);
}

function renderExpertHistoryMatrix(expertsGrid, experts, cycles) {
    const palette = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#ffa657', '#56d364', '#ff7b72'];

    const modalitySet = new Set();
    const classSet = new Set();
    const classNameByLabel = new Map();
    const expertByKey = new Map();

    experts.forEach(expert => {
        if (!expert || typeof expert !== 'object') {
            return;
        }

        const modality = String(expert.modality || '').trim();
        const classLabel = String(expert.class_label || '').trim();
        if (!modality || !classLabel) {
            return;
        }

        modalitySet.add(modality);
        classSet.add(classLabel);

        const className = String(expert.class_name || '').trim();
        if (className && !classNameByLabel.has(classLabel)) {
            classNameByLabel.set(classLabel, className);
        }

        expertByKey.set(`${modality}::${classLabel}`, expert);
    });

    const modalities = Array.from(modalitySet).sort(compareHistoryAxisValues);
    const classLabels = Array.from(classSet).sort(compareHistoryAxisValues);

    if (modalities.length === 0 || classLabels.length === 0) {
        expertsGrid.innerHTML = '<div class="empty-state" style="grid-column: 1 / -1;">No expert history available yet.</div>';
        return;
    }

    expertsGrid.style.setProperty('--history-class-columns', String(classLabels.length));

    const corner = document.createElement('div');
    corner.className = 'model-history-axis-corner';
    corner.textContent = 'Signal \\ Class';
    expertsGrid.appendChild(corner);

    classLabels.forEach(classLabel => {
        const header = document.createElement('div');
        header.className = 'model-history-col-header';
        header.textContent = classNameByLabel.get(classLabel) || classLabel;
        expertsGrid.appendChild(header);
    });

    modalities.forEach(modality => {
        const rowHeader = document.createElement('div');
        rowHeader.className = 'model-history-row-header';
        rowHeader.textContent = modality;
        expertsGrid.appendChild(rowHeader);

        classLabels.forEach((classLabel, classIdx) => {
            const matrixCell = document.createElement('div');
            matrixCell.className = 'model-history-matrix-cell';
            expertsGrid.appendChild(matrixCell);

            const expert = expertByKey.get(`${modality}::${classLabel}`);
            if (!expert) {
                matrixCell.innerHTML = '<div class="model-history-card compact-history-card" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);">N/A</div>';
                return;
            }

            const historyPoints = Array.isArray(expert.history) ? expert.history : [];
            const metricMap = buildCycleMetricMap(historyPoints, 'f1');
            const trainedMetricMap = buildTrainedCycleMetricMap(historyPoints);

            createModelHistoryChartCard(
                matrixCell,
                `${modality} | class ${classLabel}`,
                cycles,
                metricMap,
                palette[classIdx % palette.length],
                'F1',
                '',
                {
                    carryForward: true,
                    trainedMetricMap: trainedMetricMap,
                    showTitle: false,
                }
            );
        });
    });
}

function renderModelHistoryCharts(payload) {
    const content = document.getElementById('modelHistoryContent');
    const loading = document.getElementById('modelHistoryLoading');
    const ensembleContainer = document.getElementById('modelHistoryEnsemble');
    const expertsGrid = document.getElementById('modelHistoryExpertsGrid');

    const experts = Array.isArray(payload.experts) ? payload.experts : [];
    const ensembleSeries = Array.isArray(payload.ensemble_kappa) ? payload.ensemble_kappa : [];
    const cycles = normalizeHistoryCycles(payload.cycles, experts, ensembleSeries);

    destroyModelHistoryCharts();
    ensembleContainer.innerHTML = '';
    expertsGrid.innerHTML = '';

    if (cycles.length === 0) {
        expertsGrid.innerHTML = '<div class="empty-state" style="grid-column: 1 / -1;">No cycle history available yet.</div>';
        loading.style.display = 'none';
        content.style.display = 'block';
        return;
    }

    const ensembleMap = buildCycleMetricMap(ensembleSeries, 'kappa');
    createModelHistoryChartCard(
        ensembleContainer,
        'Ensemble | Cohen\'s Kappa',
        cycles,
        ensembleMap,
        '#79c0ff',
        'Kappa',
        'ensemble-card',
        {
            carryForward: false,
            useChangeMarkers: false,
        }
    );

    renderExpertHistoryMatrix(expertsGrid, experts, cycles);

    loading.style.display = 'none';
    content.style.display = 'block';
}

async function openModelHistory(projectName) {
    const modal = document.getElementById('modelHistoryModal');
    const title = document.getElementById('modelHistoryTitle');
    const loading = document.getElementById('modelHistoryLoading');
    const content = document.getElementById('modelHistoryContent');
    const ensembleContainer = document.getElementById('modelHistoryEnsemble');
    const expertsGrid = document.getElementById('modelHistoryExpertsGrid');

    title.innerText = `Model History - ${projectName}`;
    content.style.display = 'none';
    loading.style.display = 'block';
    ensembleContainer.innerHTML = '';
    expertsGrid.innerHTML = '';
    destroyModelHistoryCharts();
    modal.style.display = 'block';

    try {
        const response = await fetch(`/api/project/${encodeURIComponent(projectName)}/model_history`);
        const payload = await response.json();

        if (!response.ok || payload.status !== 'success') {
            loading.style.display = 'none';
            content.style.display = 'block';
            expertsGrid.innerHTML = `<div class="empty-state" style="color: var(--accent-red); grid-column: 1 / -1;">Failed to load model history: ${payload.message || 'Unknown error'}</div>`;
            return;
        }

        renderModelHistoryCharts(payload);
    } catch (e) {
        console.error('Failed to load model history:', e);
        loading.style.display = 'none';
        content.style.display = 'block';
        expertsGrid.innerHTML = '<div class="empty-state" style="color: var(--accent-red); grid-column: 1 / -1;">Network error while loading model history.</div>';
    }
}

function showModelDetail(modelData) {
    const modal = document.getElementById('modelDetailModal');
    const content = document.getElementById('modelDetailContent');
    const projectName = document.getElementById('modelProjectName').getAttribute('data-project');

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    let html = `<table style="width: 100%; border-collapse: collapse; text-align: left;">`;
    for (let key in modelData) {
        if (key === 'history' || key === 'parameters' || key === 'cv_metrics' || key === 'validation_metrics') continue;

        let val = modelData[key];

        // Render python file references as clickable links
        if (typeof val === 'string' && val.endsWith('.py')) {
            val = `<a href="#" style="color: var(--accent-blue); text-decoration: underline;" onclick="openCodeViewer('${projectName}', '${val}'); return false;">${val}</a>`;
        } else if (val && typeof val === 'object') {
            const pretty = JSON.stringify(val, null, 2);
            val = `<pre style="margin: 0; white-space: pre-wrap; font-size: 0.82em; color: var(--text-main);">${escapeHtml(pretty)}</pre>`;
        }

        html += `<tr>
            <td style="padding: 10px 8px; border-bottom: 1px solid var(--border-color); color: var(--accent-blue); font-weight: 600; width: 40%; word-break: break-all;">${key.replace(/_/g, ' ')}</td>
            <td style="padding: 10px 8px; border-bottom: 1px solid var(--border-color); word-break: break-all;">${val}</td>
        </tr>`;
    }
    html += `</table>`;

    const cvMetrics = modelData && typeof modelData.cv_metrics === 'object' ? modelData.cv_metrics : null;
    if (cvMetrics) {
        html += `<h4 style="margin-top: 20px; margin-bottom: 10px; color: var(--text-main);">Cross-Validation Metrics (Train Split, 3-Fold)</h4>`;
        html += `<pre style="margin: 0; padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; background: rgba(56, 139, 253, 0.08); white-space: pre-wrap; font-size: 0.82em; color: var(--text-main);">${escapeHtml(JSON.stringify(cvMetrics, null, 2))}</pre>`;
    }

    const validationMetrics = modelData && typeof modelData.validation_metrics === 'object' ? modelData.validation_metrics : null;
    if (validationMetrics) {
        html += `<h4 style="margin-top: 20px; margin-bottom: 10px; color: var(--text-main);">Validation-Set Metrics (Final Evaluation)</h4>`;
        html += `<pre style="margin: 0; padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; background: rgba(63, 185, 80, 0.10); white-space: pre-wrap; font-size: 0.82em; color: var(--text-main);">${escapeHtml(JSON.stringify(validationMetrics, null, 2))}</pre>`;
    }

    if (modelData.history && Array.isArray(modelData.history) && modelData.history.length > 0) {
        html += `<h4 style="margin-top: 20px; margin-bottom: 10px; color: var(--text-main);">History of Previous Models</h4>`;
        html += `<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9em;">`;
        html += `<thead><tr style="color: var(--text-muted); border-bottom: 1px solid var(--border-color);">
            <th style="padding: 8px;">Candidate ID</th>
            <th style="padding: 8px;">F1</th>
            <th style="padding: 8px;">Accuracy</th>
            <th style="padding: 8px;">Recall</th>
            <th style="padding: 8px;">Precision</th>
        </tr></thead><tbody>`;
        for (let hist of modelData.history) {
            html += `<tr>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color); color: var(--accent-blue);">${hist.candidate_id || 'N/A'}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${hist.f1 !== undefined ? hist.f1.toFixed(4) : '-'}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${hist.accuracy !== undefined ? hist.accuracy.toFixed(4) : '-'}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${hist.recall !== undefined ? hist.recall.toFixed(4) : '-'}</td>
                <td style="padding: 8px; border-bottom: 1px solid var(--border-color);">${hist.precision !== undefined ? hist.precision.toFixed(4) : '-'}</td>
            </tr>`;
        }
        html += `</tbody></table></div>`;
    }

    content.innerHTML = html;
    modal.style.display = "block";
}

function closeModelDetail() {
    document.getElementById('modelDetailModal').style.display = "none";
}

function openEnsembleDetail(metrics, classMap, titleOverride) {
    const modal = document.getElementById('ensembleDetailModal');
    const title = document.getElementById('ensembleDetailTitle');

    if (title) {
        title.innerText = titleOverride || 'Ensemble Model Performance';
    }

    const accuracyVal = Number(metrics && metrics.accuracy);
    const kappaVal = Number(metrics && metrics.kappa);

    // Summary Headers
    document.getElementById('ensembleAccuracyVal').innerText = Number.isFinite(accuracyVal)
        ? (accuracyVal * 100).toFixed(1) + '%'
        : 'N/A';
    document.getElementById('ensembleKappaVal').innerText = Number.isFinite(kappaVal)
        ? kappaVal.toFixed(3)
        : 'N/A';

    // Confusion Matrix
    const confTable = document.getElementById('ensembleConfMatrix');
    let confHtml = '<tr><th style="padding: 5px; border-bottom: 1px solid var(--border-color); border-right: 1px solid var(--border-color);">T \\ P</th>';
    // build header
    const confusionMatrix = Array.isArray(metrics && metrics.confusion_matrix) ? metrics.confusion_matrix : [];
    const numClasses = confusionMatrix.length;
    for (let i = 0; i < numClasses; i++) {
        let label = classMap ? (classMap[i] || i) : i;
        confHtml += `<th style="padding: 5px; border-bottom: 1px solid var(--border-color);">${label}</th>`;
    }
    confHtml += '</tr>';

    if (confusionMatrix.length > 0) {
        let maxVal = Math.max(...confusionMatrix.flat().map(v => Number(v) || 0));
        for (let r = 0; r < numClasses; r++) {
            let rowLabel = classMap ? (classMap[r] || r) : r;
            confHtml += `<tr><th style="padding: 5px; border-right: 1px solid var(--border-color);">${rowLabel}</th>`;
            for (let c = 0; c < numClasses; c++) {
                let val = Number(confusionMatrix[r][c]) || 0;
                let alpha = maxVal > 0 ? (val / maxVal) : 0;
                let bg = r === c ? `rgba(88, 166, 255, ${alpha * 0.8})` : `rgba(248, 81, 73, ${alpha * 0.8})`;
                confHtml += `<td style="padding: 10px; background-color: ${val > 0 ? bg : 'transparent'}; border: 1px solid rgba(255, 255, 255, 0.05);">${val}</td>`;
            }
            confHtml += '</tr>';
        }
    }
    confTable.innerHTML = confHtml;

    // Per-Class Report
    const reportTable = document.getElementById('ensembleClassReportRows');
    let reportHtml = '';
    const report = metrics && metrics.classification_report ? metrics.classification_report : null;
    if (report) {
        for (let key in report) {
            if (['accuracy', 'macro avg', 'weighted avg'].includes(key)) continue;
            let stats = report[key];
            if (!stats || typeof stats !== 'object') continue;
            let rowLabel = classMap ? (classMap[key] || key) : key;
            const f1 = Number.isFinite(Number(stats['f1-score'])) ? Number(stats['f1-score']).toFixed(4) : 'N/A';
            const precision = Number.isFinite(Number(stats['precision'])) ? Number(stats['precision']).toFixed(4) : 'N/A';
            const recall = Number.isFinite(Number(stats['recall'])) ? Number(stats['recall']).toFixed(4) : 'N/A';
            const support = Number.isFinite(Number(stats['support'])) ? Number(stats['support']) : 'N/A';
            reportHtml += `<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px;"><strong>${rowLabel}</strong></td>
                <td style="padding: 8px;">${f1}</td>
                <td style="padding: 8px;">${precision}</td>
                <td style="padding: 8px;">${recall}</td>
                <td style="padding: 8px; color: var(--text-muted);">${support}</td>
            </tr>`;
        }
    }
    reportTable.innerHTML = reportHtml;

    modal.style.display = "block";
}

function closeEnsembleDetail() {
    document.getElementById('ensembleDetailModal').style.display = "none";
}

async function openCodeViewer(projectName, filepath) {
    const modal = document.getElementById('modelCodeModal');
    const title = document.getElementById('modelCodeTitle');
    const loading = document.getElementById('modelCodeLoading');
    const codeContent = document.getElementById('modelCodeContent');

    title.innerText = filepath;
    codeContent.textContent = '';

    modal.style.display = "block";
    loading.style.display = "block";

    try {
        const response = await fetch(`/api/project/${projectName}/file?path=${encodeURIComponent(filepath)}`);
        const data = await response.json();

        loading.style.display = "none";

        if (data.status === 'success') {
            codeContent.className = 'language-python';
            codeContent.textContent = data.content;

            // Apply syntax highlighting
            if (window.hljs) {
                // reset hljs state on elements to allow re-highlighting
                delete codeContent.dataset.highlighted;
                hljs.highlightElement(codeContent);
            }

        } else {
            codeContent.textContent = `Error: ${data.message}`;
            codeContent.style.color = "var(--accent-red)";
        }
    } catch (e) {
        console.error("Failed to load file:", e);
        loading.style.display = "none";
        codeContent.textContent = `Failed to connect to backend: ${e.message}`;
        codeContent.style.color = "var(--accent-red)";
    }
}

// Cycle Detail Logic
async function openCycleDetail(projectName, cycleId) {
    const modal = document.getElementById('cycleDetailModal');
    const title = document.getElementById('cycleDetailTitle');
    const subtitle = document.getElementById('cycleDetailSubtitle');
    const sidebar = document.getElementById('cycleFileSidebar');
    const viewer = document.getElementById('cycleFileViewer');

    title.innerText = `Cycle Activity: ${cycleId}`;
    subtitle.innerText = `Project: ${projectName}`;
    sidebar.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-muted);">Loading files...</div>`;
    viewer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">Select a file from the sidebar to view its contents.</div>`;
    modal.style.display = "flex";

    try {
        const response = await fetch(`/api/project/${projectName}/cycle/${cycleId}/files`);
        const data = await response.json();

        if (data.status === 'success') {
            if (data.files.length === 0) {
                sidebar.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--text-muted);">No files found for this cycle.</div>`;
                return;
            }

            let html = '';
            for (let file of data.files) {
                const fullPath = `${data.cycle_root}/${file}`;
                html += `<button class="btn-sidebar-file" style="padding: 10px; background-color: var(--bg-main); border: 1px solid var(--border-color); color: var(--text-main); text-align: left; cursor: pointer; border-radius: 4px; font-family: monospace; font-size: 0.85em; transition: background-color 0.2s;" onclick="fetchAndDisplayCycleFile('${projectName}', '${fullPath}', '${file}', this)">${file}</button>`;
            }
            sidebar.innerHTML = html;
        } else {
            sidebar.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--accent-red);">Error: ${data.message}</div>`;
        }
    } catch (e) {
        console.error("Failed to load cycle files:", e);
        sidebar.innerHTML = `<div style="text-align: center; padding: 20px; color: var(--accent-red);">Failed to fetch files.</div>`;
    }
}

function closeCycleDetail() {
    document.getElementById('cycleDetailModal').style.display = "none";
}

async function fetchAndDisplayCycleFile(projectName, fullPath, filename, btnEl) {
    const viewer = document.getElementById('cycleFileViewer');

    // Highlight active button
    document.querySelectorAll('.btn-sidebar-file').forEach(el => {
        el.style.backgroundColor = 'var(--bg-main)';
        el.style.borderColor = 'var(--border-color)';
    });
    if (btnEl) {
        btnEl.style.backgroundColor = '#1f6feb33'; // light blue tint
        btnEl.style.borderColor = 'var(--accent-blue)';
    }

    viewer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">Loading ${filename}...</div>`;

    try {
        const response = await fetch(`/api/project/${projectName}/file?path=${encodeURIComponent(fullPath)}`);
        const data = await response.json();

        if (data.status === 'success') {
            const content = data.content;
            const ext = filename.split('.').pop().toLowerCase();

            if (ext === 'md') {
                // Render Markdown
                if (window.marked) {
                    viewer.innerHTML = `<div class="markdown-body" style="color: var(--text-main); line-height: 1.6;">${marked.parse(content)}</div>`;
                } else {
                    viewer.innerHTML = `<pre style="white-space: pre-wrap; font-family: monospace; font-size: 0.9em; color: var(--text-main);">${content}</pre>`;
                }
            } else if (ext === 'json') {
                // Try recursive JSON collapsable view
                try {
                    const jsonObj = JSON.parse(content);
                    const htmlTree = buildCollapsibleJson(jsonObj);
                    viewer.innerHTML = `<div style="font-family: monospace; font-size: 0.9em; line-height: 1.5; color: var(--text-main);">${htmlTree}</div>`;
                } catch (jsonErr) {
                    // Fallback to simple colored text if unparseable
                    renderCodeBlock(viewer, content, 'json');
                }
            } else {
                // Default syntax highlighter
                renderCodeBlock(viewer, content, ext);
            }

        } else {
            viewer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Error: ${data.message}</div>`;
        }
    } catch (e) {
        console.error("Failed to fetch file content:", e);
        viewer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Failed to load file.</div>`;
    }
}

function renderCodeBlock(container, content, lang) {
    let className = lang ? `language-${lang}` : '';
    // Use proper syntax-highlighter injection technique
    container.innerHTML = `<pre style="margin: 0; background-color: #0d1117; padding: 15px; border-radius: 6px; border: 1px solid var(--border-color); overflow-x: auto;"><code class="${className}"></code></pre>`;
    const codeEl = container.querySelector('code');
    codeEl.textContent = content; // strict text injection avoids XSS escaping
    if (window.hljs) {
        hljs.highlightElement(codeEl);
    }
}

function buildCollapsibleJson(data, isRoot = true) {
    // Basic primitives
    if (data === null) return `<span style="color: #ff7b72;">null</span>`;
    if (typeof data === 'boolean') return `<span style="color: #79c0ff;">${data}</span>`;
    if (typeof data === 'number') return `<span style="color: #79c0ff;">${data}</span>`;
    if (typeof data === 'string') {
        const escapedStr = data.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return `<span style="color: #a5d6ff;">"${escapedStr}"</span>`;
    }

    // Objects & Arrays
    if (Array.isArray(data)) {
        if (data.length === 0) return `<span>[]</span>`;
        let html = `<details ${isRoot ? 'open' : ''} style="margin-left: ${isRoot ? '0' : '15px'};">
            <summary style="cursor: pointer; color: var(--text-muted); user-select: none;">[</summary>
            <div style="margin-left: 20px; border-left: 1px dotted var(--border-color); padding-left: 10px;">`;
        for (let i = 0; i < data.length; i++) {
            html += `<div style="margin-bottom: 2px;">${buildCollapsibleJson(data[i], false)}${i < data.length - 1 ? '<span style="color: var(--text-muted);">,</span>' : ''}</div>`;
        }
        html += `</div><div style="color: var(--text-muted);">]</div></details>`;
        return html;
    }

    if (typeof data === 'object') {
        const keys = Object.keys(data);
        if (keys.length === 0) return `<span>{}</span>`;
        let html = `<details ${isRoot ? 'open' : ''} style="margin-left: ${isRoot ? '0' : '15px'};">
            <summary style="cursor: pointer; color: var(--text-muted); user-select: none;">{</summary>
            <div style="margin-left: 20px; border-left: 1px dotted var(--border-color); padding-left: 10px;">`;
        for (let i = 0; i < keys.length; i++) {
            const k = keys[i];
            const escapedK = k.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            html += `<div style="margin-bottom: 2px;"><span style="color: #d2a8ff;">"${escapedK}"</span><span style="color: var(--text-muted);">:</span> ${buildCollapsibleJson(data[k], false)}${i < keys.length - 1 ? '<span style="color: var(--text-muted);">,</span>' : ''}</div>`;
        }
        html += `</div><div style="color: var(--text-muted);">}</div></details>`;
        return html;
    }

    return `<span>${data}</span>`;
}

function closeCodeViewer() {
    document.getElementById('modelCodeModal').style.display = "none";
}

// Project Details Logic
function isoToDatetimeLocal(isoString) {
    if (!isoString) return '';
    const d = new Date(isoString);
    if (Number.isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    const year = d.getFullYear();
    const month = pad(d.getMonth() + 1);
    const day = pad(d.getDate());
    const hour = pad(d.getHours());
    const minute = pad(d.getMinutes());
    return `${year}-${month}-${day}T${hour}:${minute}`;
}

function datetimeLocalToUtcIso(localValue) {
    if (!localValue) return null;
    const d = new Date(localValue);
    if (Number.isNaN(d.getTime())) return null;
    return d.toISOString();
}

function setProjectEndTimeStatus(message, isError = false) {
    const statusEl = document.getElementById('projectEndTimeStatus');
    if (!statusEl) return;
    statusEl.innerText = message;
    statusEl.style.color = isError ? 'var(--accent-red)' : 'var(--text-muted)';
}

async function loadProjectEndTime(projectName) {
    const input = document.getElementById('projectEndTimeInput');
    const saveBtn = document.getElementById('projectEndTimeSaveBtn');
    if (!input || !saveBtn) return;

    input.value = '';
    saveBtn.disabled = true;
    setProjectEndTimeStatus('Loading...');

    try {
        const response = await fetch(`/api/project/${encodeURIComponent(projectName)}/end_time`);
        const data = await response.json();
        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || 'Unknown error');
        }

        if (currentProjectDetailsProject !== projectName) {
            return;
        }

        input.value = isoToDatetimeLocal(data.end_time_utc);
        setProjectEndTimeStatus(`Current UTC: ${data.end_time_utc}`);
    } catch (e) {
        console.error('Failed to load project end time:', e);
        if (currentProjectDetailsProject === projectName) {
            setProjectEndTimeStatus('Failed to load end time.', true);
        }
    } finally {
        if (currentProjectDetailsProject === projectName) {
            saveBtn.disabled = false;
        }
    }
}

async function saveProjectEndTime() {
    if (!currentProjectDetailsProject) {
        return;
    }

    const input = document.getElementById('projectEndTimeInput');
    const saveBtn = document.getElementById('projectEndTimeSaveBtn');
    if (!input || !saveBtn) return;

    const endTimeUtc = datetimeLocalToUtcIso(input.value);
    if (!endTimeUtc) {
        setProjectEndTimeStatus('Please choose a valid end time.', true);
        return;
    }

    saveBtn.disabled = true;
    setProjectEndTimeStatus('Saving...');

    try {
        const response = await fetch(`/api/project/${encodeURIComponent(currentProjectDetailsProject)}/end_time`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ end_time_utc: endTimeUtc })
        });

        const data = await response.json();
        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || 'Unknown error');
        }

        input.value = isoToDatetimeLocal(data.end_time_utc);
        setProjectEndTimeStatus(`Saved. UTC: ${data.end_time_utc}`);
        fetchData();
    } catch (e) {
        console.error('Failed to save project end time:', e);
        setProjectEndTimeStatus(`Failed to save end time: ${e.message}`, true);
    } finally {
        saveBtn.disabled = false;
    }
}

async function openProjectDetails(projectName) {
    const modal = document.getElementById('projectDetailModal');
    const viewer = document.getElementById('projectDetailViewer');
    const title = document.getElementById('projectDetailTitle');
    currentProjectDetailsProject = projectName;

    // Close the dropdown menu immediately for better UX
    if (openMenuProject) {
        document.getElementById("menu-" + openMenuProject).classList.remove("show");
        openMenuProject = null;
    }

    title.innerText = `Project Details: ${projectName}`;
    viewer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);">Loading dataset context...</div>`;
    modal.style.display = "flex";

    loadProjectEndTime(projectName);

    try {
        // Query the relative shared/context path via the generic file API
        const targetPath = "shared/context/data_context.md";
        const response = await fetch(`/api/project/${projectName}/file?path=${encodeURIComponent(targetPath)}`);
        const data = await response.json();

        if (data.status === 'success') {
            if (window.marked) {
                viewer.innerHTML = marked.parse(data.content);
            } else {
                viewer.innerHTML = `<pre style="white-space: pre-wrap;">${data.content}</pre>`;
            }
        } else {
            viewer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Could not load details: ${data.message}</div>`;
        }
    } catch (e) {
        console.error("Failed to fetch project details:", e);
        viewer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Network error loading project details.</div>`;
    }
}

function closeProjectDetails() {
    currentProjectDetailsProject = null;
    document.getElementById('projectDetailModal').style.display = "none";
}

// Live Training Polling logic
async function fetchAndRenderLiveTraining(projectName) {
    try {
        const response = await fetch(`/api/project/${encodeURIComponent(projectName)}/training_status`);
        const data = await response.json();

        if (data.status === 'running' && data.data && data.data.history && data.data.history.length > 0) {
            const epochs = data.data.history.map(d => d.epoch);
            const trainLosses = data.data.history.map(d => d.train_loss !== undefined ? d.train_loss : d.loss);
            const valLosses = data.data.history.map(d => d.val_loss !== undefined ? d.val_loss : d.loss);
            const f1s = data.data.history.map(d => d.f1 || 0.0);

            const title = document.getElementById('liveTrainingTitle');
            if (title && data.data.model_name) {
                title.innerText = `🔴 Live Training: ${data.data.model_name}`;
            }

            if (liveChartInstance) {
                liveChartInstance.data.labels = epochs;
                liveChartInstance.data.datasets[0].data = trainLosses;
                liveChartInstance.data.datasets[1].data = valLosses;
                liveChartInstance.data.datasets[2].data = f1s;
                liveChartInstance.update('none'); // Update without animation for smooth live feeling
            }

            // Populate Table
            const tbody = document.getElementById('liveTrainingTableBody');
            if (tbody) {
                let html = '';
                // Reverse iterate to show most recent epochs at the top
                for (let i = data.data.history.length - 1; i >= 0; i--) {
                    const row = data.data.history[i];
                    const tLoss = row.train_loss !== undefined ? row.train_loss : row.loss;
                    const vLoss = row.val_loss !== undefined ? row.val_loss : row.loss;

                    html += `<tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px;">${row.epoch}</td>
                        <td style="padding: 8px; color: rgba(255, 123, 114, 0.7);">${tLoss.toFixed(4)}</td>
                        <td style="padding: 8px; color: #ff7b72;">${vLoss.toFixed(4)}</td>
                        <td style="padding: 8px; color: #58a6ff;">${(row.f1 || 0).toFixed(4)}</td>
                        <td style="padding: 8px;">${(row.precision || 0).toFixed(4)}</td>
                        <td style="padding: 8px;">${(row.recall || 0).toFixed(4)}</td>
                        <td style="padding: 8px;">${(row.accuracy || 0).toFixed(4)}</td>
                    </tr>`;
                }
                tbody.innerHTML = html;
            }
        } else if (data.status === 'idle') {
            // Either not started or already finished
            if (liveChartInstance && liveChartInstance.data.labels.length === 0) {
                // only clear if we never had data
            }
        }
    } catch (e) {
        console.error("Failed to fetch live training tracking: ", e);
    }
}

function openLiveTraining(projectName) {
    const modal = document.getElementById('liveTrainingModal');

    // Cleanup any existing instance
    if (liveChartInstance) {
        liveChartInstance.destroy();
    }
    if (liveTrainingInterval) {
        clearInterval(liveTrainingInterval);
    }

    const ctx = document.getElementById('liveChart').getContext('2d');
    liveChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: 'rgba(255, 123, 114, 0.5)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    yAxisID: 'yList',
                    tension: 0.1,
                    pointRadius: 2
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#ff7b72',
                    backgroundColor: 'rgba(255, 123, 114, 0.1)',
                    yAxisID: 'yList',
                    tension: 0.1,
                    pointRadius: 2
                },
                {
                    label: 'F1-Score',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    yAxisID: 'yF1',
                    tension: 0.1,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            color: '#c9d1d9',
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#8b949e' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#8b949e' }
                },
                yList: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Loss', color: '#ff7b72' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#8b949e' }
                },
                yF1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'F1-Score', color: '#58a6ff' },
                    grid: { drawOnChartArea: false }, // avoid double grid lines
                    ticks: { color: '#8b949e' },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: { labels: { color: '#c9d1d9' } }
            }
        }
    });

    modal.style.display = "block";
    fetchAndRenderLiveTraining(projectName);
    liveTrainingInterval = setInterval(() => {
        fetchAndRenderLiveTraining(projectName);
    }, 2000); // 2 second polling interval
}

function closeLiveTraining() {
    document.getElementById('liveTrainingModal').style.display = "none";
    if (liveTrainingInterval) {
        clearInterval(liveTrainingInterval);
        liveTrainingInterval = null;
    }
}
