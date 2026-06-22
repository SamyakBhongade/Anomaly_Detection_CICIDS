/* ══════════════════════════════════════════════════════════════
   Cognitive IDS Dashboard — JavaScript
   Connects to Flask backend via REST + SSE for real-time updates.
   ══════════════════════════════════════════════════════════════ */

class CognitiveIDS {
    constructor() {
        this.apiUrl = window.location.origin;
        this.stats = { total: 0, attacks: 0, benign: 0, attack_rate: 0, avg_attack_conf: 0 };
        this.attackTypes = {};
        this.knownAlertIds = new Set();
        this.sseConnected = false;
        this.eventSource = null;
        this.lastFeedHtml = '';

        this.init();
    }

    // ── Initialization ────────────────────────────────────────
    async init() {
        console.log('🛡️ Cognitive IDS Dashboard starting...');
        await this.checkHealth();
        this.initCharts();
        await this.loadInitialData();
        this.connectSSE();
    }

    // ── Health Check ──────────────────────────────────────────
    async checkHealth() {
        try {
            const resp = await fetch(`${this.apiUrl}/api/health`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            console.log('✅ Backend healthy:', data);

            this.setDot('ml-dot', true);
            this.setDot('db-dot', true);
            this.setLiveStatus('Backend online ✓', 'online');
            return true;
        } catch (err) {
            console.warn('⚠️ Health check failed, trying stats endpoint...', err.message);
            // Fallback: try /api/stats directly
            try {
                const resp2 = await fetch(`${this.apiUrl}/api/stats`);
                if (resp2.ok) {
                    this.setDot('ml-dot', true);
                    this.setDot('db-dot', true);
                    this.setLiveStatus('Backend online ✓', 'online');
                    return true;
                }
            } catch (e) { /* ignore */ }

            this.setDot('ml-dot', false);
            this.setDot('db-dot', false);
            this.setLiveStatus('Backend offline ✗ — Run: python dashboard_app/app.py', 'offline');
            return false;
        }
    }

    // ── Load Initial Data ─────────────────────────────────────
    async loadInitialData() {
        await Promise.all([this.fetchStats(), this.fetchFeed()]);
    }

    // ── Fetch Stats ───────────────────────────────────────────
    async fetchStats() {
        try {
            const resp = await fetch(`${this.apiUrl}/api/stats`);
            const data = await resp.json();

            this.stats.total = data.total || 0;
            this.stats.attacks = data.attacks || 0;
            this.stats.benign = data.benign || 0;
            this.stats.attack_rate = data.attack_rate || 0;
            this.stats.avg_attack_conf = data.avg_attack_conf || 0;
            this.attackTypes = data.attack_types || {};

            this.renderStats();
            this.updateAttackTypeChart();
        } catch (err) {
            console.error('❌ Failed to fetch stats:', err);
        }
    }

    // ── Fetch Feed ────────────────────────────────────────────
    async fetchFeed() {
        try {
            const resp = await fetch(`${this.apiUrl}/api/feed`);
            const feed = await resp.json();
            this.renderFeed(feed);
        } catch (err) {
            console.error('❌ Failed to fetch feed:', err);
        }
    }

    // ── SSE (Server-Sent Events) Connection ───────────────────
    connectSSE() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        console.log('📡 Connecting SSE stream...');
        this.eventSource = new EventSource(`${this.apiUrl}/api/stream`);

        this.eventSource.onopen = () => {
            console.log('🔗 SSE connected');
            this.sseConnected = true;
            this.setDot('sse-dot', true);
        };

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleSSEMessage(data);
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        };

        this.eventSource.onerror = () => {
            console.warn('⚠️ SSE disconnected, retrying in 3s...');
            this.sseConnected = false;
            this.setDot('sse-dot', false);
            this.eventSource.close();
            setTimeout(() => this.connectSSE(), 3000);
        };
    }

    // ── Handle SSE Messages ───────────────────────────────────
    handleSSEMessage(data) {
        let prevBenign = this.stats.benign;
        let prevAttacks = this.stats.attacks;

        // Update stats from SSE
        if (data.stats) {
            const s = data.stats;
            this.stats.total = s.total || 0;
            this.stats.attacks = s.attacks || 0;
            this.stats.benign = s.benign || 0;
            this.stats.attack_rate = s.attack_rate || 0;
            this.stats.avg_attack_conf = s.avg_attack_conf || 0;
            this.attackTypes = s.attack_types || {};

            this.renderStats();
            this.updateAttackTypeChart();

            // Update live status bar
            const now = new Date().toLocaleString();
            this.setLiveStatus(
                `LIVE | ${now}`,
                'online'
            );

            // Calculate actual traffic volume delta since last update
            let newBenign = this.stats.benign - prevBenign;
            let newAttacks = this.stats.attacks - prevAttacks;
        }

        // Handle new attack alerts from SSE
        if (data.alerts && data.alerts.length > 0) {
            let newAlertsFound = false;
            for (const alert of data.alerts) {
                if (!this.knownAlertIds.has(alert.id)) {
                    this.knownAlertIds.add(alert.id);
                    newAlertsFound = true;

                    // Show popup for high-confidence attacks
                    if (alert.label === 1 && alert.confidence >= 0.8) {
                        this.showAlert(alert);
                    }
                }
            }
            if (newAlertsFound) {
                // Refresh the feed to show new items
                this.fetchFeed();
            }
        }
    }

    // ── Render Stats Cards ────────────────────────────────────
    renderStats() {
        this.setText('stat-total', this.stats.total.toLocaleString());
        this.setText('stat-attacks', this.stats.attacks.toLocaleString());
        this.setText('stat-benign', this.stats.benign.toLocaleString());
        this.setText('stat-attack-rate', this.stats.attack_rate + '%');
        this.setText('stat-avg-conf',
            this.stats.avg_attack_conf > 0
                ? (this.stats.avg_attack_conf * 100).toFixed(1) + '%'
                : '—'
        );
    }

    // ── Render Feed ───────────────────────────────────────────
    renderFeed(feed) {
        const container = document.getElementById('threats-feed');
        if (!feed || feed.length === 0) {
            container.innerHTML = `
                <div class="no-threats">
                    <i class="fas fa-satellite-dish"></i>
                    <p>Waiting for traffic data...<br>
                    <small>Send traffic via <code>send_traffic.py</code> or POST to <code>/api/detect</code></small></p>
                </div>`;
            return;
        }

        let html = '';
        for (const item of feed.slice(0, 30)) {
            const isAttack = item.label === 1;
            const severity = isAttack
                ? (item.confidence >= 0.8 ? 'high' : item.confidence >= 0.5 ? 'medium' : 'low')
                : 'benign';
            const typeClass = isAttack ? 'attack' : 'benign';
            const typeName = isAttack ? (item.attack_type || 'Unknown Attack') : 'Benign';
            const time = this.formatTime(item.timestamp);
            const conf = (item.confidence * 100).toFixed(1);

            html += `
                <div class="threat-item ${severity}">
                    <div class="threat-header">
                        <span class="threat-type ${typeClass}">${isAttack ? '🚨' : '✅'} ${typeName}</span>
                        <span class="threat-time">${time}</span>
                    </div>
                    <div class="threat-details">
                        <span class="threat-ip-link" onclick="window.ids.analyzeIp('${item.ip_address}')" style="cursor: pointer; color: #00d4ff; font-family: monospace; font-weight: 600; margin-right: 8px; text-decoration: underline;">[${item.ip_address || 'Unknown'}]</span>
                        Confidence: <strong>${conf}%</strong> &nbsp;|&nbsp;
                        <span class="model-tag bilstm">BiLSTM ${(item.bilstm * 100).toFixed(0)}%</span>
                        <span class="model-tag cnn">CNN ${(item.cnn * 100).toFixed(0)}%</span>
                        <span class="model-tag transformer">TF ${(item.transformer * 100).toFixed(0)}%</span>
                        <span class="model-tag vae">VAE ${(item.vae * 100).toFixed(0)}%</span>
                    </div>
                </div>`;
        }

        container.innerHTML = html;
    }


    // ── Charts ────────────────────────────────────────────────
    initCharts() {
        // Attack type doughnut chart
        const atCtx = document.getElementById('attackTypeChart');
        if (!atCtx) return;

        this.attackTypeChart = new Chart(atCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#ff4757', '#ffa502', '#5352ed', '#2ed573', '#00d4ff',
                        '#ff6b81', '#7bed9f', '#70a1ff', '#eccc68', '#a29bfe'
                    ],
                    borderWidth: 0,
                    hoverOffset: 6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '60%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#8892a4',
                            font: { size: 11 },
                            padding: 12,
                            usePointStyle: true,
                        }
                    }
                }
            }
        });
    }

    updateAttackTypeChart() {
        if (!this.attackTypeChart) return;

        const labels = Object.keys(this.attackTypes);
        const data = Object.values(this.attackTypes);

        if (labels.length === 0) {
            this.attackTypeChart.data.labels = ['No attacks yet'];
            this.attackTypeChart.data.datasets[0].data = [1];
            this.attackTypeChart.data.datasets[0].backgroundColor = ['rgba(255,255,255,0.05)'];
        } else {
            this.attackTypeChart.data.labels = labels;
            this.attackTypeChart.data.datasets[0].data = data;
            this.attackTypeChart.data.datasets[0].backgroundColor = [
                '#ff4757', '#ffa502', '#5352ed', '#2ed573', '#00d4ff',
                '#ff6b81', '#7bed9f', '#70a1ff', '#eccc68', '#a29bfe'
            ];
        }
        this.attackTypeChart.update();
    }

    // ── Alert Popup ───────────────────────────────────────────
    showAlert(alert) {
        console.log('🚨 Showing alert popup:', alert.attack_type, alert.id);

        const details = document.getElementById('alert-details');
        details.innerHTML = `
            <div class="alert-detail-item">
                <span>Attack Type</span>
                <span style="color: #ff4757; font-weight: 700;">${alert.attack_type || 'Unknown'}</span>
            </div>
            <div class="alert-detail-item">
                <span>Source IP</span>
                <span style="color: #00d4ff; font-family: monospace; font-weight: 700;">${alert.ip_address || 'Unknown'}</span>
            </div>
            <div class="alert-detail-item">
                <span>Confidence</span>
                <span style="color: #ff4757; font-weight: 700;">${(alert.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="alert-detail-item">
                <span>Detection Method</span>
                <span>4-Model Ensemble (BiLSTM + CNN + Transformer + VAE)</span>
            </div>
            <div class="alert-detail-item">
                <span>Timestamp</span>
                <span>${this.formatTime(alert.timestamp)}</span>
            </div>
            <div class="alert-detail-item">
                <span>Event ID</span>
                <span style="color: #5a6275; font-size: 0.8rem;">#${alert.id}</span>
            </div>
        `;

        document.getElementById('overlay').classList.add('show');
        document.getElementById('alert-popup').classList.add('show');

        // Auto-close after 8 seconds
        setTimeout(() => {
            if (document.getElementById('alert-popup').classList.contains('show')) {
                this.closeAlert();
            }
        }, 8000);
    }

    closeAlert() {
        document.getElementById('overlay').classList.remove('show');
        document.getElementById('alert-popup').classList.remove('show');
    }

    // ── Actions ───────────────────────────────────────────────
    async refresh() {
        console.log('🔄 Manual refresh');
        const btn = document.getElementById('btn-refresh');
        if (btn) {
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            btn.disabled = true;
        }
        await this.checkHealth();
        await this.loadInitialData();
        if (btn) {
            btn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            btn.disabled = false;
        }
    }

    async resetData() {
        if (!confirm('Reset all detection data? This clears the database.')) return;
        try {
            await fetch(`${this.apiUrl}/api/reset`, { method: 'POST' });
            console.log('🗑️ Database reset');
            this.knownAlertIds.clear();
            this.attackTypes = {};
            await this.loadInitialData();
            document.getElementById('analyzer-content').innerHTML = `
                <div class="no-threats">
                    <i class="fas fa-fingerprint"></i>
                    <p>No IP selected.<br><small>Type an IP above or click one in the Threat Feed to begin analysis.</small></p>
                </div>
            `;
        } catch (err) {
            console.error('Reset failed:', err);
        }
    }

    // ── IP Analyzer & Firewall ──────────────────────────────────────────
    analyzeIpFromInput() {
        const input = document.getElementById('ip-search-input');
        if (input && input.value.trim() !== '') {
            this.analyzeIp(input.value.trim());
        }
    }

    async analyzeIp(ip) {
        if (!ip || ip === 'Unknown') return;
        
        // Scroll to analyzer section smoothly
        document.getElementById('analyzer-section').scrollIntoView({ behavior: 'smooth' });
        
        const content = document.getElementById('analyzer-content');
        content.innerHTML = `<div style="text-align:center; padding: 30px; color:#8892a4;"><i class="fas fa-spinner fa-spin"></i> Loading history for ${ip}...</div>`;
        
        try {
            const resp = await fetch(`${this.apiUrl}/api/ip_history?ip=${encodeURIComponent(ip)}`);
            const data = await resp.json();
            
            if (data.error) {
                content.innerHTML = `<div style="color:#ff4757; text-align:center; padding: 20px;">Error: ${data.error}</div>`;
                return;
            }

            const history = data.history || [];
            const isBlocked = data.is_blocked;
            
            let html = `
                <div style="background: rgba(15,15,35,0.4); border-radius: 8px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h4 style="margin: 0; font-size: 1.2rem; display: flex; align-items: center; gap: 10px;">
                            <i class="fas fa-network-wired" style="color: #00d4ff;"></i> 
                            <span style="font-family: monospace;">${ip}</span>
                        </h4>
                        <div>
                            ${isBlocked 
                                ? `<button class="btn-block blocked" disabled style="background:#ff4757; color:white; border:none; padding:8px 16px; border-radius:6px; font-weight:bold; opacity:0.7;"><i class="fas fa-ban"></i> BLOCKED</button>`
                                : `<button id="btn-block-${ip.replace(/\./g, '-')}" class="btn-block" onclick="window.ids.blockIp('${ip}')" style="background:#ff4757; color:white; border:none; padding:8px 16px; border-radius:6px; font-weight:bold; cursor:pointer; transition: 0.2s;"><i class="fas fa-shield-alt"></i> BLOCK IP</button>`
                            }
                        </div>
                    </div>
                    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                        <div style="flex: 1; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 6px;">
                            <div style="font-size: 0.8rem; color: #8892a4; text-transform: uppercase;">Total Requests</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">${history.length}</div>
                        </div>
                        <div style="flex: 1; background: rgba(255,71,87,0.1); padding: 15px; border-radius: 6px;">
                            <div style="font-size: 0.8rem; color: #ff4757; text-transform: uppercase;">Malicious Requests</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ff4757;">${history.filter(h => h.label === 1).length}</div>
                        </div>
                    </div>
            `;
            
            if (history.length === 0) {
                html += `<p style="color:#8892a4;">No recent traffic history found for this IP.</p></div>`;
            } else {
                html += `
                    <div style="max-height: 300px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.05); border-radius: 6px;">
                        <table style="width: 100%; border-collapse: collapse; text-align: left;">
                            <thead style="background: rgba(255,255,255,0.05); position: sticky; top: 0;">
                                <tr>
                                    <th style="padding: 10px; font-size: 0.85rem; color:#8892a4;">Time</th>
                                    <th style="padding: 10px; font-size: 0.85rem; color:#8892a4;">Type</th>
                                    <th style="padding: 10px; font-size: 0.85rem; color:#8892a4;">Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                for (const h of history) {
                    const typeColor = h.label === 1 ? '#ff4757' : '#2ed573';
                    const icon = h.label === 1 ? '🚨' : '✅';
                    html += `
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.02);">
                            <td style="padding: 10px; font-size: 0.85rem;">${this.formatTime(h.timestamp)}</td>
                            <td style="padding: 10px; font-size: 0.85rem; color: ${typeColor}; font-weight: bold;">${icon} ${h.attack_type}</td>
                            <td style="padding: 10px; font-size: 0.85rem;">${(h.confidence * 100).toFixed(1)}%</td>
                        </tr>
                    `;
                }
                html += `</tbody></table></div></div>`;
            }
            
            content.innerHTML = html;
            
        } catch (err) {
            console.error('Analyzer failed:', err);
            content.innerHTML = `<div style="color:#ff4757; text-align:center; padding: 20px;">Failed to load IP history.</div>`;
        }
    }

    async blockIp(ip) {
        if (!confirm(`Are you sure you want to BLOCK ${ip}? All future traffic from this IP will be rejected.`)) return;
        
        const btnId = `btn-block-${ip.replace(/\./g, '-')}`;
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Blocking...';
            btn.disabled = true;
        }
        
        try {
            const resp = await fetch(`${this.apiUrl}/api/block_ip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: ip })
            });
            const data = await resp.json();
            
            if (data.ok) {
                if (btn) {
                    btn.innerHTML = '<i class="fas fa-ban"></i> BLOCKED';
                    btn.style.opacity = '0.7';
                    btn.classList.add('blocked');
                }
                alert(`Successfully blocked IP: ${ip}`);
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (err) {
            console.error('Block failed:', err);
            alert(`Failed to block IP: ${err.message}`);
            if (btn) {
                btn.innerHTML = '<i class="fas fa-shield-alt"></i> BLOCK IP';
                btn.disabled = false;
            }
        }
    }

    // ── Helpers ────────────────────────────────────────────────
    setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    setDot(id, online) {
        const dot = document.getElementById(id);
        if (dot) dot.className = `status-dot ${online ? 'online' : ''}`;
    }

    setLiveStatus(text, state) {
        const el = document.getElementById('live-status');
        if (el) {
            el.textContent = text;
            el.className = `live-status ${state || ''}`;
        }
    }

    formatTime(timestamp) {
        if (!timestamp) return new Date().toLocaleTimeString();
        try {
            // Handle the timestamp format from the backend (YYYY-MM-DD HH:MM:SS.mmm)
            const d = new Date(timestamp.replace(' ', 'T'));
            if (isNaN(d.getTime())) return timestamp;
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            return timestamp;
        }
    }
}

// ── Boot ──────────────────────────────────────────────────────
window.ids = new CognitiveIDS();
