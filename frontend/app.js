/**
 * ELDERLY AI MONITOR — FRONTEND DASHBOARD CONTROLLER
 * Real-time WebSocket connection, REST telemetry polling, and MJPEG video streaming.
 */

(() => {
  "use strict";

  // 1. API Configuration & Auto-Discovery
  const isFileOrigin = window.location.protocol === "file:";
  const API_BASE = isFileOrigin ? "http://127.0.0.1:8000" : window.location.origin;
  const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/alerts";

  // State
  let soundEnabled = true;
  let isCameraRunning = false;
  let alertCount = 0;
  let wsConnection = null;
  let wsReconnectTimer = null;

  // DOM Elements
  const elBackendDot = document.getElementById("backend-status-dot");
  const elBackendText = document.getElementById("backend-status-text");
  const elWsDot = document.getElementById("ws-status-dot");
  const elWsText = document.getElementById("ws-status-text");
  const elClock = document.getElementById("digital-clock");

  const elKpiFps = document.getElementById("kpi-fps");
  const elKpiTracks = document.getElementById("kpi-tracks");
  const elKpiRisk = document.getElementById("kpi-risk");
  const elKpiRiskState = document.getElementById("kpi-risk-state");
  const elKpiRiskMeter = document.getElementById("kpi-risk-meter");

  const elTelemCpu = document.getElementById("telem-cpu");
  const elTelemRam = document.getElementById("telem-ram");
  const elTelemGpu = document.getElementById("telem-gpu");

  const elVideoStream = document.getElementById("live-video-stream");
  const elOfflineOverlay = document.getElementById("stream-offline-overlay");
  const elBtnStartOverlay = document.getElementById("btn-start-camera-overlay");
  const elBtnCameraToggle = document.getElementById("btn-camera-toggle");
  const elBtnCameraReset = document.getElementById("btn-camera-reset");
  const elBtnRefreshStream = document.getElementById("btn-refresh-stream");
  const elSourceSelect = document.getElementById("camera-source-select");
  const elRtspGroup = document.getElementById("rtsp-input-group");
  const elRtspInput = document.getElementById("rtsp-url-input");

  const elEmergencyBar = document.getElementById("emergency-alert-bar");
  const elBtnDismissAlert = document.getElementById("btn-dismiss-alert");
  const elAlertBarDesc = document.getElementById("alert-bar-desc");

  const elAlertsFeed = document.getElementById("alerts-feed");
  const elEmptyPlaceholder = document.getElementById("empty-alerts-placeholder");
  const elAlertsCountBadge = document.getElementById("alerts-count-badge");
  const elBtnTestChime = document.getElementById("btn-test-chime");
  const elBtnClearAlerts = document.getElementById("btn-clear-alerts");
  const elBtnSoundToggle = document.getElementById("btn-sound-toggle");
  const elBtnFullscreen = document.getElementById("btn-fullscreen");
  const elPersonTableBody = document.getElementById("person-table-body");

  // Video container & Fall Visual Elements
  const elVideoContainer = document.getElementById("video-container");
  const elVideoCard = document.querySelector(".video-card");
  const elVideoFallBadge = document.getElementById("video-fall-badge");
  const elVideoFallText = document.getElementById("video-fall-text");

  // Web Audio API Synthesizer for Multi-harmonic Emergency Alarm Chimes & Horn
  let audioCtx = null;
  let alarmChimeInterval = null;
  let alarmChimeTimeout = null;
  let sirenInterval = null;
  let sirenTimeout = null;
  let borderFlashTimeout = null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
  }

  // Tự động mở khóa AudioContext khi người dùng chạm hoặc click bất kỳ đâu
  ["click", "keydown", "touchstart", "pointerdown"].forEach((evt) => {
    window.addEventListener(evt, () => initAudioContext(), { passive: true });
  });

  // Tạo âm chuông kim loại vang đa hòa âm (Metallic Chime Bell Harmonic Synthesis)
  function playBellChimeTone(freq, duration = 0.55, gainLevel = 0.5) {
    if (!soundEnabled) return;
    initAudioContext();
    if (!audioCtx) return;

    try {
      const now = audioCtx.currentTime;

      // 1. Tần số cơ bản (Fundamental Sine Tone)
      const osc1 = audioCtx.createOscillator();
      const gain1 = audioCtx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(freq, now);
      gain1.gain.setValueAtTime(gainLevel, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + duration);
      osc1.connect(gain1);
      gain1.connect(audioCtx.destination);
      osc1.start(now);
      osc1.stop(now + duration);

      // 2. Họa âm quãng 8 (Octave Overtone - tạo âm vang chuông thật)
      const osc2 = audioCtx.createOscillator();
      const gain2 = audioCtx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(freq * 2.015, now);
      gain2.gain.setValueAtTime(gainLevel * 0.45, now);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + duration * 0.75);
      osc2.connect(gain2);
      gain2.connect(audioCtx.destination);
      osc2.start(now);
      osc2.stop(now + duration * 0.75);

      // 3. Họa âm ánh kim bậc cao (Triangle Shimmer)
      const osc3 = audioCtx.createOscillator();
      const gain3 = audioCtx.createGain();
      osc3.type = "triangle";
      osc3.frequency.setValueAtTime(freq * 3.01, now);
      gain3.gain.setValueAtTime(gainLevel * 0.25, now);
      gain3.gain.exponentialRampToValueAtTime(0.001, now + duration * 0.45);
      osc3.connect(gain3);
      gain3.connect(audioCtx.destination);
      osc3.start(now);
      osc3.stop(now + duration * 0.45);

    } catch (e) {
      console.warn("[Dashboard] Bell chime error:", e);
    }
  }

  // Phát tiếng còi báo động khẩn cấp
  function playSirenBeep(freq, duration = 0.22) {
    if (!soundEnabled) return;
    initAudioContext();
    if (!audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = "sawtooth"; // Tiếng còi cứu thương khẩn cấp
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

      gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("[Dashboard] Audio error:", e);
    }
  }

  // 1. Chuỗi chuông báo động ngã khẩn cấp (Fall Risk Urgent Alarm Chime: C6 -> A5 -> F6 -> C6 + horn)
  function playFallAlarmChimeSequence() {
    const notes = [1046.5, 880.0, 1396.9, 1046.5];
    notes.forEach((freq, idx) => {
      setTimeout(() => {
        if (!soundEnabled) return;
        playBellChimeTone(freq, 0.45, 0.5);
      }, idx * 140);
    });

    setTimeout(() => {
      if (!soundEnabled) return;
      playSirenBeep(960, 0.25);
    }, 600);
  }

  // 2. Chuỗi chuông báo hành vi bất thường (Abnormal Behavior Warning Chime: Đinh - Đoong)
  function playAbnormalWarningChimeSequence() {
    playBellChimeTone(880.0, 0.6, 0.45);
    setTimeout(() => {
      if (!soundEnabled) return;
      playBellChimeTone(659.25, 0.8, 0.5);
    }, 240);
  }

  // Bộ phát âm thanh chuông cảnh báo ra loa
  function triggerAlarmChime(type = "fall") {
    if (!soundEnabled) return;
    initAudioContext();
    stopContinuousAlarm();

    if (type === "fall") {
      // Chuông báo té ngã: Kêu dồn dập mỗi 1.4 giây
      playFallAlarmChimeSequence();
      alarmChimeInterval = setInterval(() => {
        if (!soundEnabled) {
          stopContinuousAlarm();
          return;
        }
        playFallAlarmChimeSequence();
      }, 1400);

      // Tự động dừng sau 10 giây nếu chưa có ai nhấn xác nhận
      alarmChimeTimeout = setTimeout(() => {
        stopContinuousAlarm();
      }, 10000);
    } else {
      // Chuông báo hành vi bất thường: Kêu nhắc nhở mỗi 2.2 giây
      playAbnormalWarningChimeSequence();
      alarmChimeInterval = setInterval(() => {
        if (!soundEnabled) {
          stopContinuousAlarm();
          return;
        }
        playAbnormalWarningChimeSequence();
      }, 2200);

      alarmChimeTimeout = setTimeout(() => {
        stopContinuousAlarm();
      }, 7000);
    }

    // Dự phòng âm thanh bằng thẻ HTML5 <audio>
    const elAudio = document.getElementById("alert-sound");
    if (elAudio) {
      try {
        elAudio.currentTime = 0;
        elAudio.play().catch(() => {});
      } catch (_) {}
    }
  }

  function stopContinuousAlarm() {
    if (alarmChimeInterval) {
      clearInterval(alarmChimeInterval);
      alarmChimeInterval = null;
    }
    if (alarmChimeTimeout) {
      clearTimeout(alarmChimeTimeout);
      alarmChimeTimeout = null;
    }
    if (sirenInterval) {
      clearInterval(sirenInterval);
      sirenInterval = null;
    }
    if (sirenTimeout) {
      clearTimeout(sirenTimeout);
      sirenTimeout = null;
    }
  }

  function clearEmergencyVisuals() {
    if (elVideoContainer) elVideoContainer.classList.remove("fall-emergency-border");
    if (elVideoCard) elVideoCard.classList.remove("fall-emergency-border");
    if (elVideoFallBadge) elVideoFallBadge.style.display = "none";
    if (borderFlashTimeout) {
      clearTimeout(borderFlashTimeout);
      borderFlashTimeout = null;
    }
  }

  // 2. Clock updater
  function updateClock() {
    const now = new Date();
    elClock.textContent = now.toTimeString().split(" ")[0];
  }
  setInterval(updateClock, 1000);
  updateClock();

  // 3. WebSocket Alerts Listener
  function connectAlertsWebSocket() {
    if (wsConnection) {
      try { wsConnection.close(); } catch (_) {}
    }

    elWsDot.className = "badge-dot dot-warning";
    elWsText.textContent = "WebSocket: Đang nối...";

    try {
      wsConnection = new WebSocket(WS_URL);

      wsConnection.onopen = () => {
        elWsDot.className = "badge-dot dot-online";
        elWsText.textContent = "WebSocket: Real-time";
        if (wsReconnectTimer) {
          clearTimeout(wsReconnectTimer);
          wsReconnectTimer = null;
        }
      };

      wsConnection.onmessage = (event) => {
        try {
          const alertData = JSON.parse(event.data);
          handleIncomingAlert(alertData);
        } catch (err) {
          console.error("[Dashboard] Error parsing WebSocket message:", err);
        }
      };

      wsConnection.onclose = () => {
        elWsDot.className = "badge-dot dot-offline";
        elWsText.textContent = "WebSocket: Mất kết nối";
        scheduleWsReconnect();
      };

      wsConnection.onerror = () => {
        elWsDot.className = "badge-dot dot-offline";
        elWsText.textContent = "WebSocket: Lỗi";
      };
    } catch (e) {
      console.warn("[Dashboard] WebSocket init failed, will retry:", e);
      scheduleWsReconnect();
    }
  }

  function scheduleWsReconnect() {
    if (!wsReconnectTimer) {
      wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectAlertsWebSocket();
      }, 3000);
    }
  }

  // 4. Handle Incoming Alert
  function handleIncomingAlert(alert) {
    const personId = alert.person_id ?? "N/A";
    const primaryAction = (alert.evidence?.primary_action ?? alert.action ?? "").toLowerCase();
    const riskScore = typeof alert.risk_score === "number" ? alert.risk_score : 0.95;

    // Phân loại: Nguy cơ té ngã vs Hành vi bất thường
    const isFallRisk = (
      primaryAction === "falling" ||
      primaryAction === "stumbling" ||
      primaryAction === "lying" ||
      primaryAction === "immobile" ||
      riskScore >= 0.70 ||
      (alert.evidence?.drop_severity_score && alert.evidence.drop_severity_score > 0.35)
    );

    // 1. Show flash emergency banner
    elEmergencyBar.style.display = "block";
    const alertTypeLabel = isFallRisk ? "NGUY CƠ TÉ NGÃ KHẨN CẤP" : "HÀNH VI BẤT THƯỜNG";
    elAlertBarDesc.textContent = `[${alertTypeLabel}] Đối tượng ID: #${personId} — ${primaryAction.toUpperCase()} (Điểm rủi ro: ${riskScore.toFixed(2)}). Possible medical emergency / abnormal behavior detected. Please check the person.`;

    // 2. Hiện khung viền màu đỏ chớp nháy cảnh báo & Floating badge trên video
    if (elVideoContainer) elVideoContainer.classList.add("fall-emergency-border");
    if (elVideoCard) elVideoCard.classList.add("fall-emergency-border");
    if (elVideoFallBadge) {
      elVideoFallBadge.style.display = "flex";
      if (elVideoFallText) {
        elVideoFallText.textContent = isFallRisk
          ? `🚨 CẢNH BÁO: PHÁT HIỆN NGUY CƠ TÉ NGÃ! (ID #${personId})`
          : `⚠️ CẢNH BÁO: PHÁT HIỆN HÀNH VI BẤT THƯỜNG! (ID #${personId})`;
      }
    }

    // Tự động tắt viền đỏ sau 10 giây nếu không có cảnh báo mới
    if (borderFlashTimeout) clearTimeout(borderFlashTimeout);
    borderFlashTimeout = setTimeout(() => {
      clearEmergencyVisuals();
    }, 10000);

    // 3. KÍCH HOẠT TIẾNG CHUÔNG CẢNH BÁO RA LOA (Chuông té ngã khẩn cấp hoặc chuông hành vi bất thường)
    triggerAlarmChime(isFallRisk ? "fall" : "abnormal");

    // 4. Prepend to feed
    renderAlertCard(alert, true);

    // 5. Update KPI
    updateRiskKpi(riskScore, isFallRisk ? "NGUY CƠ CAO (TÉ NGÃ)" : "BẤT THƯỜNG / NGHI VẤN");
  }

  function renderAlertCard(alert, isNew = false) {
    if (elEmptyPlaceholder) {
      elEmptyPlaceholder.style.display = "none";
    }

    alertCount++;
    elAlertsCountBadge.textContent = `${alertCount} sự kiện`;

    const card = document.createElement("div");
    card.className = "alert-item";

    const timestamp = alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    const action = alert.evidence?.primary_action || "Khẩn cấp";
    const score = typeof alert.risk_score === "number" ? alert.risk_score.toFixed(2) : "0.95";

    card.innerHTML = `
      <div class="alert-item-header">
        <div class="alert-item-title">
          <span>🚨</span>
          <span>Đối tượng #${alert.person_id} — ${action.toUpperCase()}</span>
        </div>
        <span class="alert-item-time">${timestamp}</span>
      </div>
      <div class="alert-item-msg">
        ${alert.message || "Possible medical emergency / abnormal behavior detected. Please check the person."}
      </div>
      <div class="alert-item-meta">
        <span class="meta-field">Rủi ro: <strong>${score}</strong></span>
        <span class="meta-field">Vận tốc Vy: <strong>${alert.evidence?.vertical_velocity ?? "N/A"}</strong></span>
        <span class="meta-field">Góc thân: <strong>${alert.evidence?.torso_angle_deg ? alert.evidence.torso_angle_deg + "°" : "N/A"}</strong></span>
        <span class="meta-field">Bất động: <strong>${alert.evidence?.immobility_index ?? "N/A"}</strong></span>
      </div>
    `;

    if (isNew) {
      elAlertsFeed.insertBefore(card, elAlertsFeed.firstChild);
    } else {
      elAlertsFeed.appendChild(card);
    }
  }

  // 5. Polling REST APIs for Telemetry & Health
  async function pollStatus() {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();

      elBackendDot.className = "badge-dot dot-online";
      elBackendText.textContent = "API: Sẵn sàng";

      // Camera State
      isCameraRunning = Boolean(data.camera_active);
      elKpiFps.textContent = data.current_fps ? data.current_fps.toFixed(1) : "0.0";
      elKpiTracks.textContent = data.tracked_persons_count ?? 0;

      // Update button labels
      if (isCameraRunning) {
        elBtnCameraToggle.textContent = "DỪNG CAMERA";
        elBtnCameraToggle.style.background = "#dc2626";
        elBtnCameraToggle.style.color = "#ffffff";
        elOfflineOverlay.style.display = "none";
        
        // Ensure stream URL is attached
        const streamUrl = `${API_BASE}/stream/mjpeg`;
        if (!elVideoStream.src || elVideoStream.src !== streamUrl) {
          elVideoStream.src = streamUrl;
        }
      } else {
        elBtnCameraToggle.textContent = "KHỞI ĐỘNG CAMERA";
        elBtnCameraToggle.style.background = "";
        elBtnCameraToggle.style.color = "";
        elOfflineOverlay.style.display = "flex";
      }

      // Update simulated or active persons in table
      updatePersonTable(data.tracked_persons_count);

    } catch (err) {
      elBackendDot.className = "badge-dot dot-offline";
      elBackendText.textContent = "API: Mất kết nối";
      elOfflineOverlay.style.display = "flex";
    }
  }

  async function pollHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.telemetry) {
        elTelemCpu.textContent = `${data.telemetry.cpu_percent ?? 0}%`;
        elTelemRam.textContent = `${data.telemetry.ram_used_gb ?? 0} GB`;
        elTelemGpu.textContent = data.telemetry.gpu_available ? "CUDA (RTX 3050)" : "CPU Runtime";
      }
    } catch (_) {}
  }

  // 6. Update Risk Meter KPI
  function updateRiskKpi(score, stateText) {
    elKpiRisk.textContent = score.toFixed(2);
    elKpiRiskState.textContent = stateText || (score > 0.7 ? "NGUY CƠ CAO" : score > 0.4 ? "NGHI VẤN" : "BÌNH THƯỜNG");

    const pct = Math.min(100, Math.max(5, score * 100));
    elKpiRiskMeter.style.width = `${pct}%`;

    if (score > 0.7) {
      elKpiRisk.className = "kpi-value risk-red";
      elKpiRiskMeter.className = "kpi-meter-fill meter-red";
    } else if (score > 0.4) {
      elKpiRisk.className = "kpi-value risk-yellow";
      elKpiRiskMeter.className = "kpi-meter-fill meter-yellow";
    } else {
      elKpiRisk.className = "kpi-value risk-green";
      elKpiRiskMeter.className = "kpi-meter-fill meter-green";
    }
  }

  // 7. Update Person Kinematics Table
  function updatePersonTable(trackCount) {
    if (!trackCount || trackCount === 0) {
      elPersonTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="text-center text-muted">Không phát hiện người trong phòng (Phòng trống)</td>
        </tr>
      `;
      updateRiskKpi(0.05, "BÌNH THƯỜNG");
      return;
    }

    let rowsHtml = "";
    for (let i = 1; i <= trackCount; i++) {
      rowsHtml += `
        <tr>
          <td><strong>#${i}</strong></td>
          <td><span class="action-badge badge-walking">walking</span></td>
          <td>12.4°</td>
          <td>0.48</td>
          <td>0.02 m/s</td>
          <td><span style="color: var(--status-normal);">0.08</span></td>
          <td><span style="color: var(--status-normal);">NORMAL</span></td>
        </tr>
      `;
    }
    elPersonTableBody.innerHTML = rowsHtml;
  }

  // 8. Fetch Historical Events on Load
  async function fetchHistoricalEvents() {
    try {
      const res = await fetch(`${API_BASE}/events?limit=20`);
      if (!res.ok) return;
      const events = await res.json();
      if (Array.isArray(events) && events.length > 0) {
        if (elEmptyPlaceholder) elEmptyPlaceholder.style.display = "none";
        events.forEach(evt => renderAlertCard(evt, false));
      }
    } catch (_) {}
  }

  // 9. Interactive UI Events
  // Stream error handler
  window.handleStreamError = () => {
    if (isCameraRunning) {
      // Auto-reattach with cache buster after brief network pause
      setTimeout(() => {
        elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
      }, 800);
    } else {
      elOfflineOverlay.style.display = "flex";
    }
  };

  // Toggle Source selection
  elSourceSelect.addEventListener("change", (e) => {
    if (e.target.value === "rtsp") {
      elRtspGroup.style.display = "block";
    } else {
      elRtspGroup.style.display = "none";
    }
  });

  // Clean Start / Reset Camera
  async function resetCamera() {
    elBtnCameraToggle.textContent = "ĐANG KHỞI TẠO...";
    if (elBtnCameraReset) elBtnCameraReset.textContent = "ĐANG RESET...";
    const sourceType = elSourceSelect.value;
    const rtspUrl = elRtspInput.value.trim() || undefined;

    try {
      const res = await fetch(`${API_BASE}/camera/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: sourceType,
          rtsp_url: rtspUrl,
        }),
      });

      if (res.ok) {
        isCameraRunning = true;
        elOfflineOverlay.style.display = "none";
        // Re-attach live stream with timestamp
        setTimeout(() => {
          elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
          pollStatus();
        }, 300);
      } else {
        const err = await res.json();
        alert("Không thể khởi động camera: " + (err.detail || "Lỗi thiết bị"));
        pollStatus();
      }
    } catch (e) {
      alert("Lỗi kết nối tới Backend API: " + e.message);
      pollStatus();
    } finally {
      if (elBtnCameraReset) elBtnCameraReset.textContent = "RESET CAMERA";
    }
  }

  // Camera Toggle
  async function toggleCamera() {
    if (isCameraRunning) {
      try {
        await fetch(`${API_BASE}/camera/stop`, { method: "POST" });
        isCameraRunning = false;
        elVideoStream.src = "";
        elOfflineOverlay.style.display = "flex";
        pollStatus();
      } catch (e) {
        alert("Lỗi khi dừng camera: " + e.message);
      }
    } else {
      await resetCamera();
    }
  }

  elBtnCameraToggle.addEventListener("click", toggleCamera);
  elBtnStartOverlay.addEventListener("click", resetCamera);
  if (elBtnCameraReset) elBtnCameraReset.addEventListener("click", resetCamera);

  // Refresh stream
  elBtnRefreshStream.addEventListener("click", () => {
    elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
  });

  // Dismiss Emergency Banner & Silence Alarm
  elBtnDismissAlert.addEventListener("click", () => {
    stopContinuousAlarm();
    clearEmergencyVisuals();
    elEmergencyBar.style.display = "none";
    updateRiskKpi(0.05, "BÌNH THƯỜNG");
  });

  // Clear Alerts (Xóa các lần cảnh báo & tắt chuông)
  async function clearAllAlerts() {
    // 1. Reset frontend feed
    elAlertsFeed.innerHTML = "";
    elAlertsFeed.appendChild(elEmptyPlaceholder);
    elEmptyPlaceholder.style.display = "flex";
    alertCount = 0;
    elAlertsCountBadge.textContent = "0 sự kiện";

    // 2. Tắt chuông báo động và xóa khung viền đỏ cảnh báo
    stopContinuousAlarm();
    clearEmergencyVisuals();
    elEmergencyBar.style.display = "none";
    updateRiskKpi(0.05, "BÌNH THƯỜNG");

    // 3. Xóa lịch sử lưu trữ trên Backend API
    try {
      await fetch(`${API_BASE}/events/clear`, { method: "POST" });
    } catch (_) {}
  }

  elBtnClearAlerts.addEventListener("click", clearAllAlerts);

  // Nút Test Chuông Cảnh Báo
  if (elBtnTestChime) {
    elBtnTestChime.addEventListener("click", () => {
      initAudioContext();
      if (!soundEnabled) {
        soundEnabled = true;
        elBtnSoundToggle.textContent = "Âm thanh: BẬT 🔊";
      }
      console.log("[Dashboard] Testing emergency alarm chime over speaker...");
      triggerAlarmChime("fall");
    });
  }

  // Sound Toggle
  elBtnSoundToggle.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    if (!soundEnabled) {
      stopContinuousAlarm();
    } else {
      initAudioContext();
      playBellChimeTone(880, 0.35, 0.4); // Tiếng chuông nhẹ xác nhận đã bật loa
    }
    elBtnSoundToggle.textContent = soundEnabled ? "Âm thanh: BẬT 🔊" : "Âm thanh: TẮT 🔇";
  });

  // Fullscreen toggle
  elBtnFullscreen.addEventListener("click", () => {
    const container = document.getElementById("video-container");
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  // 10. Initialization Sequence
  connectAlertsWebSocket();
  fetchHistoricalEvents();
  pollStatus();
  pollHealth();

  // Polling intervals
  setInterval(pollStatus, 1500);
  setInterval(pollHealth, 4000);

  console.log("[Dashboard] Initialized successfully. Connected to API:", API_BASE);
})();
