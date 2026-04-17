(function () {
  "use strict";

  var MAX_LOG_LINES = 2000;
  var WS_PATH = "/ws";
  var WS_ENDPOINT = "ws://127.0.0.1:8000/ws";

  var frame = document.getElementById("frame");
  var frameWait = document.getElementById("frame-wait");
  var terminal = document.getElementById("terminal");
  var messagesEl = document.getElementById("messages");
  var statusEl = document.getElementById("status");
  var input = document.getElementById("cmd-input");
  var sendBtn = document.getElementById("send-btn");
  var viewportWrap = document.getElementById("viewport-wrap");
  var micBtn = document.getElementById("mic-btn");
  var speakBtn = document.getElementById("speak-btn");
  var workspace = document.querySelector(".workspace");
  var chatPane = document.getElementById("chat-pane");
  var chatResizer = document.getElementById("chat-resizer");

  var logLines = [];
  var ws = null;
  var reconnectAttempt = 0;
  var reconnectTimer = null;
  var firstFrameReceived = false;
  var streamWatchdog = null;

  var dragging = false;
  var dragMode = null;
  var lastX = 0;
  var lastY = 0;

  var busy = false;
  var connected = false;
  var previousBusy = false;
  var lastUserCommand = "";
  var voicesLoaded = false;
  var pendingRobotCommand = false;

  var voiceEnabled = true;
  var recognition = null;
  var recognitionSupported = false;
  var isListening = false;
  var resizingChat = false;

  function wsUrl() {
    // Force a stable endpoint. Some embedded webviews/pages rewrite location.host,
    // causing perpetual "connecting" even when backend is healthy.
    return WS_ENDPOINT;
  }

  function appendLog(line) {
    logLines.push(line);
    if (logLines.length > MAX_LOG_LINES) {
      logLines = logLines.slice(-MAX_LOG_LINES);
    }
    terminal.textContent = logLines.join("\n");
    terminal.scrollTop = terminal.scrollHeight;
  }

  function addMessage(role, text) {
    var div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function speak(text) {
    if (!voiceEnabled || !window.speechSynthesis || !text) return;
    try {
      window.speechSynthesis.cancel();
      var utter = new SpeechSynthesisUtterance(text);
      var voices = window.speechSynthesis.getVoices() || [];
      var preferred = null;
      var preferredVoicePatterns = [
        "jenny",
        "aria",
        "zira",
        "sara",
        "emma",
        "olivia",
        "female"
      ];
      // Force female voice preference with natural/neural variants first.
      for (var p = 0; p < preferredVoicePatterns.length && !preferred; p++) {
        var pattern = preferredVoicePatterns[p];
        for (var i = 0; i < voices.length; i++) {
          var n = (voices[i].name || "").toLowerCase();
          if (n.indexOf(pattern) >= 0 && (n.indexOf("natural") >= 0 || n.indexOf("neural") >= 0)) {
            preferred = voices[i];
            break;
          }
        }
      }
      for (var p2 = 0; p2 < preferredVoicePatterns.length && !preferred; p2++) {
        var pattern2 = preferredVoicePatterns[p2];
        for (var j = 0; j < voices.length; j++) {
          var n2 = (voices[j].name || "").toLowerCase();
          if (n2.indexOf(pattern2) >= 0) {
            preferred = voices[j];
            break;
          }
        }
      }
      if (!preferred && voices.length > 0) preferred = voices[0];
      if (preferred) utter.voice = preferred;
      utter.rate = 0.92;
      utter.pitch = 1.12;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
    } catch (e) {
      appendLog("[Voice] Speech output failed: " + String(e));
    }
  }

  function assistantSay(text, alsoSpeak) {
    addMessage("assistant", text);
    if (alsoSpeak) speak(text);
  }

  function actionLabel(action) {
    var map = {
      move_to_object: "move to the object",
      hover_over_object: "hover above the object",
      close_gripper: "close the gripper",
      open_gripper: "open the gripper",
      move_home: "return to home pose",
      emote_wave: "wave",
      emote_dance: "dance",
      emote_nod: "nod",
      emote_yes: "nod yes",
      emote_no: "shake no",
      emote_shake_no: "shake no",
      emote_clap: "clap",
      emote_rotate_wrist: "rotate wrist",
      emote_bow: "bow",
      emote_celebrate: "celebrate"
    };
    return map[action] || action.replace(/_/g, " ");
  }

  function planToSentence(plan) {
    if (!Array.isArray(plan) || plan.length === 0) {
      return "I could not map that to a safe action plan yet. Could you rephrase it a bit?";
    }
    var actions = [];
    for (var i = 0; i < plan.length && i < 4; i++) {
      var step = plan[i] || {};
      var a = actionLabel(step.action || "step");
      if (step.target_name) {
        a += " on " + String(step.target_name).replace(/_/g, " ");
      }
      actions.push(a);
    }
    var suffix = plan.length > 4 ? ", then I will continue with the remaining steps." : ".";
    var starters = [
      "Sounds good, I will ",
      "Alright, I will ",
      "Got it, I will ",
      "Great, I will "
    ];
    var start = starters[Math.floor(Math.random() * starters.length)];
    return start + actions.join(", then ") + suffix;
  }

  function classifyUserText(text) {
    var t = String(text || "").trim().toLowerCase();
    if (!t) return { kind: "empty", reply: "" };
    if (/^(hi|hello|hey|yo|hola)\b/.test(t)) {
      return { kind: "chat", reply: "Hey! Good to see you." };
    }
    if (/are you ready|ready\??$|you ready/.test(t)) {
      return { kind: "chat", reply: "Yes, I am ready. What should we do first?" };
    }
    if (/how are you|how's it going/.test(t)) {
      return { kind: "chat", reply: "I am doing great. What do you want me to do next?" };
    }
    if (/thank(s| you)?/.test(t)) {
      return { kind: "chat", reply: "You are welcome. Happy to help." };
    }
    // Treat these as robot-action requests.
    return { kind: "command", reply: "" };
  }

  function setStatus(connectedNow, busyNow) {
    statusEl.className = "status";
    if (!connectedNow) {
      statusEl.classList.add("bad");
      statusEl.textContent = "Disconnected · reconnecting...";
    } else if (busyNow) {
      statusEl.classList.add("busy");
      statusEl.textContent = "Running...";
    } else {
      statusEl.textContent = "Ready";
    }
    input.disabled = !connectedNow || busyNow;
    sendBtn.disabled = !connectedNow || busyNow;
  }

  function startStreamWatchdog() {
    if (streamWatchdog) clearTimeout(streamWatchdog);
    streamWatchdog = setTimeout(function () {
      if (connected && !firstFrameReceived) {
        appendLog("[Stream] Connected but no video frames yet. Check the VLA Server window for renderer errors.");
        statusEl.className = "status bad";
        statusEl.textContent = "Connected, waiting for stream...";
      }
    }, 8000);
  }

  function sendWs(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  function pointerToUv(clientX, clientY) {
    var img = frame;
    var container = viewportWrap;
    var cr = container.getBoundingClientRect();
    var iw = img.naturalWidth || 640;
    var ih = img.naturalHeight || 480;
    var cw = cr.width;
    var ch = cr.height;
    var scale = Math.min(cw / iw, ch / ih);
    var dw = iw * scale;
    var dh = ih * scale;
    var ox = cr.left + (cw - dw) / 2;
    var oy = cr.top + (ch - dh) / 2;
    var x = clientX - ox;
    var y = clientY - oy;
    var u = x / dw;
    var v = y / dh;
    return { u: Math.min(1, Math.max(0, u)), v: Math.min(1, Math.max(0, v)) };
  }

  function onPointerDown(e) {
    if (!frame.naturalWidth) return;

    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;

    if ((e.button === 0 && e.shiftKey) || (e.button === 0 && e.ctrlKey)) {
      dragMode = "object";
      var uv = pointerToUv(e.clientX, e.clientY);
      sendWs({ type: "pointer", kind: "down", u: uv.u, v: uv.v });
    } else if (e.button === 2 || e.button === 1) {
      dragMode = "cameraPan";
    } else {
      dragMode = "cameraOrbit";
    }

    try {
      e.target.setPointerCapture(e.pointerId);
    } catch (err) {}
    e.preventDefault();
  }

  function onPointerMove(e) {
    if (!dragging || !frame.naturalWidth) return;

    var dx = e.clientX - lastX;
    var dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (dragMode === "object") {
      var uv = pointerToUv(e.clientX, e.clientY);
      sendWs({ type: "pointer", kind: "move", u: uv.u, v: uv.v });
    } else if (dragMode === "cameraOrbit") {
      sendWs({ type: "camera", op: "orbit", dx: -dx, dy: dy });
    } else if (dragMode === "cameraPan") {
      sendWs({ type: "camera", op: "pan", dx: dx, dy: dy });
    }
  }

  function onPointerUp(e) {
    if (!dragging) return;
    if (dragMode === "object") {
      sendWs({ type: "pointer", kind: "up", u: 0, v: 0 });
    }
    dragging = false;
    dragMode = null;
    try {
      e.target.releasePointerCapture(e.pointerId);
    } catch (err) {}
  }

  function onWheel(e) {
    if (!frame.naturalWidth) return;
    var raw = e.deltaY / 120;
    var delta = Math.max(-2, Math.min(2, raw));
    sendWs({ type: "camera", op: "zoom", delta: delta });
    e.preventDefault();
  }

  function connect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    try {
      ws = new WebSocket(wsUrl());
    } catch (e) {
      appendLog("[WebSocket] Failed to create socket: " + String(e));
      connected = false;
      setStatus(connected, busy);
      var d = Math.min(30000, 500 * Math.pow(2, reconnectAttempt));
      reconnectAttempt += 1;
      reconnectTimer = setTimeout(connect, d);
      return;
    }

    ws.onopen = function () {
      reconnectAttempt = 0;
      connected = true;
      firstFrameReceived = false;
      setStatus(connected, busy);
      startStreamWatchdog();
    };

    ws.onclose = function () {
      connected = false;
      setStatus(connected, busy);
      var delay = Math.min(30000, 500 * Math.pow(2, reconnectAttempt));
      reconnectAttempt += 1;
      reconnectTimer = setTimeout(connect, delay);
    };

    ws.onerror = function () {
      if (ws) ws.close();
    };

    ws.onmessage = function (ev) {
      try {
        var msg = JSON.parse(ev.data);
        if (msg.type === "frame" && msg.data) {
          firstFrameReceived = true;
          if (streamWatchdog) {
            clearTimeout(streamWatchdog);
            streamWatchdog = null;
          }
          frame.src = "data:image/jpeg;base64," + msg.data;
          frame.hidden = false;
          frameWait.hidden = true;
        } else if (msg.type === "log" && msg.line !== undefined) {
          appendLog(msg.line);
        } else if (msg.type === "error" && msg.message) {
          appendLog(msg.message);
          assistantSay("I hit an issue while executing that. Please check the terminal panel.", true);
        } else if (msg.type === "plan" && msg.json !== undefined) {
          if (pendingRobotCommand) {
            assistantSay(planToSentence(msg.json), true);
          }
        } else if (msg.type === "busy") {
          previousBusy = busy;
          busy = !!msg.value;
          setStatus(connected, busy);
          // Do not spam a repetitive completion message every time.
        }
      } catch (e) {}
    };
  }

  function isCompactLayout() {
    return window.matchMedia("(max-width: 900px)").matches;
  }

  function clampChatWidth(widthPx) {
    if (!workspace) return widthPx;
    var totalWidth = workspace.getBoundingClientRect().width;
    var minWidth = 280;
    var maxWidth = Math.min(700, Math.max(minWidth, totalWidth - 320));
    return Math.max(minWidth, Math.min(maxWidth, widthPx));
  }

  function setChatWidth(widthPx) {
    if (!workspace || isCompactLayout()) return;
    var safeWidth = clampChatWidth(widthPx);
    workspace.style.setProperty("--chat-width", safeWidth + "px");
    try {
      window.localStorage.setItem("chatPanelWidth", String(Math.round(safeWidth)));
    } catch (err) {}
  }

  function restoreChatWidth() {
    if (!workspace || isCompactLayout()) return;
    var parsed = NaN;
    try {
      parsed = parseInt(window.localStorage.getItem("chatPanelWidth") || "", 10);
    } catch (err) {}
    if (!isNaN(parsed)) {
      setChatWidth(parsed);
      return;
    }
    if (chatPane) {
      setChatWidth(chatPane.getBoundingClientRect().width || 340);
    }
  }

  function onChatResizeStart(e) {
    if (!chatResizer || isCompactLayout()) return;
    resizingChat = true;
    workspace.classList.add("resizing");
    if (chatResizer.setPointerCapture) {
      chatResizer.setPointerCapture(e.pointerId);
    }
    e.preventDefault();
  }

  function onChatResizeMove(e) {
    if (!resizingChat || !workspace || isCompactLayout()) return;
    var rect = workspace.getBoundingClientRect();
    var nextWidth = rect.right - e.clientX;
    setChatWidth(nextWidth);
  }

  function onChatResizeEnd(e) {
    if (!resizingChat) return;
    resizingChat = false;
    workspace.classList.remove("resizing");
    if (chatResizer && chatResizer.releasePointerCapture) {
      try {
        chatResizer.releasePointerCapture(e.pointerId);
      } catch (err) {}
    }
  }

  function sendCommandFromText(text) {
    var t = String(text || "").trim();
    if (!t || busy || !connected) return;
    lastUserCommand = t;
    addMessage("user", t);
    var classification = classifyUserText(t);
    if (classification.kind === "chat") {
      pendingRobotCommand = false;
      assistantSay(classification.reply, true);
      return;
    }
    pendingRobotCommand = true;
    sendWs({ type: "command", text: t });
    var replies = [
      "Got it, I am on it.",
      "Perfect, I will handle that now.",
      "Okay, I am taking care of it.",
      "Sure thing, I will do that now."
    ];
    assistantSay(replies[Math.floor(Math.random() * replies.length)], true);
  }

  function sendCommand() {
    var t = input.value.trim();
    if (!t || busy || !connected) return;
    input.value = "";
    sendCommandFromText(t);
  }

  function updateMicButton() {
    if (!recognitionSupported) {
      micBtn.textContent = "🚫";
      micBtn.title = "Voice input unavailable";
      micBtn.disabled = true;
      return;
    }
    micBtn.disabled = false;
    micBtn.textContent = "🎤";
    micBtn.title = isListening ? "Stop voice input" : "Start voice input";
    micBtn.classList.toggle("active", isListening);
  }

  function updateSpeakButton() {
    speakBtn.textContent = voiceEnabled ? "Speak: On" : "Speak: Off";
    speakBtn.classList.toggle("active", voiceEnabled);
  }

  function initVoiceInput() {
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      recognitionSupported = false;
      updateMicButton();
      return;
    }
    recognitionSupported = true;
    recognition = new SR();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = function () {
      isListening = true;
      updateMicButton();
    };

    recognition.onresult = function (event) {
      var text = "";
      for (var i = event.resultIndex; i < event.results.length; i++) {
        text += event.results[i][0].transcript;
      }
      input.value = text.trim();
      var latest = event.results[event.results.length - 1];
      if (latest && latest.isFinal) {
        sendCommand();
      }
    };

    recognition.onerror = function (event) {
      appendLog("[Voice] Speech recognition error: " + String(event.error || "unknown"));
      isListening = false;
      updateMicButton();
    };

    recognition.onend = function () {
      isListening = false;
      updateMicButton();
    };

    updateMicButton();
  }

  function toggleMic() {
    if (!recognitionSupported || !recognition) return;
    try {
      if (isListening) recognition.stop();
      else recognition.start();
    } catch (e) {
      appendLog("[Voice] Unable to start microphone: " + String(e));
    }
  }

  sendBtn.addEventListener("click", sendCommand);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") sendCommand();
  });

  frame.addEventListener("pointerdown", onPointerDown);
  frame.addEventListener("pointermove", onPointerMove);
  frame.addEventListener("pointerup", onPointerUp);
  frame.addEventListener("pointercancel", onPointerUp);
  frame.addEventListener("wheel", onWheel, { passive: false });
  frame.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  if (chatResizer) {
    chatResizer.addEventListener("pointerdown", onChatResizeStart);
    window.addEventListener("pointermove", onChatResizeMove);
    window.addEventListener("pointerup", onChatResizeEnd);
    window.addEventListener("pointercancel", onChatResizeEnd);
    window.addEventListener("resize", restoreChatWidth);
  }

  micBtn.addEventListener("click", toggleMic);
  speakBtn.addEventListener("click", function () {
    voiceEnabled = !voiceEnabled;
    updateSpeakButton();
    if (!voiceEnabled && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  });

  initVoiceInput();
  if (window.speechSynthesis && !voicesLoaded) {
    voicesLoaded = true;
    // Trigger voice list population in Chromium-based browsers.
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = function () {
      window.speechSynthesis.getVoices();
    };
  }
  updateSpeakButton();
  restoreChatWidth();
  connect();
})();

