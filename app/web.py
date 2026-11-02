from fastapi.responses import HTMLResponse


CHAT_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thuốc AI</title>
  <style>
    :root { --ink:#172033; --muted:#758096; --blue:#5b5cf0; --line:#e6e9f2; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); font:15px Inter,ui-sans-serif,system-ui,sans-serif;
      background:linear-gradient(135deg,#f6f7ff 0%,#eef9ff 48%,#fff 100%); }
    .app { width:min(940px,100%); height:100vh; min-height:620px; margin:auto; padding:22px;
      display:flex; flex-direction:column; }
    .shell { min-height:0; flex:1; display:flex; flex-direction:column; overflow:hidden;
      background:rgba(255,255,255,.82); border:1px solid rgba(255,255,255,.9);
      border-radius:28px; box-shadow:0 24px 70px rgba(48,62,110,.14); backdrop-filter:blur(16px); }
    header { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:20px 24px;
      border-bottom:1px solid var(--line); background:rgba(255,255,255,.7); }
    .brand { display:flex; gap:12px; align-items:center; min-width:0; }
    .logo { flex:0 0 42px; width:42px; height:42px; display:grid; place-items:center; color:#fff; font-size:22px;
      border-radius:14px; background:linear-gradient(135deg,#6869fa,#3cb8dc); box-shadow:0 8px 18px #6667e955; }
    h1 { margin:0; font-size:18px; letter-spacing:-.3px; }
    header small { display:block; margin-top:3px; color:var(--muted); font-size:12px; }
    .status { flex:0 0 auto; color:#20a36a; font-size:12px; }
    .status.offline { color:#d14343; }
    .status i { display:inline-block; width:7px; height:7px; margin-right:5px; border-radius:50%; background:#26c281; }
    .status.offline i { background:#e05252; }
    #chat { flex:1; min-height:0; overflow-y:auto; padding:28px 8%; scroll-behavior:smooth; }
    .welcome { margin:10% auto; max-width:430px; text-align:center; color:var(--muted); }
    .welcome .emoji { font-size:42px; margin-bottom:10px; }
    .welcome h2 { margin:0 0 8px; color:var(--ink); font-size:23px; }
    .welcome p { margin:0; line-height:1.55; }
    .message { display:flex; gap:10px; margin:18px 0; align-items:flex-end; animation:up .2s ease-out; }
    .message.user { justify-content:flex-end; }
    .message > div:not(.avatar) { min-width:0; max-width:90%; }
    .avatar { flex:0 0 30px; width:30px; height:30px; display:grid; place-items:center; color:#fff; font-size:14px;
      border-radius:10px; background:var(--blue); }
    .bubble { width:fit-content; max-width:100%; padding:12px 15px; line-height:1.55; white-space:pre-wrap;
      border-radius:18px 18px 18px 5px; background:#f0f2f7; overflow-wrap:anywhere; }
    .user .bubble { color:#fff; border-radius:18px 18px 5px 18px; background:linear-gradient(135deg,#6366ef,#4d8de8); }
    .user .bubble img { display:block; width:min(280px,100%); max-height:220px; margin:-2px 0 9px;
      object-fit:cover; border-radius:12px; }
    .bubble.error { color:#a12828; background:#fff1f1; }
    .meta { margin:0 4px 5px; color:#9aa3b5; font-size:10px; }
    .typing { display:inline-flex; gap:4px; padding:4px 2px; }
    .typing span { width:6px; height:6px; border-radius:50%; background:#a8b0c2; animation:blink 1.2s infinite; }
    .typing span:nth-child(2) { animation-delay:.2s; }
    .typing span:nth-child(3) { animation-delay:.4s; }
    .composer { margin:0 18px 18px; padding:10px; border:1px solid var(--line); border-radius:19px; background:#fff;
      box-shadow:0 8px 25px #3440600d; }
    .attachment { display:none; align-items:center; gap:8px; margin:0 4px 8px; padding:7px 10px; color:#526078;
      font-size:12px; background:#f5f7fb; border-radius:10px; }
    .attachment.show { display:flex; }
    .attachment img { width:35px; height:35px; object-fit:cover; border-radius:7px; }
    .attachment span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .attachment button { margin-left:auto; padding:0 5px; color:#8992a4; background:none; font-size:18px; }
    .row { display:flex; gap:8px; align-items:center; }
    textarea { flex:1; min-width:0; resize:none; height:42px; max-height:120px; overflow-y:auto;
      padding:11px 12px; border:0; outline:0; color:var(--ink); font:inherit; background:transparent; }
    .icon, #send { border:0; cursor:pointer; }
    .icon { flex:0 0 38px; width:38px; height:38px; display:grid; place-items:center; color:#6f7890;
      background:#f4f6fb; border-radius:12px; font-size:19px; }
    #send { flex:0 0 42px; width:42px; height:42px; color:#fff; border-radius:13px; background:var(--blue);
      font-size:17px; box-shadow:0 7px 15px #5b5cf044; }
    button:disabled { opacity:.55; cursor:wait; }
    .hint { margin:7px 5px 0; color:#a0a8b8; font-size:10px; }
    @keyframes up { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
    @keyframes blink { 0%,80%,100% { opacity:.25; } 40% { opacity:1; } }
    @media(max-width:600px) {
      .app { padding:0; min-height:100vh; }
      .shell { border-radius:0; border:0; }
      header { padding:16px; }
      #chat { padding:20px 14px; }
      .composer { margin:0 10px 10px; }
      .status { display:none; }
    }
  </style>
</head>
<body>
<main class="app"><section class="shell">
  <header>
    <div class="brand">
      <div class="logo">✚</div>
      <div><h1>Thuốc AI</h1><small>Trợ lý nhận diện và giải đáp thuốc</small></div>
    </div>
    <div id="status" class="status"><i></i><span>Đang kiểm tra...</span></div>
    <button id="clear" class="icon" title="Cuộc trò chuyện mới">↻</button>
  </header>
  <div id="chat">
    <div class="welcome">
      <div class="emoji">💊</div>
      <h2>Xin chào!</h2>
      <p>Chọn ảnh thuốc và đặt câu hỏi.<br>Mình sẽ cùng bạn tìm hiểu thông tin.</p>
    </div>
  </div>
  <div class="composer">
    <div id="attachment" class="attachment">
      <img id="preview" alt="">
      <span id="filename"></span>
      <button id="remove" title="Bỏ ảnh">×</button>
    </div>
    <div class="row">
      <label class="icon" title="Chọn ảnh">＋<input id="image" type="file" accept="image/*" hidden></label>
      <textarea id="question" rows="1" placeholder="Nhập câu hỏi của bạn..." autocomplete="off"></textarea>
      <button id="send" title="Gửi">➤</button>
    </div>
    <div class="hint">AI có thể mắc lỗi. Hãy dùng thông tin như tài liệu tham khảo.</div>
  </div>
</section></main>
<script>
  const API = "/api/v1";
  const chat = document.getElementById("chat");
  const image = document.getElementById("image");
  const question = document.getElementById("question");
  const send = document.getElementById("send");
  const attachment = document.getElementById("attachment");
  const preview = document.getElementById("preview");
  const filename = document.getElementById("filename");
  const remove = document.getElementById("remove");
  const clear = document.getElementById("clear");
  const status = document.getElementById("status");
  const MAX_BYTES = 10 * 1024 * 1024;
  const STORAGE_KEY = "thuocai.chat.v1";
  let history = [];
  let transcript = [];
  let busy = false;

  function welcomeHTML() {
    return '<div class="welcome"><div class="emoji">💊</div><h2>Xin chào!</h2>'
      + '<p>Chọn ảnh thuốc và đặt câu hỏi.<br>Mình sẽ cùng bạn tìm hiểu thông tin.</p></div>';
  }

  function saveConversation() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ history, transcript }));
    } catch (_) { /* storage đầy hoặc bị chặn: bỏ qua */ }
  }

  function restoreConversation() {
    let data;
    try {
      data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    } catch (_) {
      data = null;
    }
    if (!data) return;
    if (Array.isArray(data.history)) history = data.history;
    if (Array.isArray(data.transcript)) {
      let restored = 0;
      data.transcript.forEach((message) => {
        if (message && typeof message.text === "string") {
          add(message.text, message.role === "user" ? "user" : "model",
            message.error ? "error" : null, false);
          restored += 1;
        }
      });
      if (restored) chat.scrollTop = chat.scrollHeight;
    }
  }

  function add(text, role, extraClass, remember = true, imageSrc = null) {
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();
    const box = document.createElement("div");
    box.className = "message " + role;
    if (role === "model") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = "✦";
      box.appendChild(avatar);
    }
    const body = document.createElement("div");
    if (role === "model") {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = "Thuốc AI";
      body.appendChild(meta);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble" + (extraClass ? " " + extraClass : "");
    if (text === null) {
      bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    } else if (role === "user" && imageSrc) {
      const imagePreview = document.createElement("img");
      imagePreview.src = imageSrc;
      imagePreview.alt = "Ảnh thuốc đã gửi";
      bubble.appendChild(imagePreview);
      const messageText = document.createElement("div");
      messageText.textContent = text;
      bubble.appendChild(messageText);
    } else {
      bubble.textContent = text;
    }
    body.appendChild(bubble);
    box.appendChild(body);
    chat.appendChild(box);
    chat.scrollTop = chat.scrollHeight;
    if (remember && typeof text === "string") {
      transcript.push({ role, text, error: extraClass === "error" });
      saveConversation();
    }
    return box;
  }

  function clearImage() {
    if (preview.src) URL.revokeObjectURL(preview.src);
    image.value = "";
    preview.removeAttribute("src");
    filename.textContent = "";
    attachment.classList.remove("show");
  }

  function setStatus(online) {
    status.classList.toggle("offline", !online);
    status.querySelector("span").textContent = online ? "Đang hoạt động" : "Mất kết nối";
  }

  async function checkHealth() {
    try {
      const response = await fetch(API + "/health", { cache: "no-store" });
      setStatus(response.ok);
    } catch (_) {
      setStatus(false);
    }
  }

  function friendlyError(response, detail) {
    if (response.status === 413) return "Ảnh quá lớn. Vui lòng chọn ảnh nhỏ hơn 10 MB.";
    if (response.status === 415) return "Tệp bạn chọn không phải ảnh. Hãy chọn JPG, PNG hoặc WEBP.";
    if (response.status === 503 && /GEMINI_API_KEY/i.test(detail || "")) {
      return "Máy chủ chưa cấu hình GEMINI_API_KEY. Hãy đặt biến môi trường này rồi khởi động lại server.";
    }
    if (detail) return detail;
    return "Máy chủ trả lỗi " + response.status + ". Vui lòng thử lại.";
  }

  image.onchange = () => {
    const file = image.files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      clearImage();
      add("Vui lòng chọn tệp ảnh (JPG, PNG hoặc WEBP).", "model", "error", false);
      return;
    }
    if (file.size > MAX_BYTES) {
      clearImage();
      add("Ảnh quá lớn. Vui lòng chọn ảnh nhỏ hơn 10 MB.", "model", "error", false);
      return;
    }
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = URL.createObjectURL(file);
    filename.textContent = file.name;
    attachment.classList.add("show");
  };

  remove.onclick = clearImage;

  clear.onclick = () => {
    if (busy) return;
    history = [];
    transcript = [];
    try { localStorage.removeItem(STORAGE_KEY); } catch (_) { /* bỏ qua */ }
    clearImage();
    question.value = "";
    chat.innerHTML = welcomeHTML();
    question.focus();
  };

  async function ask() {
    if (busy) return;
    const text = question.value.trim();
    const file = image.files[0];
    if (!text) {
      add("Bạn hãy nhập câu hỏi trước khi gửi nhé.", "model", "error", false);
      question.focus();
      return;
    }
    busy = true;
    send.disabled = true;
    const messageImageUrl = file ? URL.createObjectURL(file) : null;
    add(text, "user", null, true, messageImageUrl);
    question.value = "";
    question.style.height = "42px";
    if (!file) {
      add("Bạn đã gửi tin nhắn, nhưng mình cần ảnh thuốc để nhận diện. "
        + "Hãy bấm dấu ＋ để chọn ảnh rồi gửi lại câu hỏi nhé.", "model", "error");
      busy = false;
      send.disabled = false;
      question.focus();
      return;
    }
    const pending = add(null, "model");
    const bubble = pending.querySelector(".bubble");
    const form = new FormData();
    form.append("question", text);
    form.append("image", file);
    form.append("history", JSON.stringify(history));
    try {
      const response = await fetch(API + "/rag-chat", { method: "POST", body: form });
      let data = {};
      try { data = await response.json(); } catch (_) { data = {}; }
      if (!response.ok) throw Error(friendlyError(response, data.detail));
      const answer = (data.answer || "").trim() || "Mình chưa có câu trả lời cho ảnh này.";
      bubble.classList.remove("error");
      bubble.textContent = answer;
      transcript.push({ role: "model", text: answer, error: false });
      history.push(
        { role: "user", parts: [{ text: text }] },
        { role: "model", parts: [{ text: answer }] }
      );
      history = history.slice(-10);
      saveConversation();
      setStatus(true);
    } catch (error) {
      const message = "Mình gặp lỗi: " + error.message;
      bubble.classList.add("error");
      bubble.textContent = message;
      transcript.push({ role: "model", text: message, error: true });
      saveConversation();
      if (!navigator.onLine) setStatus(false);
    } finally {
      busy = false;
      send.disabled = false;
      chat.scrollTop = chat.scrollHeight;
      question.focus();
    }
  }

  send.onclick = ask;
  question.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  };
  question.oninput = () => {
    question.style.height = "42px";
    question.style.height = Math.min(question.scrollHeight, 120) + "px";
  };

  restoreConversation();
  checkHealth();
</script>
</body></html>"""


def chat_page() -> HTMLResponse:
    return HTMLResponse(CHAT_PAGE)
