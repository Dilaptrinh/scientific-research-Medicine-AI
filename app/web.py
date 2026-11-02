from fastapi.responses import HTMLResponse


CHAT_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medicine AI Chat</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #f3f4f6; color: #172033; font: 16px system-ui, sans-serif; }
    main { width: min(760px, 100%); min-height: 100vh; margin: auto; padding: 24px 14px; display: flex; flex-direction: column; }
    h1 { margin: 0 0 16px; font-size: 24px; }
    #chat { flex: 1; min-height: 420px; padding: 14px; overflow-y: auto; background: white; border: 1px solid #d8dce5; border-radius: 12px; }
    .message { max-width: 85%; margin: 10px 0; padding: 10px 13px; white-space: pre-wrap; border-radius: 12px; line-height: 1.45; }
    .user { margin-left: auto; background: #dbeafe; }
    .model { background: #eef0f4; }
    .controls { display: flex; gap: 8px; margin-top: 12px; }
    input[type=file] { max-width: 230px; }
    input[type=text] { flex: 1; min-width: 0; padding: 11px; border: 1px solid #bcc3d0; border-radius: 8px; }
    button { padding: 10px 16px; color: white; background: #2563eb; border: 0; border-radius: 8px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    small { color: #5d6677; }
  </style>
</head>
<body>
<main>
  <h1>Medicine AI Chat</h1>
  <small>Chọn ảnh thuốc, sau đó hỏi nhiều câu liên tiếp.</small>
  <div id="chat"></div>
  <div class="controls">
    <input id="image" type="file" accept="image/*">
    <input id="question" type="text" placeholder="Ví dụ: Thuốc này là gì?" autocomplete="off">
    <button id="send">Gửi</button>
  </div>
</main>
<script>
  const chat = document.getElementById("chat");
  const image = document.getElementById("image");
  const question = document.getElementById("question");
  const send = document.getElementById("send");
  let history = [];

  function addMessage(text, role) {
    const item = document.createElement("div");
    item.className = "message " + role;
    item.textContent = text;
    chat.appendChild(item);
    chat.scrollTop = chat.scrollHeight;
  }

  async function ask() {
    const text = question.value.trim();
    const file = image.files[0];
    if (!text || !file) {
      addMessage("Hãy chọn ảnh và nhập câu hỏi.", "model");
      return;
    }
    addMessage(text, "user");
    question.value = "";
    send.disabled = true;
    const form = new FormData();
    form.append("question", text);
    form.append("image", file);
    form.append("history", JSON.stringify(history));
    try {
      const response = await fetch("/api/v1/gemini/vision", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể nhận câu trả lời.");
      addMessage(data.answer, "model");
      history.push({ role: "user", parts: [{ text }] });
      history.push({ role: "model", parts: [{ text: data.answer }] });
    } catch (error) {
      addMessage("Lỗi: " + error.message, "model");
    } finally {
      send.disabled = false;
      question.focus();
    }
  }

  send.addEventListener("click", ask);
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter") ask();
  });
</script>
</body>
</html>"""


def chat_page() -> HTMLResponse:
    return HTMLResponse(CHAT_PAGE)
