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
    header { display:flex; align-items:center; justify-content:space-between; padding:20px 24px;
      border-bottom:1px solid var(--line); background:rgba(255,255,255,.7); }
    .brand { display:flex; gap:12px; align-items:center; }
    .logo { width:42px; height:42px; display:grid; place-items:center; color:#fff; font-size:22px;
      border-radius:14px; background:linear-gradient(135deg,#6869fa,#3cb8dc); box-shadow:0 8px 18px #6667e955; }
    h1 { margin:0; font-size:18px; letter-spacing:-.3px; }
    header small { display:block; margin-top:3px; color:var(--muted); font-size:12px; }
    .status { color:#20a36a; font-size:12px; } .status i { display:inline-block; width:7px; height:7px;
      margin-right:5px; border-radius:50%; background:#26c281; }
    #chat { flex:1; min-height:0; overflow-y:auto; padding:28px 8%; scroll-behavior:smooth; }
    .welcome { margin:12% auto; max-width:430px; text-align:center; color:var(--muted); }
    .welcome .emoji { font-size:42px; margin-bottom:10px; } .welcome h2 { margin:0 0 8px; color:var(--ink); font-size:23px; }
    .welcome p { margin:0; line-height:1.55; }
    .message { display:flex; gap:10px; margin:18px 0; align-items:flex-end; animation:up .2s ease-out; }
    .message.user { justify-content:flex-end; } .avatar { flex:0 0 30px; width:30px; height:30px;
      display:grid; place-items:center; color:#fff; font-size:14px; border-radius:10px; background:var(--blue); }
    .bubble { max-width:min(620px,82%); padding:12px 15px; line-height:1.55; white-space:pre-wrap; border-radius:18px 18px 18px 5px;
      background:#f0f2f7; } .user .bubble { color:#fff; border-radius:18px 18px 5px 18px; background:linear-gradient(135deg,#6366eF,#4d8de8); }
    .meta { margin:0 4px 5px; color:#9aa3b5; font-size:10px; }
    .composer { margin:0 18px 18px; padding:10px; border:1px solid var(--line); border-radius:19px; background:#fff; box-shadow:0 8px 25px #3440600d; }
    .attachment { display:none; align-items:center; gap:8px; margin:0 4px 8px; padding:7px 10px; color:#526078; font-size:12px; background:#f5f7fb; border-radius:10px; }
    .attachment.show { display:flex; } .attachment img { width:35px; height:35px; object-fit:cover; border-radius:7px; }
    .attachment button { margin-left:auto; padding:0 5px; color:#8992a4; background:none; font-size:18px; }
    .row { display:flex; gap:8px; align-items:center; } textarea { flex:1; min-width:0; resize:none; height:42px;
      padding:11px 12px; border:0; outline:0; color:var(--ink); font:inherit; background:transparent; }
    .icon, #send { border:0; cursor:pointer; } .icon { width:38px; height:38px; color:#6f7890; background:#f4f6fb; border-radius:12px; font-size:19px; }
    #send { width:42px; height:42px; color:#fff; border-radius:13px; background:var(--blue); font-size:17px; box-shadow:0 7px 15px #5b5cf044; }
    button:disabled { opacity:.55; cursor:wait; } .hint { margin:7px 5px 0; color:#a0a8b8; font-size:10px; }
    @keyframes up { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
    @media(max-width:600px) { .app { padding:0; min-height:100vh; } .shell { border-radius:0; } header { padding:16px; }
      #chat { padding:20px 14px; } .composer { margin:0 10px 10px; } .status { display:none; } }
  </style>
</head>
<body>
<main class="app"><section class="shell">
  <header><div class="brand"><div class="logo">✚</div><div><h1>Thuốc AI</h1><small>Trợ lý nhận diện và giải đáp thuốc</small></div></div>
    <div class="status"><i></i>Đang hoạt động</div><button id="clear" class="icon" title="Cuộc trò chuyện mới">↻</button>
  </header>
  <div id="chat"><div class="welcome"><div class="emoji">💊</div><h2>Xin chào!</h2><p>Chọn ảnh thuốc và đặt câu hỏi.<br>Mình sẽ cùng bạn tìm hiểu thông tin.</p></div></div>
  <div class="composer"><div id="attachment" class="attachment"><img id="preview"><span id="filename"></span><button id="remove" title="Bỏ ảnh">×</button></div>
    <div class="row"><label class="icon" title="Chọn ảnh">＋<input id="image" type="file" accept="image/*" hidden></label>
      <textarea id="question" rows="1" placeholder="Nhập câu hỏi của bạn..." autocomplete="off"></textarea><button id="send" title="Gửi">➤</button></div>
    <div class="hint">AI có thể mắc lỗi. Hãy dùng thông tin như tài liệu tham khảo.</div>
  </div>
</section></main>
<script>
  const chat=document.getElementById("chat"), image=document.getElementById("image"), question=document.getElementById("question");
  const send=document.getElementById("send"), attachment=document.getElementById("attachment"), preview=document.getElementById("preview");
  const filename=document.getElementById("filename"), remove=document.getElementById("remove"), clear=document.getElementById("clear");
  let history=[];
  function add(text,role){const box=document.createElement("div");box.className="message "+role;
    if(role==="model"){const avatar=document.createElement("div");avatar.className="avatar";avatar.textContent="✦";box.appendChild(avatar);}
    const body=document.createElement("div");if(role==="model"){const meta=document.createElement("div");meta.className="meta";meta.textContent="Thuốc AI";body.appendChild(meta);}
    const bubble=document.createElement("div");bubble.className="bubble";bubble.textContent=text;body.appendChild(bubble);box.appendChild(body);chat.appendChild(box);chat.scrollTop=chat.scrollHeight;}
  image.onchange=()=>{const file=image.files[0];if(!file)return;preview.src=URL.createObjectURL(file);filename.textContent=file.name;attachment.classList.add("show");};
  remove.onclick=()=>{image.value="";attachment.classList.remove("show");};
  clear.onclick=()=>{history=[];chat.innerHTML='<div class="welcome"><div class="emoji">💊</div><h2>Xin chào!</h2><p>Chọn ảnh thuốc và đặt câu hỏi.<br>Mình sẽ cùng bạn tìm hiểu thông tin.</p></div>';};
  async function ask(){const text=question.value.trim(),file=image.files[0];if(!text||!file){add("Hãy chọn ảnh thuốc và nhập câu hỏi nhé.","model");return;}
    add(text,"user");question.value="";send.disabled=true;const form=new FormData();form.append("question",text);form.append("image",file);form.append("history",JSON.stringify(history));
    const loading="Đang xem ảnh và suy nghĩ...";add(loading,"model");const pending=chat.lastElementChild;
    try{const response=await fetch("/api/v1/gemini/vision",{method:"POST",body:form}),data=await response.json();if(!response.ok)throw Error(data.detail||"Không thể nhận câu trả lời.");
      pending.querySelector(".bubble").textContent=data.answer;history.push({role:"user",parts:[{text:text}]},{role:"model",parts:[{text:data.answer}]});
    }catch(error){pending.querySelector(".bubble").textContent="Mình gặp lỗi: "+error.message;}finally{send.disabled=false;question.focus();}}
  send.onclick=ask;question.onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask();}};
</script></body></html>"""


def chat_page() -> HTMLResponse:
    return HTMLResponse(CHAT_PAGE)
